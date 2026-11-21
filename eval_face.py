"""人脸验证的量化评估：FAR/FRR/EER/ROC AUC + 质量门控与多帧融合消融。

三种数据来源：
    python eval_face.py --data-dir D:/faces            # D:/faces/<person>/*.jpg
    python eval_face.py --lfw --people 40              # 自动下载 LFW（缓存到 sklearn 目录）
    python eval_face.py --data-dir D:/faces --no-degrade

评估协议：
    每个人按固定种子切分「注册图」与「探针图」；注册图分别构造单帧模板和融合模板，
    探针与本人模板的相似度记为正例，与其他所有人模板的相似度记为负例，再扫阈值。
    质量门控配置会额外统计探针拒识率与端到端 FRR（拒识算本人失败）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mods.face_eval import verification_summary  # noqa: E402
from mods.face_quality import fuse_features  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def unit(vec):
    """比对前统一 L2 归一化，和线上 matching.cosine_similarity 保持一致。"""
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-8 else None


@dataclass
class Sample:
    person: str
    name: str
    frame: np.ndarray | None
    feature: np.ndarray | None = None
    passed: bool = False
    issues: tuple = ()


def collect_probe_paths(samples, enroll, seed):
    """按固定种子切出注册集与探针集；样本不足的人直接跳过。"""
    rng = np.random.default_rng(seed)
    splits = {}
    for person, rows in samples.items():
        usable = [row for row in rows if row.feature is not None]
        if len(usable) < enroll + 1:
            continue
        order = rng.permutation(len(usable))
        splits[person] = (
            [usable[i] for i in order[:enroll]],
            [usable[i] for i in order[enroll:]],
        )
    return splits


def build_templates(splits, gate: bool, fused: bool):
    """构造模板；gate=True 时只用通过质量门控的注册帧。"""
    templates = {}
    for person, (enrolled, _probes) in splits.items():
        usable = [row for row in enrolled if row.feature is not None and (row.passed or not gate)]
        if not usable:
            continue
        if fused:
            template = fuse_features([row.feature for row in usable])
        else:
            template = fuse_features([usable[0].feature])
        if template is not None:
            templates[person] = template
    return templates


def score_pairs(splits, templates, gate):
    """正例：探针 vs 本人模板；负例：探针 vs 其他所有人模板。"""
    genuine, impostor = [], []
    rejected_genuine = 0
    total_genuine_probes = 0
    accepted_genuine = 0

    for person, (_enrolled, probes) in splits.items():
        template = templates.get(person)
        if template is None:
            continue
        others = [t for name, t in templates.items() if name != person]
        for row in probes:
            total_genuine_probes += 1
            if row.feature is None or (gate and not row.passed):
                rejected_genuine += 1
                continue
            accepted_genuine += 1
            score = float(np.dot(template, row.feature))
            genuine.append(score)
            for other in others:
                impostor.append(float(np.dot(other, row.feature)))

    return {
        "genuine": genuine,
        "impostor": impostor,
        "rejected_genuine": rejected_genuine,
        "total_genuine_probes": total_genuine_probes,
        "accepted_genuine": accepted_genuine,
    }


def evaluate_config(splits, gate, fused, threshold):
    templates = build_templates(splits, gate=gate, fused=fused)
    if len(templates) < 2:
        raise SystemExit("可用身份不足 2 人，无法计算冒名对；请增加数据或降低 --enroll")
    scored = score_pairs(splits, templates, gate)
    summary = verification_summary(scored["genuine"], scored["impostor"], threshold=threshold)
    summary["identities"] = len(templates)
    summary["rejected_genuine"] = scored["rejected_genuine"]
    summary["total_genuine_probes"] = scored["total_genuine_probes"]
    summary["reject_rate"] = (
        scored["rejected_genuine"] / scored["total_genuine_probes"]
        if scored["total_genuine_probes"]
        else 0.0
    )
    eer_t = summary["eer_threshold"]
    accepted_fail = sum(1 for s in scored["genuine"] if s <= eer_t)
    summary["e2e_frr_at_eer"] = (
        (scored["rejected_genuine"] + accepted_fail) / scored["total_genuine_probes"]
        if scored["total_genuine_probes"]
        else 0.0
    )
    return summary


def evaluate_protected(splits, gate, fused, bits, threshold, seed):
    """可撤销模板的验证指标 + 吊销 + 跨系统不可关联三组证据。

    注册模板用每个人各自的种子投影；探针用同一颗种子投影后算位置一致率。
    另外两组实验分别换掉种子：
        revocation    同一人重新签发种子，旧模板与新人脸的一致率应回落到 ≈0.5
        cross_system  同一人在两套系统（不同种子）里的模板互相不可关联
    """
    from mods.template_protection import hamming_similarity, protect_feature

    rng = np.random.default_rng(seed)
    raw_templates = build_templates(splits, gate=gate, fused=fused)
    if len(raw_templates) < 2:
        raise SystemExit("可用身份不足 2 人，无法计算冒名对；请增加数据或降低 --enroll")

    seeds = {person: rng.bytes(32) for person in raw_templates}
    templates = {}
    for person, raw in raw_templates.items():
        template = protect_feature(raw, seeds[person], bits=bits)
        if template is not None:
            templates[person] = template

    genuine, impostor = [], []
    rejected_genuine = 0
    total_genuine_probes = 0
    for person, (_enrolled, probes) in splits.items():
        template = templates.get(person)
        if template is None:
            continue
        others = [item for name, item in templates.items() if name != person]
        for row in probes:
            total_genuine_probes += 1
            if row.feature is None or (gate and not row.passed):
                rejected_genuine += 1
                continue
            probe = protect_feature(row.feature, seeds[person], bits=bits)
            if probe is None:
                continue
            genuine.append(hamming_similarity(template, probe))
            for other in others:
                impostor.append(hamming_similarity(other, probe))

    if not genuine or not impostor:
        raise SystemExit("可撤销模板没有产生有效比对对，请检查数据量")

    summary = verification_summary(genuine, impostor, threshold=threshold)
    summary["identities"] = len(templates)
    #: 与未保护配置同口径：被质量门控拒识的探针也算本人失败
    summary["rejected_genuine"] = rejected_genuine
    summary["total_genuine_probes"] = total_genuine_probes
    summary["reject_rate"] = (
        rejected_genuine / total_genuine_probes if total_genuine_probes else 0.0
    )
    accepted_fail = sum(1 for score in genuine if score <= summary["eer_threshold"])
    summary["e2e_frr_at_eer"] = (
        (rejected_genuine + accepted_fail) / total_genuine_probes
        if total_genuine_probes
        else 0.0
    )

    revoked_scores = []
    cross_scores = []
    for person, raw in raw_templates.items():
        old = templates.get(person)
        if old is None:
            continue
        renewed = protect_feature(raw, rng.bytes(32), bits=bits)
        cross = protect_feature(raw, rng.bytes(32), bits=bits)
        if renewed is not None:
            revoked_scores.append(hamming_similarity(old, renewed))
        if cross is not None:
            cross_scores.append(hamming_similarity(old, cross))

    summary["revocation"] = {
        "templates": len(revoked_scores),
        "mean_similarity": round(float(np.mean(revoked_scores)), 4),
        "max_similarity": round(float(np.max(revoked_scores)), 4),
        "accepted_at_threshold": int(sum(score > threshold for score in revoked_scores)),
    }
    summary["cross_system"] = {
        "pairs": len(cross_scores),
        "mean_similarity": round(float(np.mean(cross_scores)), 4),
        "max_similarity": round(float(np.max(cross_scores)), 4),
    }
    return summary


def degrade(frame, kind):
    """模拟真实终端常见退化：轻微失焦、光线不足、头部倾斜。

    模糊核按画面长边的 4% 估计，保证不同分辨率的数据集得到相近的相对模糊程度。
    """
    if kind == "blur":
        long_side = max(frame.shape[:2])
        ksize = max(3, int(round(long_side * 0.04)) | 1)
        return cv2.GaussianBlur(frame, (ksize, ksize), 0)
    if kind == "dark":
        return np.clip(frame.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if kind == "rotate":
        height, width = frame.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 20.0, 1.0)
        return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return frame


def evaluate_degradation(splits, threshold, limit, seed):
    """对探针做模糊/暗光/旋转，比较有/无质量门控的本人失败率与冒名放行率。"""
    from mods.face_detector import extract_face_sample

    rng = np.random.default_rng(seed)
    templates = {}
    for person, (enrolled, _rows) in splits.items():
        template = fuse_features([row.feature for row in enrolled if row.passed])
        if template is None:
            template = fuse_features([row.feature for row in enrolled])
        if template is not None:
            templates[person] = template

    probes = []
    for person, (_enrolled, rows) in splits.items():
        own_template = templates.get(person)
        if own_template is None:
            continue
        others = [t for name, t in templates.items() if name != person]
        if not others:
            continue
        for row in rows:
            if row.frame is not None:
                probes.append((own_template, others, row.frame))
    if len(probes) > limit:
        picks = rng.choice(len(probes), size=limit, replace=False)
        probes = [probes[i] for i in picks]

    report = {}
    for kind in ("blur", "dark", "rotate"):
        states = {
            "gated": {"total": 0, "rejected": 0, "genuine_fail": 0,
                      "far_hits": 0, "far_total": 0, "scores": []},
            "ungated": {"total": 0, "rejected": 0, "genuine_fail": 0,
                        "far_hits": 0, "far_total": 0, "scores": []},
        }
        for template, others, frame in probes:
            result = extract_face_sample(degrade(frame, kind), quality_gate=False)
            for state in states.values():
                state["total"] += 1

            if result is None or result[0] is None:
                for state in states.values():
                    state["rejected"] += 1
                continue

            feature, quality = result
            feature = unit(feature)
            if feature is None:
                for state in states.values():
                    state["rejected"] += 1
                continue
            own_score = float(np.dot(template, feature))
            impostor_scores = [float(np.dot(other, feature)) for other in others]

            #: 有门控：质量不过关直接拒识，不做任何比对
            if quality.passed:
                gated = states["gated"]
                gated["scores"].append(own_score)
                gated["genuine_fail"] += int(own_score <= threshold)
                gated["far_hits"] += sum(1 for score in impostor_scores if score > threshold)
                gated["far_total"] += len(impostor_scores)
            else:
                states["gated"]["rejected"] += 1

            ungated = states["ungated"]
            ungated["scores"].append(own_score)
            ungated["genuine_fail"] += int(own_score <= threshold)
            ungated["far_hits"] += sum(1 for score in impostor_scores if score > threshold)
            ungated["far_total"] += len(impostor_scores)

        def _pack(state):
            total = state["total"] or 1
            return {
                "probes": state["total"],
                "reject_rate": round(state["rejected"] / total, 4),
                #: 本人失败率含"被门控拒识"，等价于用户眼里的端到端失败
                "frr_total": round((state["rejected"] + state["genuine_fail"]) / total, 4),
                "far": round(state["far_hits"] / state["far_total"], 4) if state["far_total"] else 0.0,
                "mean_score": round(float(np.mean(state["scores"])), 4) if state["scores"] else None,
            }

        report[kind] = {mode: _pack(state) for mode, state in states.items()}
    return report


def load_directory(path):
    root = Path(path)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")
    people = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        items = []
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            image = cv2.imread(str(image_path))
            if image is not None:
                items.append(Sample(person_dir.name, image_path.name, image))
        if items:
            people[person_dir.name] = items
    if len(people) < 2:
        raise SystemExit("至少需要两个身份目录（每个目录一个同名人的图片）")
    return people


def load_lfw(people_limit, max_per_person, seed):
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        raise SystemExit("未安装 scikit-learn，无法使用 --lfw；请改用 --data-dir")
    try:
        data = fetch_lfw_people(min_faces_per_person=10, color=True, resize=1.0)
    except Exception as exc:  # 下载失败通常是没有网络
        raise SystemExit(f"LFW 下载/读取失败: {exc}\n可改用 --data-dir 指定本地人脸目录")

    buckets = defaultdict(list)
    for image, label in zip(data.images, data.target):
        buckets[data.target_names[label]].append(image)

    rng = np.random.default_rng(seed)
    eligible = [name for name, images in buckets.items() if len(images) >= 5]
    rng.shuffle(eligible)
    selected = sorted(eligible[:people_limit])

    people = {}
    for name in selected:
        images = buckets[name]
        order = rng.permutation(len(images))[:max_per_person]
        people[name] = [
            #: sklearn 返回 0~1 的 float32，必须先还原到 0~255 再转 BGR，
            #: 否则直接 astype(uint8) 会把整张图压成 0/1 的黑色
            Sample(name, f"lfw-{int(i)}",
                   cv2.cvtColor(np.clip(images[i] * 255.0, 0, 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR))
            for i in order
        ]
    return people


def extract_all(people):
    from mods.face_detector import extract_face_sample

    total = sum(len(items) for items in people.values())
    done = 0
    for rows in people.values():
        for row in rows:
            result = extract_face_sample(row.frame, quality_gate=False)
            if result is not None:
                feature, quality = result
                row.feature = unit(feature) if feature is not None else None
                row.passed = quality.passed
                row.issues = quality.issues
            done += 1
            if done % 20 == 0 or done == total:
                print(f"\r提取特征 {done}/{total}", end="", flush=True)
    print()
    return people


def print_table(title, rows):
    print(f"\n{title}")
    header = f"{'配置':<26}{'身份':>6}{'EER':>9}{'AUC':>9}{'阈值':>8}{'拒识率':>9}{'端到端FRR':>11}"
    print(header)
    print("-" * len(header))
    for label, item in rows:
        print(
            f"{label:<26}{item['identities']:>6}"
            f"{item['eer']:>9.4f}{item['auc']:>9.4f}{item['eer_threshold']:>8.2f}"
            f"{item['reject_rate']:>9.2%}{item['e2e_frr_at_eer']:>11.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description="人脸验证指标评估与消融")
    parser.add_argument("--data-dir", help="人脸目录：<root>/<person>/*.jpg")
    parser.add_argument("--lfw", action="store_true", help="使用 LFW（首次运行会下载）")
    parser.add_argument("--people", type=int, default=40, help="LFW 选取身份数，默认 40")
    parser.add_argument("--max-per-person", type=int, default=10, help="每人最多图片数")
    parser.add_argument("--enroll", type=int, default=4, help="每人用于注册的图片数")
    parser.add_argument("--threshold", type=float, default=0.6, help="线上固定阈值，默认 0.6")
    parser.add_argument("--degrade-samples", type=int, default=80, help="退化实验探针数上限")
    parser.add_argument("--no-degrade", action="store_true", help="跳过模糊/暗光/旋转实验")
    parser.add_argument("--protect", action="store_true",
                        help="额外评估可撤销模板：精度代价、吊销有效性、跨系统不可关联")
    parser.add_argument("--template-bits", type=int, default=512,
                        help="受保护模板的二进制位数，默认 512")
    parser.add_argument("--template-threshold", type=float, default=0.6,
                        help="受保护模板的一致率阈值，默认 0.6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="允许 CUDA；默认强制 CPU，因为 LFW 小图上逐张 GPU 同步比 CPU 慢一个数量级",
    )
    parser.add_argument("--json", default="reports/face_eval.json", help="指标输出路径")
    parser.add_argument("--no-write", action="store_true", help="不写 JSON")
    args = parser.parse_args()

    if not args.data_dir and not args.lfw:
        parser.error("必须指定 --data-dir 或 --lfw")

    if not args.use_gpu:
        #: face_detector 在 import 时读这个变量，必须在导入模型前设置
        os.environ.setdefault("SECURITY_FORCE_CPU", "1")

    people = load_lfw(args.people, args.max_per_person, args.seed) if args.lfw else load_directory(args.data_dir)
    print(f"数据集：{len(people)} 个身份，{sum(len(v) for v in people.values())} 张图片")

    people = extract_all(people)
    no_face = sum(1 for rows in people.values() for row in rows if row.feature is None)
    quality_rejected = sum(1 for rows in people.values() for row in rows if row.feature is not None and not row.passed)
    print(f"无效帧：未检测到人脸 {no_face}，质量门控不通过 {quality_rejected}")

    splits = collect_probe_paths(people, args.enroll, args.seed)
    if len(splits) < 2:
        raise SystemExit("可用身份不足 2 人，请增加数据或降低 --enroll")

    configs = {
        "单帧模板 + 无质量门控": (False, False, args.threshold),
        "融合模板 + 无质量门控": (False, True, args.threshold),
        "融合模板 + 质量门控": (True, True, args.threshold),
    }
    results = {}
    for label, (gate, fused, threshold) in configs.items():
        results[label] = evaluate_config(splits, gate=gate, fused=fused, threshold=threshold)

    print_table("验证指标（探针 vs 模板，阈值扫描）", results.items())

    protected = None
    if args.protect:
        protected = evaluate_protected(
            splits, gate=True, fused=True, bits=args.template_bits,
            threshold=args.template_threshold, seed=args.seed,
        )
        baseline = results["融合模板 + 质量门控"]
        protected["precision_cost"] = {
            "eer_delta": round(protected["eer"] - baseline["eer"], 4),
            "auc_delta": round(protected["auc"] - baseline["auc"], 4),
        }
        print(f"\n可撤销模板（{args.template_bits} bit，固定阈值 {args.template_threshold:.2f}）")
        print(f"  EER {protected['eer']:.4f}  AUC {protected['auc']:.4f}  "
              f"EER 阈值 {protected['eer_threshold']:.2f}")
        fixed = protected["at_threshold"]
        print(f"  固定阈值 {fixed['threshold']:.2f}：FAR {fixed['far']:.2%}  "
              f"FRR {fixed['frr']:.2%}  拒识率 {protected['reject_rate']:.2%}  "
              f"端到端 FRR {protected['e2e_frr_at_eer']:.2%}")
        print(f"  相对融合+门控的精度代价：EER {protected['precision_cost']['eer_delta']:+.4f}，"
              f"AUC {protected['precision_cost']['auc_delta']:+.4f}")
        rev = protected["revocation"]
        print(f"  吊销：{rev['templates']} 个模板换种子后平均一致率 "
              f"{rev['mean_similarity']:.4f}（最高 {rev['max_similarity']:.4f}，"
              f"仍被放行 {rev['accepted_at_threshold']} 个）")
        cross = protected["cross_system"]
        print(f"  跨系统不可关联：{cross['pairs']} 对模板平均一致率 "
              f"{cross['mean_similarity']:.4f}（最高 {cross['max_similarity']:.4f}）")

    report = {
        "dataset": {
            "identities": len(people),
            "images": sum(len(v) for v in people.values()),
            "no_face": no_face,
            "quality_rejected": quality_rejected,
            "enroll_images": args.enroll,
            "seed": args.seed,
            "source": "lfw" if args.lfw else str(Path(args.data_dir).resolve()),
        },
        "verification": results,
    }
    if protected is not None:
        report["protected"] = protected

    if not args.no_degrade:
        degrade_threshold = results["融合模板 + 质量门控"]["eer_threshold"]
        print(f"\n退化探针实验（固定阈值 {degrade_threshold:.2f}，最多 {args.degrade_samples} 张探针）")
        degradation = evaluate_degradation(
            splits, degrade_threshold, max(1, args.degrade_samples), args.seed
        )
        header = (f"{'退化':<8}{'门控':<8}{'探针':>6}{'拒识率':>9}"
                  f"{'本人失败率':>12}{'冒名放行率':>12}{'平均相似度':>12}")
        print(header)
        print("-" * len(header))
        for kind, pair in degradation.items():
            for mode, label in (("gated", "开启"), ("ungated", "关闭")):
                item = pair[mode]
                mean = "—" if item["mean_score"] is None else f"{item['mean_score']:.4f}"
                print(
                    f"{kind:<8}{label:<8}{item['probes']:>6}"
                    f"{item['reject_rate']:>9.2%}{item['frr_total']:>12.2%}"
                    f"{item['far']:>12.2%}{mean:>12}"
                )
        report["degradation"] = degradation

    if not args.no_write:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n指标已写入 {out}")


if __name__ == "__main__":
    main()
