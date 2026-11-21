from PyQt5.QtWidgets import QApplication
import sys
from mods import auto_ui
from mods import hand_ui
class hand():
    def __init__(self):
        self.window=QApplication(sys.argv)
        self.app_window=hand_ui.loginwindow()
        self.app_window.show()
        sys.exit(self.window.exec_())
class auto():
    def __init__(self):
       self.window=QApplication(sys.argv)
       self.lockscreen=auto_ui.Lockscreen()
       self.lockscreen.show()
       sys.exit(self.lockscreen.exec_())
class controller():
    def __init__(self):
        self.is_auto='--auto' in sys.argv

    def use_auto_or_hand(self):
        if self.is_auto:
            auto()
        else:
            hand()
if __name__=='__main__':
    start=controller()
    start.use_auto_or_hand()