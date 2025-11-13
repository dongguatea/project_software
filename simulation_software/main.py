#!/usr/bin/env python3
"""
仿真软件管理系统 - 主程序
包含三个功能模块：参数输入与仿真调用、图像识别、参数寻优
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow

def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setApplicationName("仿真软件管理系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("BUPT Simulation Lab")
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon("resources/icon.png"))
    
    # 创建主窗口
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()