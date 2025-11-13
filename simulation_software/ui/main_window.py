"""
主窗口类 - 包含三个功能模块的主界面
"""
from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, 
                            QWidget, QMenuBar, QAction, QToolBar, 
                            QStatusBar, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence

from .parameter_input_widget import ParameterInputWidget
from .image_recognition_widget import ImageRecognitionWidget  # 后续实现
from .parameter_optimization_widget import ParameterOptimizationWidget  # 后续实现

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("仿真软件管理系统 v1.0")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化UI组件
        self.init_ui()
        self.init_menu()
        self.init_toolbar()
        self.init_statusbar()
        
    def init_ui(self):
        """初始化主界面"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡控件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 添加三个功能模块的选项卡
        self.init_tabs()
        
        
        
    def init_tabs(self):
        """初始化选项卡"""
        # 第一个选项卡：参数输入与仿真调用
        self.parameter_widget = ParameterInputWidget()
        self.tab_widget.addTab(self.parameter_widget, "参数输入与仿真")
        
        # 第二个选项卡：图像识别（暂时用占位符）
        placeholder_widget = QWidget()
        self.tab_widget.addTab(placeholder_widget, "图像识别")
        
        # 第三个选项卡：参数寻优（暂时用占位符）
        placeholder_widget2 = QWidget()
        self.tab_widget.addTab(placeholder_widget2, "参数寻优")
        
        # 连接选项卡切换信号
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
    def init_menu(self):
        """初始化菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        # 新建配置
        new_action = QAction('新建配置(&N)', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self.new_config)
        file_menu.addAction(new_action)
        
        # 打开配置
        open_action = QAction('打开配置(&O)', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_config)
        file_menu.addAction(open_action)
        
        # 保存配置
        save_action = QAction('保存配置(&S)', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction('退出(&X)', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具(&T)')
        
        # 数据库连接测试
        db_test_action = QAction('数据库连接测试(&D)', self)
        db_test_action.triggered.connect(self.test_database_connection)
        tools_menu.addAction(db_test_action)
        
        # 清理临时文件
        clean_action = QAction('清理临时文件(&C)', self)
        clean_action.triggered.connect(self.clean_temp_files)
        tools_menu.addAction(clean_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 关于
        about_action = QAction('关于(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def init_toolbar(self):
        """初始化工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 添加常用操作按钮
        new_action = QAction('新建', self)
        new_action.triggered.connect(self.new_config)
        toolbar.addAction(new_action)
        
        open_action = QAction('打开', self)
        open_action.triggered.connect(self.open_config)
        toolbar.addAction(open_action)
        
        save_action = QAction('保存', self)
        save_action.triggered.connect(self.save_config)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 数据库连接测试
        db_action = QAction('数据库测试', self)
        db_action.triggered.connect(self.test_database_connection)
        toolbar.addAction(db_action)
        
    def init_statusbar(self):
        """初始化状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪", 2000)
        
    def on_tab_changed(self, index):
        """选项卡切换事件"""
        tab_names = ["参数输入与仿真", "图像识别", "参数寻优"]
        if index < len(tab_names):
            self.status_bar.showMessage(f"当前模块: {tab_names[index]}")
            
    def new_config(self):
        """新建配置"""
        self.status_bar.showMessage("新建配置", 2000)
        # 重置当前选项卡的参数
        current_widget = self.tab_widget.currentWidget()
        if hasattr(current_widget, 'reset_parameters'):
            current_widget.reset_parameters()
            
    def open_config(self):
        """打开配置"""
        self.status_bar.showMessage("打开配置", 2000)
        # TODO: 实现配置文件加载
        QMessageBox.information(self, "提示", "配置加载功能将在后续版本实现")
        
    def save_config(self):
        """保存配置"""
        self.status_bar.showMessage("保存配置", 2000)
        # TODO: 实现配置文件保存
        QMessageBox.information(self, "提示", "配置保存功能将在后续版本实现")
        
    def test_database_connection(self):
        """测试数据库连接"""
        self.status_bar.showMessage("测试数据库连接中...", 0)
        
        try:
            from ..utils.database_manager import DatabaseManager
            db_manager = DatabaseManager()
            if db_manager.test_connection():
                QMessageBox.information(self, "数据库连接", "数据库连接成功！")
                self.status_bar.showMessage("数据库连接成功", 3000)
            else:
                QMessageBox.warning(self, "数据库连接", "数据库连接失败！请检查配置。")
                self.status_bar.showMessage("数据库连接失败", 3000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据库连接测试出错：{str(e)}")
            self.status_bar.showMessage("数据库连接测试出错", 3000)
            
    def clean_temp_files(self):
        """清理临时文件"""
        self.status_bar.showMessage("清理临时文件", 2000)
        # TODO: 实现临时文件清理
        QMessageBox.information(self, "提示", "临时文件清理功能将在后续版本实现")
        
    def show_about(self):
        """显示关于信息"""
        about_text = """
        <h3>仿真软件管理系统 v1.0</h3>
        <p>一个集成参数输入、图像识别和参数寻优的仿真管理平台</p>
        <p><b>开发者：</b> BUPT Simulation Lab</p>
        <p><b>技术栈：</b> Python, PyQt5, MySQL, OpenCV</p>
        <p><b>功能模块：</b></p>
        <ul>
        <li>参数输入与仿真调用</li>
        <li>图像识别与指标计算</li>
        <li>参数寻优</li>
        </ul>
        """
        QMessageBox.about(self, "关于", about_text)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(self, '确认退出', 
                                   '确定要退出仿真软件管理系统吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()