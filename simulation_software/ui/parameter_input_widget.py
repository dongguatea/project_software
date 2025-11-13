"""
参数输入界面 - 第一个功能模块
用于输入仿真参数，生成配置文件，并调用仿真软件
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QGroupBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
                            QComboBox, QCheckBox, QPushButton, QTextEdit, QTabWidget,
                            QProgressBar, QMessageBox, QFileDialog, QTableWidget,
                            QTableWidgetItem, QHeaderView, QScrollArea,QTimeEdit,QTime,QSlider,QDateEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot,QDate
from PyQt5.QtGui import QFont
from .EntityRow import EntityRow
import os
import sys

class ParameterInputWidget(QWidget):
    """参数输入界面主控件"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_default_values()
        
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout(self)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 创建参数输入区域
        self.create_sensor_group(scroll_layout)
        self.create_entity_group(scroll_layout)
        self.create_environment_group(scroll_layout)
        self.create_simulation_group(scroll_layout)
        
        # 创建控制按钮区域
        self.create_control_buttons(scroll_layout)
        
        # 创建进度和日志区域
        self.create_progress_log_area(scroll_layout)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
    def create_sensor_group(self, parent_layout):
        """创建传感器参数组"""
        group = QGroupBox("传感器参数")
        layout = QGridLayout(group)
        
        # 传感器名称
        layout.addWidget(QLabel("传感器类型:"), 0, 0)
        self.sensor_name_combo = QComboBox()
        self.sensor_name_combo.addItems(['Mid-wave Infrared - MWIR', 'Long Wave Infrared - LWIR'])
        layout.addWidget(self.sensor_name_combo, 0, 1)
        
        # 温度范围
        layout.addWidget(QLabel("最低温度 (°C):"), 1, 0)
        self.min_temp_spin = QSpinBox()
        self.min_temp_spin.setRange(-50, 100)
        self.min_temp_spin.setValue(0)
        layout.addWidget(self.min_temp_spin, 1, 1)
        
        layout.addWidget(QLabel("最高温度 (°C):"), 1, 2)
        self.max_temp_spin = QSpinBox()
        self.max_temp_spin.setRange(-50, 150)
        self.max_temp_spin.setValue(50)
        layout.addWidget(self.max_temp_spin, 1, 3)
        
        # FOV像素
        layout.addWidget(QLabel("水平FOV像素:"), 2, 0)
        self.h_fov_pixels_combo = QComboBox()
        self.h_fov_pixels_combo.addItems(['320', '640', '1024'])
        layout.addWidget(self.h_fov_pixels_combo, 2, 1)
        
        layout.addWidget(QLabel("垂直FOV像素:"), 2, 2)
        self.v_fov_pixels_combo = QComboBox()
        self.v_fov_pixels_combo.addItems(['256', '512', '1024'])
        layout.addWidget(self.v_fov_pixels_combo, 2, 3)
        
        # 传感器仿真参数
        layout.addWidget(QLabel("噪声百分比 (0-1):"), 3, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 1.0)
        self.noise_spin.setSingleStep(0.1)
        self.noise_spin.setDecimals(2)
        self.noise_spin.setValue(0.0)
        layout.addWidget(self.noise_spin, 3, 1)
        
        layout.addWidget(QLabel("模糊百分比 (0-1):"), 3, 2)
        self.blur_spin = QDoubleSpinBox()
        self.blur_spin.setRange(0.0, 1.0)
        self.blur_spin.setSingleStep(0.1)
        self.blur_spin.setDecimals(2)
        self.blur_spin.setValue(0.0)
        layout.addWidget(self.blur_spin, 3, 3)
        
        # 轨迹文件路径
        layout.addWidget(QLabel("轨迹文件路径:"), 4, 0)
        self.track_path_edit = QLineEdit()
        self.track_path_edit.setText(r'E:\track.trk')
        layout.addWidget(self.track_path_edit, 4, 1, 1, 2)
        
        track_browse_btn = QPushButton("浏览...")
        track_browse_btn.clicked.connect(self.browse_track_file)
        layout.addWidget(track_browse_btn, 4, 3)
        
        parent_layout.addWidget(group)

#待修改，预期实现效果：主要实体后可以添加按钮，用来增加实体参数，并且设置实体的x，y,z，phi,theta,rcoll以及flt、ms文件路径
# 干扰同理，也可以不添加        
    def create_entity_group(self, parent_layout):
      group = QGroupBox("实体参数")
      layout = QGridLayout(group)

      # 主要实体（名字列表，用逗号分隔）
      layout.addWidget(QLabel("主要实体:"), 0, 0)
      self.entity_list_edit = QLineEdit()
      self.entity_list_edit.setText("water,manbo")
      layout.addWidget(self.entity_list_edit, 0, 1, 1, 2)

      # —— 删除：干扰实体数量（整块移除）——
      # （无）

      # —— 你原来的“主要实体坐标 / 干扰实体坐标”如果不打算继续使用，可移除；
      # 若保留也行，但收集时我们不再读这两个字段 —— 改为从 EntityRow 收集精确坐标 —— #

      # 添加按钮
      from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget
      self.btn_add_from_list = QPushButton("从主要实体列表添加")
      self.btn_add_one = QPushButton("＋ 添加实体")
      layout.addWidget(self.btn_add_from_list, 1, 1)
      layout.addWidget(self.btn_add_one, 1, 2)

      # 行容器：放多个 EntityRow
      self.entity_rows_wrap = QWidget()
      self.entity_rows_layout = QVBoxLayout(self.entity_rows_wrap)
      self.entity_rows_layout.setContentsMargins(0, 0, 0, 0)
      self.entity_rows_layout.setSpacing(8)
      layout.addWidget(self.entity_rows_wrap, 2, 0, 1, 3)

      # 连接
      self.btn_add_one.clicked.connect(lambda: self.add_entity_row())
      self.btn_add_from_list.clicked.connect(self.add_entities_from_text)

      # 预置：根据主要实体文本生成若干行（可选）
      self.add_entities_from_text()

      parent_layout.addWidget(group)
        
    def create_environment_group(self, parent_layout):
        """创建环境参数组"""
        group = QGroupBox("环境参数")
        layout = QGridLayout(group)
        
        row = QHBoxLayout()
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm")
        self.time_edit.setTimeRange(QTime(0,0), QTime(23,59))
        self.time_edit.setTime(QTime(9,30))  # 初始值
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 24*60 - 1)  # 0..1439
        self.time_slider.setSingleStep(1)
        self.time_slider.setPageStep(5)
        row.addWidget(QLabel("仿真时间:"))
        row.addWidget(self.time_edit)
        row.addWidget(self.time_slider)

        self.time_slider.valueChanged.connect(self.on_slider_changed)
        self.time_edit.timeChanged.connect(self.on_time_changed)

        layout.addLayout(row,0,0,1,4)

        # # 时间设置
        # layout.addWidget(QLabel("仿真时间:"), 0, 0)
        # self.time_line = QLineEdit()
        # self.time_line.setText('00:00-23:59')
        # layout.addWidget(self.time_line, 0, 1)
        
        # 日期设置
        layout.addWidget(QLabel("仿真日期(月/日/年):"), 1, 0)
        self.date_edit = QDateEdit()
        self.date_edit.setDisplayFormat("MM/dd/yyyy")
        self.date_edit.setDate(QDate(2024, 7, 24))
        self.date_edit.setCalendarPopup(True)
        layout.addWidget(self.date_edit, 1, 1)
        
        # 气溶胶模型
        layout.addWidget(QLabel("气溶胶模型:"), 2, 0)
        self.haze_model_combo = QComboBox()
        self.haze_model_combo.addItems([str(i) for i in range(0,11)])
        layout.addWidget(self.haze_model_combo, 2, 1)
        
        # 降雨率
        layout.addWidget(QLabel("降雨率:"), 3, 0)
        self.rain_rate_spin = QSpinBox()
        self.rain_rate_spin.setRange(0,100)
        layout.addWidget(self.rain_rate_spin, 3, 1)
        
        # 能见度
        layout.addWidget(QLabel("能见度设置(如果气溶胶模型非0时该项为0):"), 3, 2)
        # self.visibility_line = QLineEdit()
        # self.visibility_line.setText('0')
        self.visibility_spin = QSpinBox()
        self.visibility_spin.setRange(0,10000)
        # layout.addWidget(self.visibility_line, 2, 1)
        layout.addWidget(self.visibility_spin,3,3)
        parent_layout.addWidget(group)
        
    def create_simulation_group(self, parent_layout):
        """创建仿真设置组"""
        group = QGroupBox("仿真设置")
        layout = QGridLayout(group)
        
        # 数据库连接设置
        layout.addWidget(QLabel("数据库主机:"), 0, 0)
        self.db_host_edit = QLineEdit()
        self.db_host_edit.setText("localhost")
        layout.addWidget(self.db_host_edit, 0, 1)
        
        layout.addWidget(QLabel("数据库端口:"), 0, 2)
        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(0,65536)
        self.db_port_spin.setValue(3306)
        layout.addWidget(self.db_port_spin, 0, 3)
        
        layout.addWidget(QLabel("用户名:"), 1, 0)
        self.db_user_edit = QLineEdit()
        self.db_user_edit.setText("root")
        layout.addWidget(self.db_user_edit, 1, 1)
        
        layout.addWidget(QLabel("密码:"), 1, 2)
        self.db_password_edit = QLineEdit()
        self.db_password_edit.setEchoMode(QLineEdit.Password)
        self.db_password_edit.setText("asd515359")
        layout.addWidget(self.db_password_edit, 1, 3)
        
        layout.addWidget(QLabel("数据库名:"), 2, 0)
        self.db_name_edit = QLineEdit()
        self.db_name_edit.setText("config_db")
        layout.addWidget(self.db_name_edit, 2, 1)
        
        # 输出目录设置
        layout.addWidget(QLabel("输出目录:"), 3, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "xmlcreate", "generate"))
        layout.addWidget(self.output_dir_edit, 3, 1, 1, 2)
        
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        layout.addWidget(output_browse_btn, 3, 3)
        
        parent_layout.addWidget(group)
        
    def create_control_buttons(self, parent_layout):
        """创建控制按钮"""
        button_layout = QHBoxLayout()
        
        # 参数预览按钮
        self.preview_btn = QPushButton("参数预览")
        self.preview_btn.clicked.connect(self.preview_parameters)
        button_layout.addWidget(self.preview_btn)
        
        # 生成配置文件按钮
        self.generate_btn = QPushButton("生成配置文件")
        self.generate_btn.clicked.connect(self.generate_config_files)
        button_layout.addWidget(self.generate_btn)
        
        # 导入数据库按钮
        self.import_db_btn = QPushButton("导入数据库")
        self.import_db_btn.clicked.connect(self.import_to_database)
        button_layout.addWidget(self.import_db_btn)
        
        # 生成XML按钮
        self.generate_xml_btn = QPushButton("生成XML文件")
        self.generate_xml_btn.clicked.connect(self.generate_xml_files)
        button_layout.addWidget(self.generate_xml_btn)
        
        # 启动仿真按钮
        self.start_simulation_btn = QPushButton("启动仿真")
        self.start_simulation_btn.clicked.connect(self.start_simulation)
        self.start_simulation_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.start_simulation_btn)
        
        # 一键执行按钮
        self.one_click_btn = QPushButton("一键执行全流程")
        self.one_click_btn.clicked.connect(self.execute_full_pipeline)
        self.one_click_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        button_layout.addWidget(self.one_click_btn)
        
        button_layout.addStretch()
        
        # 重置按钮
        self.reset_btn = QPushButton("重置参数")
        self.reset_btn.clicked.connect(self.reset_parameters)
        button_layout.addWidget(self.reset_btn)
        
        parent_layout.addLayout(button_layout)
        
    def create_progress_log_area(self, parent_layout):
        """创建进度条和日志区域"""
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        parent_layout.addWidget(self.progress_bar)
        
        # 日志区域
        log_group = QGroupBox("执行日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        # 日志控制按钮
        log_button_layout = QHBoxLayout()
        clear_log_btn = QPushButton("清空日志")
        clear_log_btn.clicked.connect(self.clear_log)
        log_button_layout.addWidget(clear_log_btn)
        
        save_log_btn = QPushButton("保存日志")
        save_log_btn.clicked.connect(self.save_log)
        log_button_layout.addWidget(save_log_btn)
        
        log_button_layout.addStretch()
        log_layout.addLayout(log_button_layout)
        
        parent_layout.addWidget(log_group)
    

    def add_entity_row(self, name: str = ""):
      row = EntityRow(name=name, parent=self)
      if hasattr(row, "btn_remove"):
          row.btn_remove.clicked.connect(lambda: self._remove_entity_row(row))
      self.entity_rows_layout.addWidget(row)
      return row

    def _remove_entity_row(self, row):
      row.setParent(None)
      row.deleteLater()

    def add_entities_from_text(self):
      text = self.entity_list_edit.text().strip()
      if not text:
          return
      names = [s.strip() for s in text.split(",") if s.strip()]
      for nm in names:
          self.add_entity_row(name=nm)
    def on_slider_changed(self,v):
      t = self.minutes_to_qtime(v)
      if t != self.time_edit.time():
        self.time_edit.blockSignals(True)
        self.time_edit.setTime(t)
        self.time_edit.blockSignals(False)
      

    def on_time_changed(self,t: QTime):
      v = self.qtime_to_minutes(t)
      if v != self.time_slider.value():
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(v)
        self.time_slider.blockSignals(False)
      
    
    def minutes_to_qtime(self,minutes):
      hours = minutes // 60
      mins = minutes % 60
      return QTime(hours, mins)
    def qtime_to_minutes(self,t: QTime):
      return t.hour() * 60 + t.minute()
    
    def load_default_values(self):
        """加载默认值"""
        self.log("界面初始化完成，已加载默认参数")
        
    def browse_track_file(self):
        """浏览轨迹文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择轨迹文件", "", "Track Files (*.trk);;All Files (*)")
        if file_path:
            self.track_path_edit.setText(file_path)
            
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            
    def preview_parameters(self):
        """预览参数"""
        params = self.collect_parameters()
        preview_text = self.format_parameters_for_preview(params)
        
        # 创建参数预览对话框
        from PyQt5.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton
        dialog = QDialog(self)
        dialog.setWindowTitle("参数预览")
        dialog.setGeometry(200, 200, 600, 400)
        
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setPlainText(preview_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()
        
    def collect_parameters(self):
        """收集界面上的所有参数"""
        params={
            'sensor': {
                'name': self.sensor_name_combo.currentText(),
                'min_temperature': self.min_temp_spin.value(),
                'max_temperature': self.max_temp_spin.value(),
                'h_fov_pixels': int(self.h_fov_pixels_combo.currentText()),
                'v_fov_pixels': int(self.v_fov_pixels_combo.currentText()),
                'noise': self.noise_spin.value(),
                'blur': self.blur_spin.value(),
                'track_path': self.track_path_edit.text()
            },
            'environment': {
                'time': self.time_edit.time().toString("HH:mm"),
                'date': self.date_edit.date().toString("MM/dd/yyyy"),
                'haze_model': int(self.haze_model_combo.currentText()),
                'rain_rate': float(self.rain_rate_spin.value()),
                'visibility': int(self.visibility_spin.value())
            },
            'database': {
                'host': self.db_host_edit.text(),
                'port': self.db_port_spin.value(),
                'user': self.db_user_edit.text(),
                'password': self.db_password_edit.text(),
                'database': self.db_name_edit.text()
            },
            'simulation': {
                'output_dir': self.output_dir_edit.text()
            }
        }
        # 收集实体参数
        entity_list = []
        entity_coords = []   # 每个元素是 (x, y, z, heading, pitch, roll)
        category = []

        if hasattr(self, "entity_rows_layout"):
            for i in range(self.entity_rows_layout.count()):
                w = self.entity_rows_layout.itemAt(i).widget()
                if isinstance(w, EntityRow):
                    data = w.get_data()
                    nm = data.get("name", "").strip()
                    if not nm:
                        continue
                    entity_list.append(nm)
                    entity_coords.append((
                        data.get("x", 0.0), data.get("y", 0.0), data.get("z", 0.0),
                        data.get("heading", 0.0), data.get("pitch", 0.0), data.get("roll", 0.0)
                    ))
                    category.append(data.get("category", ""))

        params['entity'] = {
            'entity_list': entity_list,
            'entity_coords': entity_coords,
            'category': category
        }
        return params

        
    def format_parameters_for_preview(self, params):
        """格式化参数用于预览"""
        text = "=== 仿真参数预览 ===\\n\\n"
        
        text += "传感器参数:\\n"
        text += f"  类型: {params['sensor']['name']}\\n"
        text += f"  温度范围: {params['sensor']['min_temperature']}°C - {params['sensor']['max_temperature']}°C\\n"
        text += f"  FOV像素: {params['sensor']['h_fov_pixels']} x {params['sensor']['v_fov_pixels']}\\n"
        text += f"  噪声: {params['sensor']['noise']}, 模糊: {params['sensor']['blur']}\\n"
        text += f"  轨迹文件: {params['sensor']['track_path']}\\n\\n"
        
        text += "实体参数:\\n"
        text += f"  主要实体: {', '.join(params['entity']['entity_list'])}\\n"
        text += f"  干扰实体数量: {params['entity']['interference_count']}\\n"
        text += f"  主要实体坐标: {params['entity']['main_coords']}\\n"
        text += f"  干扰实体坐标: {params['entity']['interference_coords']}\\n\\n"
        
        text += "环境参数:\\n"
        text += f"  时间: {params['environment']['time']}\\n"
        text += f"  日期: {params['environment']['date']}\\n"
        text += f"  气溶胶模型: {params['environment']['haze_model']}\\n"
        text += f"  降雨率: {params['environment']['rain_rate']}\\n"
        text += f"  能见度: {params['environment']['visibility']}\\n\\n"
        
        text += "数据库参数:\\n"
        text += f"  主机: {params['database']['host']}:{params['database']['port']}\\n"
        text += f"  数据库: {params['database']['database']}\\n"
        text += f"  用户: {params['database']['user']}\\n\\n"
        
        text += "输出设置:\\n"
        text += f"  输出目录: {params['simulation']['output_dir']}\\n"
        
        return text
        
    def generate_config_files(self):
        """生成配置文件"""
        self.log("开始生成配置文件...")
        self.show_progress("生成配置文件中...")
        
        try:
            # 这里调用后端的generate_fix_data逻辑
            from ..core.simulation_manager import SimulationManager
            params = self.collect_parameters()
            
            manager = SimulationManager()
            success = manager.generate_config_files(params)
            
            if success:
                self.log("配置文件生成成功")
                QMessageBox.information(self, "成功", "配置文件生成成功！")
            else:
                self.log("配置文件生成失败")
                QMessageBox.warning(self, "警告", "配置文件生成失败，请检查参数设置")
                
        except Exception as e:
            self.log(f"生成配置文件时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成配置文件时出错：{str(e)}")
        finally:
            self.hide_progress()
            
    def import_to_database(self):
        """导入数据库"""
        self.log("开始导入数据库...")
        self.show_progress("导入数据库中...")
        
        try:
            from ..core.simulation_manager import SimulationManager
            params = self.collect_parameters()
            
            manager = SimulationManager()
            success = manager.import_to_database(params)
            
            if success:
                self.log("数据库导入成功")
                QMessageBox.information(self, "成功", "数据已成功导入数据库！")
            else:
                self.log("数据库导入失败")
                QMessageBox.warning(self, "警告", "数据库导入失败，请检查数据库连接")
                
        except Exception as e:
            self.log(f"导入数据库时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"导入数据库时出错：{str(e)}")
        finally:
            self.hide_progress()
            
    def generate_xml_files(self):
        """生成XML文件"""
        self.log("开始生成XML文件...")
        self.show_progress("生成XML文件中...")
        
        try:
            from ..core.simulation_manager import SimulationManager
            params = self.collect_parameters()
            
            manager = SimulationManager()
            success = manager.generate_xml_files(params)
            
            if success:
                self.log("XML文件生成成功")
                QMessageBox.information(self, "成功", "XML文件生成成功！")
            else:
                self.log("XML文件生成失败")
                QMessageBox.warning(self, "警告", "XML文件生成失败")
                
        except Exception as e:
            self.log(f"生成XML文件时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"生成XML文件时出错：{str(e)}")
        finally:
            self.hide_progress()
            
    def start_simulation(self):
        """启动仿真"""
        self.log("开始启动仿真...")
        self.show_progress("启动仿真中...")
        
        try:
            from ..core.simulation_manager import SimulationManager
            params = self.collect_parameters()
            
            manager = SimulationManager()
            success = manager.start_simulation(params)
            
            if success:
                self.log("仿真启动成功")
                QMessageBox.information(self, "成功", "仿真已启动！")
            else:
                self.log("仿真启动失败")
                QMessageBox.warning(self, "警告", "仿真启动失败")
                
        except Exception as e:
            self.log(f"启动仿真时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"启动仿真时出错：{str(e)}")
        finally:
            self.hide_progress()
            
    def execute_full_pipeline(self):
        """执行完整流程"""
        self.log("开始执行完整流程...")
        
        # 创建并启动工作线程
        self.pipeline_thread = PipelineThread(self.collect_parameters())
        self.pipeline_thread.progress_updated.connect(self.update_progress)
        self.pipeline_thread.log_updated.connect(self.log)
        self.pipeline_thread.finished.connect(self.on_pipeline_finished)
        
        self.show_progress("执行完整流程中...")
        self.one_click_btn.setEnabled(False)
        self.pipeline_thread.start()
        
    def on_pipeline_finished(self, success, message):
        """完整流程执行完成"""
        self.hide_progress()
        self.one_click_btn.setEnabled(True)
        
        if success:
            self.log("完整流程执行成功")
            QMessageBox.information(self, "成功", "完整流程执行成功！")
        else:
            self.log(f"完整流程执行失败: {message}")
            QMessageBox.warning(self, "失败", f"完整流程执行失败：{message}")
            
    def reset_parameters(self):
        """重置参数"""
        reply = QMessageBox.question(self, '确认重置', 
                                   '确定要重置所有参数到默认值吗？',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.load_default_values()
            self.log("参数已重置到默认值")
            
    def show_progress(self, message):
        """显示进度条"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.log(message)
        
    def hide_progress(self):
        """隐藏进度条"""
        self.progress_bar.setVisible(False)
        
    def update_progress(self, value):
        """更新进度"""
        if self.progress_bar.maximum() != 0:
            self.progress_bar.setValue(value)
            
    def log(self, message):
        """添加日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def clear_log(self):
        """清空日志"""
        self.log_text.clear()
        
    def save_log(self):
        """保存日志"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存日志", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.toPlainText())
                self.log(f"日志已保存到: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存日志失败：{str(e)}")


class PipelineThread(QThread):
    """完整流程执行线程"""
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        
    def run(self):
        """执行完整流程"""
        try:
            from ..core.simulation_manager import SimulationManager
            manager = SimulationManager()
            
            # 步骤1: 生成配置文件
            self.log_updated.emit("步骤1/4: 生成配置文件...")
            self.progress_updated.emit(25)
            if not manager.generate_config_files(self.parameters):
                self.finished.emit(False, "生成配置文件失败")
                return
                
            # 步骤2: 导入数据库
            self.log_updated.emit("步骤2/4: 导入数据库...")
            self.progress_updated.emit(50)
            if not manager.import_to_database(self.parameters):
                self.finished.emit(False, "导入数据库失败")
                return
                
            # 步骤3: 生成XML文件
            self.log_updated.emit("步骤3/4: 生成XML文件...")
            self.progress_updated.emit(75)
            if not manager.generate_xml_files(self.parameters):
                self.finished.emit(False, "生成XML文件失败")
                return
                
            # 步骤4: 启动仿真
            self.log_updated.emit("步骤4/4: 启动仿真...")
            self.progress_updated.emit(100)
            if not manager.start_simulation(self.parameters):
                self.finished.emit(False, "启动仿真失败")
                return
                
            self.finished.emit(True, "完整流程执行成功")
            
        except Exception as e:
            self.finished.emit(False, str(e))