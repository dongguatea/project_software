# EntityRow.py
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QDoubleSpinBox, QFileDialog
)
from PyQt5.QtCore import Qt

class EntityRow(QWidget):
    """统一的实体行：name, category, x/y/z, heading/pitch/roll, flt, ms, 删除"""
    def __init__(self, name: str = "", parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        r = 0

        # 名称
        lay.addWidget(QLabel("名称:"), r, 0)
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("例如：water / manbo / entity_01")
        lay.addWidget(self.name_edit, r, 1, 1, 3)

        # —— 新增：category —— #
        r += 1
        lay.addWidget(QLabel("category:"), r, 0)
        self.category_edit = QLineEdit("defaultCategory")
        self.category_edit.setPlaceholderText("例如：main / interference / defaultCategory")
        lay.addWidget(self.category_edit, r, 1, 1, 3)

        # 位置
        r += 1
        lay.addWidget(QLabel("位置(x,y,z):"), r, 0)
        self.x = QDoubleSpinBox(); self.x.setRange(-1e9, 1e9); self.x.setDecimals(3)
        self.y = QDoubleSpinBox(); self.y.setRange(-1e9, 1e9); self.y.setDecimals(3)
        self.z = QDoubleSpinBox(); self.z.setRange(-1e9, 1e9); self.z.setDecimals(3)
        pos_box = QHBoxLayout(); pos_box.addWidget(self.x); pos_box.addWidget(self.y); pos_box.addWidget(self.z)
        pos_wrap = QWidget(); pos_wrap.setLayout(pos_box)
        lay.addWidget(pos_wrap, r, 1, 1, 3)

        # 姿态
        r += 1
        lay.addWidget(QLabel("姿态(H/P/R):"), r, 0)
        self.h = QDoubleSpinBox(); self.h.setRange(-360, 360); self.h.setDecimals(3)
        self.p = QDoubleSpinBox(); self.p.setRange(-360, 360); self.p.setDecimals(3)
        self.roll = QDoubleSpinBox(); self.roll.setRange(-360, 360); self.roll.setDecimals(3)
        att_box = QHBoxLayout(); att_box.addWidget(self.h); att_box.addWidget(self.p); att_box.addWidget(self.roll)
        att_wrap = QWidget(); att_wrap.setLayout(att_box)
        lay.addWidget(att_wrap, r, 1, 1, 3)

        # flt 路径
        r += 1
        lay.addWidget(QLabel("FLT路径:"), r, 0)
        self.flt_path = QLineEdit(); self.flt_path.setPlaceholderText("选择 *.flt 或直接粘贴路径")
        btn_flt = QPushButton("浏览"); btn_flt.clicked.connect(self._browse_flt)
        flt_box = QHBoxLayout(); flt_box.addWidget(self.flt_path); flt_box.addWidget(btn_flt)
        flt_wrap = QWidget(); flt_wrap.setLayout(flt_box)
        lay.addWidget(flt_wrap, r, 1, 1, 3)

        # ms 路径
        r += 1
        lay.addWidget(QLabel("MS路径:"), r, 0)
        self.ms_path = QLineEdit(); self.ms_path.setPlaceholderText("选择 *.ms 或直接粘贴路径")
        btn_ms = QPushButton("浏览"); btn_ms.clicked.connect(self._browse_ms)
        ms_box = QHBoxLayout(); ms_box.addWidget(self.ms_path); ms_box.addWidget(btn_ms)
        ms_wrap = QWidget(); ms_wrap.setLayout(ms_box)
        lay.addWidget(ms_wrap, r, 1, 1, 3)

        # 删除按钮
        r += 1
        self.btn_remove = QPushButton("删除该实体")
        self.btn_remove.setProperty("is_remove", True)
        lay.addWidget(self.btn_remove, r, 1, 1, 1, alignment=Qt.AlignLeft)

    def _browse_flt(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 FLT 文件", "", "FLT 文件 (*.flt);;所有文件 (*)")
        if path:
            self.flt_path.setText(path)

    def _browse_ms(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 MS 文件", "", "MS 文件 (*.ms);;所有文件 (*)")
        if path:
            self.ms_path.setText(path)

    def get_data(self):
        return {
            "name": self.name_edit.text().strip(),
            "category": self.category_edit.text().strip(),
            "x": self.x.value(), "y": self.y.value(), "z": self.z.value(),
            "heading": self.h.value(), "pitch": self.p.value(), "roll": self.roll.value(),
            "flt": self.flt_path.text().strip(),
            "ms": self.ms_path.text().strip(),
        }
