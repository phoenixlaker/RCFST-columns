# main.py
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QGridLayout, QMessageBox)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from cal_test import final_out

class CalWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ultimate Strength Calculator')
        self.setFixedSize(600, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f2f5;
            }
            QLabel {
                font-family: 'Microsoft YaHei';
                font-size: 14px;
            }
            QLineEdit {
                font-family: 'Microsoft YaHei';
                font-size: 14px;
                padding: 6px;
                border: 1px solid #ccc;
                border-radius: 6px;
                background: white;
            }
            QPushButton {
                font-family: 'Microsoft YaHei';
                font-size: 15px;
                background-color: #4caf50;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)

        self.labels = [
            'Sectional Height (H) (mm)',
            'Sectional Width (B) (mm)',
            'Thickness of steel tube (t) (mm)',
            'Column length (L) (mm)',
            'Cylinder compressive strength of concrete (fc\') (MPa)',
            'Standard axial compressive strength of concrete (fck) (MPa)',
            'Yield strength of steel (fy) (MPa)',
            'eccentricity (e) (mm)',
            "Young's modulus of concrete (Ec) (MPa)",
            "Young's modulus of steel (Es) (MPa)"
        ]

        # 提供一组默认值
        self.default_values = [
            300,   # H
            300,   # B
            10,    # t
            3000,  # L
            40,    # fc'
            30,    # fck
            345,   # fy
            20,    # e
            30000, # Ec
            200000 # Es
        ]

        self.inputs = []
        grid = QGridLayout()
        grid.setSpacing(12)

        # 加标题
        self.title_label = QLabel('Ultimate Strength Prediction System')
        self.title_label.setFont(QFont('Microsoft YaHei', 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        grid.addWidget(self.title_label, 0, 0, 1, 2)

        # 输入框从第1行开始
        for i, label_text in enumerate(self.labels):
            label = QLabel(label_text)
            input_field = QLineEdit()
            input_field.setText(str(self.default_values[i]))
            grid.addWidget(label, i + 1, 0)
            grid.addWidget(input_field, i + 1, 1)
            self.inputs.append(input_field)

        # 计算按钮和结果输出再往后排
        self.calc_button = QPushButton('Calculate Ultimate Strength')
        self.result_label = QLabel('Ultimate Strength (Pu): ')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont('Microsoft YaHei', 16))

        grid.addWidget(self.calc_button, len(self.labels) + 1, 0, 1, 2)
        grid.addWidget(self.result_label, len(self.labels) + 2, 0, 1, 2)

        self.setLayout(grid)

class CalController:
    def __init__(self, window):
        self.window = window
        self.bindEvents()

    def bindEvents(self):
        self.window.calc_button.clicked.connect(self.calculate)

    def calculate(self):
        try:
            params = [float(input_field.text()) for input_field in self.window.inputs]
            if len(params) != 10:
                raise ValueError('Please fill all fields.')

            H, B, t, L, fc, fck, fy, e, Ec, Es = params
            Pu = final_out(H, B, t, e, L, fy, fc, fck, Es, Ec)
            Pu_value = Pu[0]  # 取第一个结果

            self.window.result_label.setText(f'Ultimate Strength (Pu): {Pu_value:.2f} kN')

        except Exception as ex:
            QMessageBox.warning(self.window, 'Error', f'Calculation failed: {ex}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = CalWindow()
    controller = CalController(win)
    win.show()
    sys.exit(app.exec_())
