import sys
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton)
from PyQt5.QtGui import (QPainter, QPen, QColor, QFont, QBrush, 
                         QCursor, QTransform)
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect

class OpticalSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Принцип Ферма с визуализацией углов")
        self.setGeometry(100, 100, 1200, 800)
        
        # Параметры просмотра
        self.view_offset = QPoint(600, 400)
        self.scale = 80.0
        self.pan_start = QPoint()
        self.panning = False
        
        # Параметры моделирования
        self.n1 = 1.0
        self.n2 = 1.5
        self.A = (-2, 5)
        self.B = (3, -5)
        
        self.init_ui()
        self.reset_simulation()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        # Панель управления
        control_panel = QHBoxLayout()
        
        params = [
            ("n1:", "1.0"), ("n2:", "1.5"),
            ("A x:", "-2"), ("A y:", "5"),
            ("B x:", "3"), ("B y:", "-5")
        ]
        
        self.inputs = {}
        for label, default in params:
            lbl = QLabel(label)
            inp = QLineEdit(default)
            self.inputs[label.replace(" ", "")] = inp
            control_panel.addWidget(lbl)
            control_panel.addWidget(inp)

        self.btn_update = QPushButton("Обновить")
        self.btn_update.clicked.connect(self.update_parameters)
        control_panel.addWidget(self.btn_update)

        self.btn_start = QPushButton("Старт")
        self.btn_start.clicked.connect(self.start_animation)
        self.btn_stop = QPushButton("Стоп")
        self.btn_stop.clicked.connect(self.stop_animation)
        control_panel.addWidget(self.btn_start)
        control_panel.addWidget(self.btn_stop)

        layout.addLayout(control_panel)

        # Холст для рисования
        self.canvas = CanvasWidget(self)
        layout.addWidget(self.canvas)

        # Таймер анимации
        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_step)

    def update_parameters(self):
        try:
            self.n1 = float(self.inputs["n1:"].text())
            self.n2 = float(self.inputs["n2:"].text())
            self.A = (
                float(self.inputs["Ax:"].text()),
                float(self.inputs["Ay:"].text())
            )
            self.B = (
                float(self.inputs["Bx:"].text()),
                float(self.inputs["By:"].text())
            )
            self.reset_simulation()
        except ValueError:
            pass

    def reset_simulation(self):
        self.canvas.n1 = self.n1
        self.canvas.n2 = self.n2
        self.canvas.A = self.A
        self.canvas.B = self.B
        self.canvas.reset()
        self.canvas.update()

    def start_animation(self):
        self.timer.start(500)

    def stop_animation(self):
        self.timer.stop()

    def animation_step(self):
        if not self.canvas.next_iteration():
            self.timer.stop()
        self.canvas.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pan_start = event.pos()
            self.panning = True
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.pan_start
            self.view_offset += delta
            self.pan_start = event.pos()
            self.canvas.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.scale *= zoom_factor
        else:
            self.scale /= zoom_factor
        self.canvas.update()

class CanvasWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.n1 = 1.0
        self.n2 = 1.5
        self.A = (-2, 5)
        self.B = (3, -5)
        self.history = []
        self.true_x = None
        self.reset()

    def reset(self):
        self.history = []
        self.true_x = None
        self.generator = self.golden_section_generator()
        next(self.generator)

    def golden_section_generator(self):
        gr = (math.sqrt(5) + 1) / 2
        a = min(self.A[0], self.B[0])
        b = max(self.A[0], self.B[0])
        c = b - (b - a)/gr
        d = a + (b - a)/gr
        fc = self.time_function(c)
        fd = self.time_function(d)
        
        while abs(b - a) > 1e-5:
            yield {'a': a, 'b': b, 'c': c, 'd': d, 'fc': fc, 'fd': fd}
            
            if fc < fd:
                b = d
            else:
                a = c
            
            c = b - (b - a)/gr
            d = a + (b - a)/gr
            fc = self.time_function(c)
            fd = self.time_function(d)
        
        self.true_x = (a + b)/2
        yield True

    def time_function(self, x):
        d1 = math.sqrt((x - self.A[0])**2 + (0 - self.A[1])**2)
        d2 = math.sqrt((self.B[0] - x)**2 + (self.B[1] - 0)**2)
        return self.n1 * d1 + self.n2 * d2

    def next_iteration(self):
        try:
            state = next(self.generator)
            if state is not True:
                self.history.append(state)
                return True
            return False
        except StopIteration:
            return False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Получаем параметры преобразования из главного окна
        main_window = self.parent().parent()
        offset = main_window.view_offset
        scale = main_window.scale

        # Преобразование координат
        def map_point(x, y):
            return QPoint(
                int(offset.x() + x * scale),
                int(offset.y() - y * scale))
        
        # Отрисовка границы сред
        painter.setPen(QPen(Qt.black, 2))
        left_bound = map_point(-100, 0)
        right_bound = map_point(100, 0)
        painter.drawLine(left_bound, right_bound)

        # Отрисовка точек A и B
        a_scr = map_point(*self.A)
        b_scr = map_point(*self.B)
        painter.setPen(QPen(Qt.blue, 10))
        painter.drawPoint(a_scr)
        painter.drawPoint(b_scr)
        painter.setPen(QPen(Qt.blue, 1))
        painter.drawText(a_scr.x() - 20, a_scr.y() - 10, "A")
        painter.drawText(b_scr.x() - 20, b_scr.y() - 10, "B")

        # Отрисовка пробных лучей
        for i, state in enumerate(self.history):
            alpha = int(30 + 70 * i/len(self.history)) if self.history else 255
            color = QColor(150, 150, 150)
            color.setAlpha(alpha)
            
            c_scr = map_point(state['c'], 0)
            d_scr = map_point(state['d'], 0)
            
            painter.setPen(QPen(color, 1))
            
            # Луч через точку c
            painter.drawLine(a_scr, c_scr)
            painter.drawLine(c_scr, b_scr)
            # Отображение времени для c
            time_color = QColor(90, 90, 90, alpha)
            painter.setPen(QPen(time_color, 1))
            painter.drawText(c_scr.x() + 5, c_scr.y() - 10, f"t={state['fc']:.6f}")
            
            # Луч через точку d
            painter.setPen(QPen(color, 1))
            painter.drawLine(a_scr, d_scr)
            painter.drawLine(d_scr, b_scr)
            # Отображение времени для d
            painter.setPen(QPen(time_color, 1))
            painter.drawText(d_scr.x() + 5, d_scr.y() - 10, f"t={state['fd']:.6f}")

        # Отрисовка истинного пути
        if self.true_x is not None:
            mid_scr = map_point(self.true_x, 0)
            
            # Оптимальный путь
            painter.setPen(QPen(Qt.red, 3))
            painter.drawLine(a_scr, mid_scr)
            painter.drawLine(mid_scr, b_scr)
            
            # Расчет углов
            theta1 = math.atan2(self.true_x - self.A[0], self.A[1])
            theta2 = math.atan2(self.B[0] - self.true_x, abs(self.B[1]))
            
            
            # Информационная панель
            self.draw_info_panel(painter, theta1, theta2)
        

    def draw_info_panel(self, painter, theta1, theta2):
        true_time = self.time_function(self.true_x)
        snell_lhs = self.n1 * math.sin(theta1)
        snell_rhs = self.n2 * math.sin(theta2)
        
        info = [
            f"Время прохождения: {true_time:.7f}",
            f"n₁⋅sinθ₁ = {snell_lhs:.4f}",
            f"n₂⋅sinθ₂ = {snell_rhs:.4f}",
            "✓ Закон Снеллиуса выполнен" if abs(snell_lhs - snell_rhs) < 1e-3 else "✗ Ошибка Снеллиуса"
        ]
        
        painter.setPen(QPen(Qt.black, 1))
        painter.setFont(QFont('Arial', 12, QFont.Bold))
        
        # Фон для текста
        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
        text_width = max(painter.fontMetrics().width(line) for line in info)
        painter.drawRect(
            self.width() - text_width - 30, 20,
            text_width + 20, len(info) * 25 + 10
        )
        
        # Текст
        for i, line in enumerate(info):
            painter.drawText(
                self.width() - text_width - 20, 45 + i * 25,
                line
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpticalSimulator()
    window.show()
    sys.exit(app.exec_())

# Воздух: n ≈ 1.0
# Вода: n ≈ 1.33
# Стекло: n ≈ 1.5
# Алмаз: n ≈ 2.4