import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QSizePolicy, QDoubleSpinBox, QComboBox)
from PyQt5.QtCore import Qt
from matplotlib.patches import Rectangle, Arc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class HorseshoeMagnet:
    def __init__(self, a, b, d, M, Na=20, Nb=20):
        self.a = a
        self.b = b 
        self.d = d
        self.M = M
        self.Na = Na
        self.Nb = Nb
        self.ax = a / Na
        self.ay = b / Nb
        self.c = M / (4 * np.pi) * self.ax * self.ay

    def H_ext(self, xm, ym, zm):
        Hx, Hy, Hz = 0.0, 0.0, 0.0

        for ip in range(1, 3):
            for ia in range(1, self.Na + 1):
                x = self.ax / 2 + (ia - 1) * self.ax + (ip - 1) * self.d
                for ib in range(1, self.Nb + 1):
                    y = self.ay / 2 + (ib - 1) * self.ay
                    r2 = zm**2 + (x - xm)**2 + (y - ym)**2
                    r = np.sqrt(r2)
                    r3 = r * r2
                    k = 3 - 2 * ip

                    Hx += self.c * (xm - x) * k / r3
                    Hy += self.c * (ym - y) * k / r3
                    Hz += self.c * zm * k / r3

        return Hx, Hy, Hz

    def plot_field(self, plane='Y=0', x_range=(-0.5, 0.7), y_range=(-0.5, 0.5), z_range=(-0.5, 0.5), num_points=30):
        if plane == 'Y=0':
            y_coord = 0.0
            x = np.linspace(x_range[0], x_range[1], num_points)
            z = np.linspace(z_range[0], z_range[1], num_points)
            X, Z = np.meshgrid(x, z)

            Hx, Hy, Hz = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Hx[i,j], Hy[i,j], Hz[i,j] = self.H_ext(X[i,j], y_coord, Z[i,j])

            H_magnitude = np.sqrt(Hx**2 + Hz**2)
            Hx_norm, Hz_norm = Hx / H_magnitude, Hz / H_magnitude

            self.axes.clear()

            # Горизонтальное расположение магнита (вид сверху)
            self.axes.add_patch(Rectangle((0, -self.b/2), self.a, self.b,
                                    color='blue', alpha=0.4, label='Северный полюс'))
            self.axes.add_patch(Rectangle((self.d, -self.b/2), self.a, self.b,
                                    color='red', alpha=0.4, label='Южный полюс'))
            self.axes.add_patch(Rectangle((self.a, -self.b/2), self.d-self.a, self.b,
                                    color='gray', alpha=0.3, label='Соединение'))

            self.axes.quiver(X, Z, Hx_norm, Hz_norm, angles='xy', scale_units='xy', scale=15, pivot='mid')
            self.axes.set_xlabel("x, м")
            self.axes.set_ylabel("z, м")
            self.axes.set_title(f"Магнитное поле в плоскости Y=0\nРазмеры: a={self.a}, b={self.b}, d={self.d}")
            self.axes.set_xlim(x_range)
            self.axes.set_ylim(z_range)
            self.axes.set_aspect('equal')
            self.axes.legend()
            self.canvas.draw()

        elif plane == 'Z=0':
            z_coord = 0.0
            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)
            X, Y = np.meshgrid(x, y)

            Hx, Hy, Hz = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Hx[i,j], Hy[i,j], Hz[i,j] = self.H_ext(X[i,j], Y[i,j], z_coord)

            H_magnitude = np.sqrt(Hx**2 + Hy**2)
            Hx_norm, Hy_norm = Hx / H_magnitude, Hy / H_magnitude
            self.axes.clear()

            # Вертикальное расположение магнита (вид сбоку)
            pole_height = self.b * 1.5
            arc_radius = self.d/2 + self.a
            
            # Дуга подковы
            self.axes.add_patch(Arc((self.d/2, 0), width=arc_radius*2, height=pole_height*2, 
                               theta1=0, theta2=180, color='gray', linewidth=10, alpha=0.3))
            
            # Ножки подковы
            self.axes.add_patch(Rectangle((0, -pole_height/2), self.a, pole_height/2,
                              color='blue', alpha=0.4, label='Северный полюс'))
            self.axes.add_patch(Rectangle((self.d, -pole_height/2), self.a, pole_height/2,
                              color='blue', alpha=0.4))
            self.axes.add_patch(Rectangle((0, -pole_height), self.a, pole_height/2,
                              color='red', alpha=0.4, label='Южный полюс'))
            self.axes.add_patch(Rectangle((self.d, -pole_height), self.a, pole_height/2,
                              color='red', alpha=0.4))

            self.axes.quiver(X, Y, Hx_norm, Hy_norm, angles='xy', scale_units='xy', scale=15, pivot='mid')
            self.axes.set_xlabel("x, м")
            self.axes.set_ylabel("y, м")
            self.axes.set_title(f"Магнитное поле в плоскости Z=0\nРазмеры: a={self.a}, b={self.b}, d={self.d}")
            self.axes.set_xlim(x_range)
            self.axes.set_ylim(y_range)
            self.axes.set_aspect('equal')
            self.axes.legend()
            self.canvas.draw()


class MagnetGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Визуализатор магнитного поля подковы")
        
        # Параметры ввода
        self.create_input_widgets()
        
        # График
        self.fig, self.axes = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Разметка
        self.setup_layout()
        
        # Начальный график
        self.update_plot()

    def create_input_widgets(self):
        self.a_label = QLabel("Ширина полюса a (м):")
        self.a_edit = QLineEdit("0.1")
        self.b_label = QLabel("Высота полюса b (м):") 
        self.b_edit = QLineEdit("0.05")
        self.d_label = QLabel("Расстояние d (м):")
        self.d_edit = QLineEdit("0.2")
        self.M_label = QLabel("Намагниченность M (А/м):")
        self.M_edit = QLineEdit("1e6")
        
        self.plane_label = QLabel("Плоскость:")
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["Y=0", "Z=0"])
        
        self.update_button = QPushButton("Обновить график")
        self.update_button.clicked.connect(self.update_plot)

    def setup_layout(self):
        input_layout = QVBoxLayout()
        for widget in [self.a_label, self.a_edit, self.b_label, self.b_edit,
                      self.d_label, self.d_edit, self.M_label, self.M_edit,
                      self.plane_label, self.plane_combo, self.update_button]:
            input_layout.addWidget(widget)
            
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(plot_layout)
        
        self.setLayout(main_layout)

    def update_plot(self):
        try:
            a = float(self.a_edit.text())
            b = float(self.b_edit.text())
            d = float(self.d_edit.text())
            M = float(self.M_edit.text())
            plane = self.plane_combo.currentText()
            
            self.magnet = HorseshoeMagnet(a, b, d, M)
            self.magnet.axes = self.axes
            self.magnet.canvas = self.canvas
            
            # Увеличиваем диапазон для отображения поля
            size = max(a, b, d) * 3
            self.magnet.plot_field(plane=plane, 
                                 x_range=(-size, size*2),
                                 y_range=(-size, size),
                                 z_range=(-size, size),
                                 num_points=30)
            
        except ValueError as e:
            print(f"Ошибка ввода: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MagnetGUI()
    gui.show()
    sys.exit(app.exec_())
