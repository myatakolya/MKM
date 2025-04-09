import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton,QVBoxLayout, QHBoxLayout, QSizePolicy, QDoubleSpinBox, QComboBox)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class HorseshoeMagnet:
    """
    Класс для вычисления и визуализации магнитного поля подковообразного магнита.
    """

    def __init__(self, a, b, d, M, Na=20, Nb=20):
        """
        Инициализация объекта подковообразного магнита.

        Args:
            a: Ширина прямоугольной площадки, м.
            b: Высота прямоугольной площадки, м.
            d: Расстояние между прямоугольными площадками, м.
            M: Намагниченность, А/м.
            Na: Количество ячеек разбиения по ширине.
            Nb: Количество ячеек разбиения по высоте.
        """
        self.a = a
        self.b = b
        self.d = d
        self.M = M
        self.Na = Na
        self.Nb = Nb
        self.ax = a / Na  # Размеры ячеек по x
        self.ay = b / Nb  # Размеры ячеек по y
        self.c = M / (4 * np.pi) * self.ax * self.ay  # Константа

    def H_ext(self, xm, ym, zm):
        """
        Вычисление магнитного поля H в точке (xm, ym, zm).

        Args:
            xm: x-координата точки, м.
            ym: y-координата точки, м.
            zm: z-координата точки, м.

        Returns:
            Кортеж: (Hx, Hy, Hz) - компоненты вектора магнитного поля.
        """
        Hx, Hy, Hz = 0.0, 0.0, 0.0  # Обнуление сумматоров

        for ip in range(1, 3):  # Два полюса
            for ia in range(1, self.Na + 1):
                x = self.ax / 2 + (ia - 1) * self.ax + (ip - 1) * self.d  # Текущая координата x
                for ib in range(1, self.Nb + 1):
                    y = self.ay / 2 + (ib - 1) * self.ay  # Текущая координата y
                    r2 = zm**2 + (x - xm)**2 + (y - ym)**2
                    r = np.sqrt(r2)
                    r3 = r * r2
                    k = 3 - 2 * ip  # Определение знака фиктивного заряда

                    Hx += self.c * (xm - x) * k / r3
                    Hy += self.c * (ym - y) * k / r3
                    Hz += self.c * zm * k / r3

        return Hx, Hy, Hz

    def plot_field(self, plane='Y=0', x_range=(-0.1, 0.3), y_range=(-0.1, 0.1), z_range=(-0.1, 0.1), num_points=20):
        """
        Изображение картины поля в заданной плоскости.

        Args:
            plane: Плоскость для визуализации ('Y=0' или 'Z=0').
            x_range: Диапазон значений x для графика (tuple).
            y_range: Диапазон значений y для графика (tuple).
            z_range: Диапазон значений z для графика (tuple).
            num_points: Количество точек для построения графика (сетка).
        """

        if plane == 'Y=0':
            y_coord = 0.0  # Фиксированная координата y
            x = np.linspace(x_range[0], x_range[1], num_points)
            z = np.linspace(z_range[0], z_range[1], num_points)
            X, Z = np.meshgrid(x, z)  # Создание сетки координат

            Hx, Hy, Hz = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)  # Инициализация массивов для компонент поля
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Hx[i, j], Hy[i, j], Hz[i, j] = self.H_ext(X[i, j], y_coord, Z[i, j])

            # Нормализация векторов для улучшения визуализации
            H_magnitude = np.sqrt(Hx**2 + Hz**2)
            Hx_norm, Hz_norm = Hx / H_magnitude, Hz / H_magnitude

            self.axes.clear() #Clear axes instead of creating new figure

            self.axes.quiver(X, Z, Hx_norm, Hz_norm, angles='xy', scale_units='xy', scale=20, pivot='mid')
            self.axes.set_xlabel("x, м")
            self.axes.set_ylabel("z, м")
            self.axes.set_title(f"Картина магнитного поля (H) в плоскости Y=0\n a={self.a}, b={self.b}, d={self.d}, M={self.M}") #Added parameters to title
            self.axes.set_xlim(x_range)
            self.axes.set_ylim(z_range)
            self.axes.set_aspect('equal', adjustable='box')  # Equal aspect ratio
            self.canvas.draw() #Use existing canvas


        elif plane == 'Z=0':
            z_coord = 0.0  # Фиксированная координата z
            x = np.linspace(x_range[0], x_range[1], num_points)
            y = np.linspace(y_range[0], y_range[1], num_points)
            X, Y = np.meshgrid(x, y)  # Создание сетки координат

            Hx, Hy, Hz = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)  # Инициализация массивов для компонент поля
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Hx[i, j], Hy[i, j], Hz[i, j] = self.H_ext(X[i, j], Y[i, j], z_coord)

            # Нормализация векторов для улучшения визуализации
            H_magnitude = np.sqrt(Hx**2 + Hy**2)
            Hx_norm, Hy_norm = Hx / H_magnitude, Hy / H_magnitude
            self.axes.clear() #Clear axes instead of creating new figure

            self.axes.quiver(X, Y, Hx_norm, Hy_norm, angles='xy', scale_units='xy', scale=20, pivot='mid')
            self.axes.set_xlabel("x, м")
            self.axes.set_ylabel("y, м")
            self.axes.set_title(f"Картина магнитного поля (H) в плоскости Z=0\n a={self.a}, b={self.b}, d={self.d}, M={self.M}") #Added parameters to title
            self.axes.set_xlim(x_range)
            self.axes.set_ylim(y_range)
            self.axes.set_aspect('equal', adjustable='box')  # Equal aspect ratio
            self.canvas.draw() #Use existing canvas
        else:
            print("Ошибка: Неверно указана плоскость. Используйте 'Y=0' или 'Z=0'.")


class MagnetGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Horseshoe Magnet Field Visualizer")

        # --- Input Parameters ---
        self.a_label = QLabel("Ширина a (м):")
        self.a_edit = QLineEdit("0.1")

        self.b_label = QLabel("Высота b (м):")
        self.b_edit = QLineEdit("0.05")

        self.d_label = QLabel("Расстояние d (м):")
        self.d_edit = QLineEdit("0.2")

        self.M_label = QLabel("Намагниченность M (А/м):")
        self.M_edit = QLineEdit("1e6")

        self.Na_label = QLabel("Разбиение Na:")
        self.Na_edit = QLineEdit("20")

        self.Nb_label = QLabel("Разбиение Nb:")
        self.Nb_edit = QLineEdit("20")

        self.plane_label = QLabel("Плоскость:")
        self.plane_combo = QComboBox()
        self.plane_combo.addItem("Y=0")
        self.plane_combo.addItem("Z=0")

        self.update_button = QPushButton("Обновить график")
        self.update_button.clicked.connect(self.update_plot)


        # --- Matplotlib Setup ---
        self.fig, self.axes = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Layout ---
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.a_label)
        input_layout.addWidget(self.a_edit)
        input_layout.addWidget(self.b_label)
        input_layout.addWidget(self.b_edit)
        input_layout.addWidget(self.d_label)
        input_layout.addWidget(self.d_edit)
        input_layout.addWidget(self.M_label)
        input_layout.addWidget(self.M_edit)
        input_layout.addWidget(self.Na_label)
        input_layout.addWidget(self.Na_edit)
        input_layout.addWidget(self.Nb_label)
        input_layout.addWidget(self.Nb_edit)
        input_layout.addWidget(self.plane_label)
        input_layout.addWidget(self.plane_combo)
        input_layout.addWidget(self.update_button)


        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)


        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(plot_layout)

        self.setLayout(main_layout)

        # --- Initial Plot ---
        self.update_plot()

    def update_plot(self):
         """Обновляет график с новыми параметрами."""
         try:
             a = float(self.a_edit.text())
             b = float(self.b_edit.text())
             d = float(self.d_edit.text())
             M = float(self.M_edit.text())
             Na = int(self.Na_edit.text())
             Nb = int(self.Nb_edit.text())
             plane = self.plane_combo.currentText()

             # Создаем объект подковообразного магнита с новыми параметрами
             self.magnet = HorseshoeMagnet(a, b, d, M, Na, Nb) #Store magnet in class
             self.magnet.axes = self.axes  # Pass axes object
             self.magnet.canvas = self.canvas #Pass canvas object

             # Строим картину поля
             self.magnet.plot_field(plane=plane, x_range=(-0.2, 0.4), y_range=(-0.2, 0.2), z_range=(-0.2, 0.2), num_points=25)


         except ValueError:
             print("Ошибка: Введите числовые значения.")
         except Exception as e:
             print(f"Произошла ошибка: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MagnetGUI()
    gui.show()
    sys.exit(app.exec_())