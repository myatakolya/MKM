import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QTextEdit, QSizePolicy, QDoubleSpinBox, QCheckBox)
from PyQt5.QtCore import Qt


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class Mathematic:
    """
    Класс, содержащий математические функции и константы.
    """
    GRAVITY = 9.81  # Ускорение свободного падения, м/с^2


class Physic:
    """
    Класс для моделирования физических явлений, связанных с движением тела в поле силы тяжести и сопротивлением воздуха.
    """

    def __init__(self, v0, vM, gravity=Mathematic.GRAVITY, dt=0.01):
        """
        Инициализация объекта Physic.

        Args:
            v0: Начальная скорость тела, м/с.
            vM: Предельная скорость тела, м/с (скорость установившегося падения).
            gravity: Ускорение свободного падения, м/с^2.
            dt: Временной шаг для численного интегрирования, с.
        """
        self.v0 = v0
        self.vM = vM
        self.gravity = gravity
        self.dt = dt
        self.g_vec = np.array([0.0, -self.gravity])  # Precompute gravity vector
        self.air_res_coeff = self.gravity / self.vM ** 2  # Precompute air resistance coefficient

    def range_no_air_resistance(self, angle_degrees):
        """
        Вычисляет дальность полета тела без учета сопротивления воздуха.

        Args:
            angle_degrees: Угол вылета в градусах.

        Returns:
            Дальность полета, м.
        """
        angle_radians = math.radians(angle_degrees)
        return (self.v0**2 * math.sin(2 * angle_radians)) / self.gravity

    def optimal_angle_no_air_resistance(self):
        """
        Вычисляет оптимальный угол вылета для максимальной дальности без учета сопротивления воздуха.

        Returns:
            Оптимальный угол в градусах.
        """
        return 45.0

    def predictor_corrector(self, angle_degrees, wind_speed=0):
        """
        Реализует метод Предиктор-Корректор для моделирования движения тела с учетом сопротивления воздуха и ветра (Векторная реализация).

        Args:
            angle_degrees: Угол вылета в градусах.
            wind_speed: Скорость ветра, направленного против оси x, м/с.

        Returns:
            Кортеж: (массив x-координат, массив y-координат, время полета)
        """
        angle_radians = np.radians(angle_degrees)
        # Начальные условия в виде векторов
        pos = np.array([0.0, 0.0])  # [x, y] - вектор позиции (начало координат)
        vel = np.array([self.v0 * np.cos(angle_radians) - wind_speed, self.v0 * np.sin(angle_radians)]) # [vx, vy] - вектор скорости (начальная скорость с учетом ветра)

        # Initial acceleration
        acc = self.g_vec  # Initial acceleration is just gravity (no air resistance initially)

        trajectory = [pos.copy()] # Initial position
        time = 0
        dt = self.dt  # Cache dt for faster access

        while pos[1] >= 0:  # Пока y >= 0

            # Предиктор:
            pos_pred = pos + vel * dt + 0.5 * acc * dt**2  # ri+1 = ri + vi * dt + (1/2) * ai * dt^2 - прогнозируем положение
            vel_pred = vel + acc * dt  # vi+1 = vi + ai * dt - прогнозируем скорость

            # Корректор:
            acc_pred = self.g_vec - (self.air_res_coeff) * np.linalg.norm(vel_pred) * vel_pred  # ai+1 = g - (g / vM^2) * vi+1 - вычисляем ускорение с учетом сопротивления воздуха
            vel = vel + 0.5 * (acc + acc_pred) * dt  # vi+1 = vi + ((ai + ai+1) / 2) * dt - уточняем скорость
            pos = pos + vel * dt #ri+1 = ri + vi * dt  -  уточняем позицию (упрощенно, можно использовать более точную формулу)
            acc = self.g_vec - (self.air_res_coeff) * np.linalg.norm(vel) * vel # Обновляем ускорение для следующей итерации
            
            trajectory.append(pos.copy()) # Store a copy of the position
            time += dt

        trajectory = np.array(trajectory)  # Convert to NumPy array
        return trajectory[:, 0], trajectory[:, 1], time  # Return x, y, and time

    def find_optimal_angle_with_air_resistance(self, wind_speed=0, angle_step=0.1):
        """
        Находит оптимальный угол вылета для максимальной дальности с учетом сопротивления воздуха и ветра.

        Args:
            wind_speed: Скорость ветра, направленного против оси x, м/с.
            angle_step: Шаг изменения угла в градусах.

        Returns:
            Оптимальный угол в градусах.
        """
        angles = np.arange(0, 90, angle_step)
        max_ranges = np.zeros_like(angles)

        for i, angle in enumerate(angles):
            trajectory_x, _, _ = self.predictor_corrector(angle, wind_speed)
            max_ranges[i] = trajectory_x[-1]

        optimal_angle_index = np.argmax(max_ranges) # Index of maximum range
        return angles[optimal_angle_index]

    def vertical_fall_predictor_corrector(self, initial_height):
         """
         Моделирует вертикальное падение тела с учетом сопротивления воздуха с использованием метода предиктор-корректор (векторная версия).

         Args:
             initial_height: Начальная высота тела, м.

         Returns:
             Кортеж: (массив времен, массив высот, массив скоростей)
         """
         y = initial_height
         vy = 0  # Начальная вертикальная скорость
         times = [0]
         heights = [y]
         velocities = [vy]
         time = 0

         while y > 0:
             # Предиктор
             ay = - self.gravity - (self.gravity / self.vM ** 2) * vy * abs(vy)  # Ускорение всегда против скорости
             vy_pred = vy + ay * self.dt
             y_pred = y + vy * self.dt

             # Корректор
             ay_pred = - self.gravity - (self.gravity / self.vM ** 2) * vy_pred * abs(vy_pred)
             vy = vy + 0.5 * (ay + ay_pred) * self.dt
             y = y + 0.5 * (vy + vy_pred) * self.dt

             time += self.dt
             times.append(time)
             heights.append(y)
             velocities.append(vy)

         return times, heights, velocities


class PhysicsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physics Simulation")

        # --- UI Elements ---
        self.v0_label = QLabel("Начальная скорость (v0):")
        self.v0_entry = QLineEdit("750")
        self.vM_label = QLabel("Предельная скорость (vM):")
        self.vM_entry = QLineEdit("150")
        self.wind_label = QLabel("Скорость ветра:")
        self.wind_entry = QLineEdit("20")
        self.height_label = QLabel("Начальная высота:")
        self.height_entry = QLineEdit("100")
        self.angle_label = QLabel("Угол выстрела (градусы):")
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0.0, 90.0)
        self.angle_spinbox.setValue(45.0)
        self.angle_step_label = QLabel("Шаг угла (градусы):")
        self.angle_step_spinbox = QDoubleSpinBox()
        self.angle_step_spinbox.setRange(0.1, 10.0)
        self.angle_step_spinbox.setValue(0.1)

        self.calculate_optimal_button = QPushButton("Найти оптимальный угол")
        self.calculate_optimal_button.clicked.connect(self.calculate_optimal)

        self.calculate_trajectory_button = QPushButton("Рассчитать траекторию")
        self.calculate_trajectory_button.clicked.connect(self.calculate_trajectory)

        self.use_optimal_angle_checkbox = QCheckBox("Использовать оптимальный угол")
        self.use_optimal_angle_checkbox.setChecked(False)

        self.result_label = QLabel("Результаты:")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)

        # --- Matplotlib figures ---
        self.trajectory_fig, self.trajectory_ax = plt.subplots()
        self.trajectory_canvas = FigureCanvas(self.trajectory_fig)
        self.trajectory_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trajectory_toolbar = NavigationToolbar(self.trajectory_canvas, self)

        self.vertical_fig, self.vertical_ax = plt.subplots()
        self.vertical_canvas = FigureCanvas(self.vertical_fig)
        self.vertical_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vertical_toolbar = NavigationToolbar(self.vertical_canvas, self)

        # --- Layout ---
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.v0_label)
        input_layout.addWidget(self.v0_entry)
        input_layout.addWidget(self.vM_label)
        input_layout.addWidget(self.vM_entry)
        input_layout.addWidget(self.wind_label)
        input_layout.addWidget(self.wind_entry)
        input_layout.addWidget(self.height_label)
        input_layout.addWidget(self.height_entry)
        input_layout.addWidget(self.angle_label)
        input_layout.addWidget(self.angle_spinbox)
        input_layout.addWidget(self.angle_step_label)
        input_layout.addWidget(self.angle_step_spinbox)
        input_layout.addWidget(self.calculate_optimal_button)
        input_layout.addWidget(self.calculate_trajectory_button)
        input_layout.addWidget(self.use_optimal_angle_checkbox)

        results_layout = QVBoxLayout()
        results_layout.addWidget(self.result_label)
        results_layout.addWidget(self.result_text)

        trajectory_layout = QVBoxLayout()
        trajectory_layout.addWidget(self.trajectory_toolbar)
        trajectory_layout.addWidget(self.trajectory_canvas)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.vertical_toolbar)
        vertical_layout.addWidget(self.vertical_canvas)

        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(results_layout)
        main_layout.addLayout(trajectory_layout)
        main_layout.addLayout(vertical_layout)

        self.setLayout(main_layout)
        self.optimal_angle = None # Store optimal angle

    def calculate_optimal(self):
        try:
            v0 = float(self.v0_entry.text())
            vM = float(self.vM_entry.text())
            wind_speed = float(self.wind_entry.text())
            angle_step = self.angle_step_spinbox.value()

            # Создаем объект Physic
            physics = Physic(v0=v0, vM=vM)

            # Находим оптимальный угол
            self.optimal_angle = physics.find_optimal_angle_with_air_resistance(wind_speed, angle_step)
            self.result_text.clear()
            self.result_text.insertPlainText(f"Оптимальный угол: {self.optimal_angle:.2f} градусов\n")

        except ValueError:
            self.result_text.clear()
            self.result_text.insertPlainText("Ошибка: Введите числовые значения.")

    def calculate_trajectory(self):
        try:
            v0 = float(self.v0_entry.text())
            vM = float(self.vM_entry.text())  # Not used, but should be read
            wind_speed = float(self.wind_entry.text())
            initial_height = float(self.height_entry.text())

            if self.use_optimal_angle_checkbox.isChecked() and self.optimal_angle is not None:
                angle = self.optimal_angle
            else:
                 angle = self.angle_spinbox.value()

            # Создаем объект Physic
            physics = Physic(v0=v0, vM=vM)

            # Расчет дальности без сопротивления воздуха
            range_no_air = physics.range_no_air_resistance(angle)

            # Расчет траектории с сопротивлением воздуха
            trajectory_x_wind, trajectory_y_wind, _ = physics.predictor_corrector(angle, wind_speed)
            max_range_with_wind = trajectory_x_wind[-1]

            # Расчет вертикального падения
            times, heights, velocities = physics.vertical_fall_predictor_corrector(initial_height)

            # Вывод результатов
            results = f"Угол выстрела: {angle:.2f} градусов\n"
            results += f"Дальность (без сопротивления воздуха): {range_no_air:.2f} метров\n"
            results += f"Максимальная дальность (с ветром): {max_range_with_wind:.2f} метров\n"
            results += f"Время падения: {times[-1]:.2f} секунд\n"
            results += f"Скорость при падении: {velocities[-1]:.2f} м/с"
            self.result_text.clear()
            self.result_text.insertPlainText(results)

            # Обновление графика траектории
            self.trajectory_ax.clear()
            self.trajectory_ax.plot(trajectory_x_wind, trajectory_y_wind, label="Траектория с ветром (predictor-corrector)")
            self.trajectory_ax.set_xlabel("x (м)")
            self.trajectory_ax.set_ylabel("y (м)")
            self.trajectory_ax.set_title("Траектория полета пули с учетом ветра")
            self.trajectory_ax.legend()
            self.trajectory_ax.grid(True)
            self.trajectory_canvas.draw()

            # Обновление графика вертикального падения
            self.vertical_ax.clear()
            self.vertical_ax.plot(times, heights, label="Высота")
            self.vertical_ax.plot(times, velocities, label="Скорость")
            self.vertical_ax.set_xlabel("Время (с)")
            self.vertical_ax.set_ylabel("Высота (м) / Скорость (м/с)")
            self.vertical_ax.set_title("Вертикальное падение с сопротивлением воздуха")
            self.vertical_ax.legend()
            self.vertical_ax.grid(True)
            self.vertical_canvas.draw()

        except ValueError:
            self.result_text.clear()
            self.result_text.insertPlainText("Ошибка: Введите числовые значения.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PhysicsGUI()
    gui.show()
    sys.exit(app.exec_())
