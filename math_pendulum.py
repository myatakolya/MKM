import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation

class Physicist:
    def __init__(self, g=9.81, length=1.0):
        self.g = g
        self.length = length
    
    def huygens_formula(self):
        """Вычисляет период колебаний по формуле Гюйгенса."""
        return 2 * np.pi * np.sqrt(self.length / self.g)
    
    def equation(self, theta, omega, damping):
        """Дифференциальное уравнение маятника."""
        return - (self.g / self.length) * np.sin(theta) - damping * omega

class Mathematician:
    def __init__(self, physicist, theta0, omega0, step, damping, points):
        self.physicist = physicist
        self.theta0 = theta0
        self.omega0 = omega0
        self.step = step
        self.damping = damping
        self.points = points
    
    def integrate(self):
        """Интегрирует дифференциальное уравнение маятника."""
        theta, omega = self.theta0, self.omega0
        t_values, theta_values, omega_values = [0], [theta], [omega]
        
        for i in range(1, self.points):
            omega += self.physicist.equation(theta, omega, self.damping) * self.step
            theta += omega * self.step
            t_values.append(i * self.step)
            theta_values.append(theta)
            omega_values.append(omega)
        
        return np.array(t_values), np.array(theta_values), np.array(omega_values) #Преобразуем списки в массивы

    def compute_period(self, t_values, theta_values):
        """Вычисляет период колебаний на основе данных моделирования."""
        # Находим все пересечения нуля (сверху вниз)
        crossings = np.where((theta_values[:-1] >= 0) & (theta_values[1:] < 0))[0]

        # Если нет пересечений или только одно, возвращаем None
        if len(crossings) < 2:
            return None

        # Вычисляем периоды между последовательными пересечениями
        periods = np.diff(t_values[crossings])

        # Возвращаем средний период
        return np.mean(periods)

class PendulumApp(QWidget):
    def __init__(self):
        super().__init__()
        self.physicist = Physicist() # выносим физика сюда, чтобы он не пересоздавался при каждом расчете
        self.ani = None # Инициализируем self.ani как None
        self.t_values, self.theta_values, self.omega_values = None, None, None # Инициализируем данные графиков
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout()
        
        # Левая панель - ввод данных и вывод результатов
        control_panel = QVBoxLayout()
        
        self.length_input = QLineEdit("1.0")  # Длина маятника
        self.theta_input = QLineEdit("0.2")
        self.omega_input = QLineEdit("0.0")
        self.step_input = QLineEdit("0.01")
        self.damping_input = QLineEdit("0.05")
        self.points_input = QLineEdit("1000")
        
        control_panel.addWidget(QLabel("Длина маятника (м):")) # new
        control_panel.addWidget(self.length_input)   # new
        control_panel.addWidget(QLabel("Начальное отклонение (рад):"))
        control_panel.addWidget(self.theta_input)
        control_panel.addWidget(QLabel("Начальная скорость (рад/с):"))
        control_panel.addWidget(self.omega_input)
        control_panel.addWidget(QLabel("Шаг интегрирования:"))
        control_panel.addWidget(self.step_input)
        control_panel.addWidget(QLabel("Декремент затухания:"))
        control_panel.addWidget(self.damping_input)
        control_panel.addWidget(QLabel("Число точек:"))
        control_panel.addWidget(self.points_input)
        
        self.start_button = QPushButton("Пуск")
        self.stop_button = QPushButton("Стоп")
        self.refresh_button = QPushButton("Обновить")
        
        control_panel.addWidget(self.start_button)
        control_panel.addWidget(self.stop_button)
        control_panel.addWidget(self.refresh_button)
        
        # Вывод результатов
        self.iterations_label = QLabel("Итерации: 0")
        self.oscillations_label = QLabel("Колебания: 0")
        self.huygens_label = QLabel("Формула Гюйгенса: 0")
        self.computed_period_label = QLabel("Вычисленный период: 0") #убрали "Точное значение периода", так как его не существует

        control_panel.addWidget(self.iterations_label)
        control_panel.addWidget(self.oscillations_label)
        control_panel.addWidget(self.huygens_label)
        control_panel.addWidget(self.computed_period_label)
        
        layout.addLayout(control_panel)
        
        # Правая панель - графики и маятник
        self.figure, (self.ax_pendulum, self.ax1, self.ax2) = plt.subplots(3, 1, figsize=(5, 7))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        self.setWindowTitle("Маятник")
        self.setGeometry(100, 100, 800, 600)
        
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.refresh_button.clicked.connect(self.refresh_simulation)
    
    def start_simulation(self):
        try:
            length = float(self.length_input.text())
            theta0 = float(self.theta_input.text())
            omega0 = float(self.omega_input.text())
            step = float(self.step_input.text())
            damping = float(self.damping_input.text())
            points = int(self.points_input.text())
        except ValueError:
            print("Ошибка: Некорректный ввод чисел.")
            return
        
        #Обновляем длину маятника
        self.physicist.length = length

        # Создаем Mathematician
        mathematician = Mathematician(self.physicist, theta0, omega0, step, damping, points)
        
        # Интегрируем и получаем результаты
        self.t_values, self.theta_values, self.omega_values = mathematician.integrate()
        
        # Вычисляем период Гюйгенса и период из моделирования
        huygens_period = self.physicist.huygens_formula()
        
        #Повышаем точность оценки периода, беря больше циклов колебаний
        num_cycles = 5  # Например, 5 циклов
        if len(self.t_values) > 2 and num_cycles > 0:
           end_index = min(len(self.t_values), int(num_cycles * huygens_period / mathematician.step))  # Index for num_cycles
           computed_period = mathematician.compute_period(self.t_values[:end_index], self.theta_values[:end_index])  # compute period using a limited range
        else:
           computed_period = None

        if computed_period is None:
            computed_period_str = "Недостаточно данных для вычисления"
        else:
            computed_period_str = f"{computed_period:.6f}" # Форматируем в строку (больше знаков после запятой)

        # Обновляем labels в GUI
        self.iterations_label.setText(f"Итерации: {points}")
        self.oscillations_label.setText(f"Приблизительно колебаний: {int(points * float(self.step_input.text()) / huygens_period)}") # Улучшенная оценка
        self.huygens_label.setText(f"Период (Гюйгенс): {huygens_period:.6f}") # Больше знаков
        self.computed_period_label.setText(f"Вычисленный период: {computed_period_str}")

        # Запускаем анимацию (останавливаем предыдущую, если есть)
        self.refresh_simulation() #Останавливаем и очищаем перед стартом
        self.ani = FuncAnimation(self.figure, self.update_plot, frames=len(self.t_values), interval=5, repeat=False)
    
    def stop_simulation(self):
        if self.ani is not None:
            self.ani.event_source.stop()
    
    def refresh_simulation(self):
        self.stop_simulation() # остановка перед очисткой
        self.ax_pendulum.clear()
        self.ax1.clear()
        self.ax2.clear()
        
        #Сохраняем пределы осей
        if self.t_values is not None and self.theta_values is not None:
            self.ax1.set_xlim(0, self.t_values[-1])
            self.ax1.set_ylim(np.min(self.theta_values), np.max(self.theta_values))

        self.canvas.draw()
    
    def update_plot(self, i):
        try:  # Добавим блок try-except для обработки возможных ошибок
            self.ax_pendulum.clear()
            self.ax1.clear()
            self.ax2.clear()
            
            x = np.sin(self.theta_values[i])
            y = -np.cos(self.theta_values[i])
            
            self.ax_pendulum.plot([0, x], [0, y], 'k-', lw=2)
            self.ax_pendulum.plot(x, y, 'ro', markersize=10)
            self.ax_pendulum.set_xlim(-1.2, 1.2)
            self.ax_pendulum.set_ylim(-1.2, 0.2)
            self.ax_pendulum.set_title("Маятник")
            self.ax_pendulum.set_aspect('equal')
            
            self.ax1.plot(self.t_values[:i], self.theta_values[:i], 'b')
            self.ax1.set_title("Колебания")
            self.ax1.set_xlabel("Время (с)") # Добавлено
            self.ax1.set_ylabel("Угол (рад)") # Добавлено

            #Сохраняем пределы графика колебаний
            self.ax1.set_xlim(0, self.t_values[-1])
            self.ax1.set_ylim(np.min(self.theta_values), np.max(self.theta_values))


            self.ax2.plot(self.theta_values[:i], self.omega_values[:i], 'r')
            self.ax2.set_title("Фазовый портрет")
            self.ax2.set_xlabel("Угол (рад)")     # Добавлено
            self.ax2.set_ylabel("Угловая скорость (рад/с)") # Добавлено

            #Сохраняем пределы фазового портрета (менее важно, но можно)
            #self.ax2.set_xlim(np.min(self.theta_values), np.max(self.theta_values))
            #self.ax2.set_ylim(np.min(self.omega_values), np.max(self.omega_values))

            self.figure.tight_layout() # Очень полезно для предотвращения наложения подписей
            self.canvas.draw()
        except Exception as e:
            print(f"Ошибка при обновлении графика: {e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PendulumApp()
    ex.show()
    sys.exit(app.exec_())