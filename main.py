# main.py
import inspect
from math import sqrt, exp, log
import sys
import matplotlib.pyplot as plt

from matrix import solve_sle

# ==================== АППРОКСИРОВАНИЕ ДАННЫХ ====================

# Линейная аппроксимация: модель вида y = a + b * x
# xs, ys – списки входных данных; n – количество точек.
def linear_approximation(xs, ys, n):
    # Вычисляем суммы для нормальных уравнений:
    # sx = Σ x_i, sxx = Σ x_i^2, sy = Σ y_i, sxy = Σ x_i * y_i
    sx = sum(xs)
    sxx = sum(x ** 2 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))

    # Решаем систему уравнений размера 2x2 методом solve_sle:
    # [n sx] [a] = [sy]
    # [sx sxx] [b] = [sxy]
    a, b = solve_sle(
        [[n, sx], [sx, sxx]],  # матрица коэффициентов
        [sy, sxy],             # столбец свободных членов
        2                       # размерность системы
    )
    # Возвращаем функцию f(x) и найденные коэффициенты a, b
    return lambda xi: a + b * xi, a, b

# Квадратичная аппроксимация: модель y = a + b*x + c*x^2
def quadratic_approximation(xs, ys, n):
    # Вычисляем моменты и смешанные суммы
    sx = sum(xs)
    sxx = sum(x ** 2 for x in xs)
    sxxx = sum(x ** 3 for x in xs)
    sxxxx = sum(x ** 4 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxxy = sum(x ** 2 * y for x, y in zip(xs, ys))

    # Система 3x3 для коэффициентов a, b, c
    a, b, c = solve_sle(
        [
            [n, sx, sxx],
            [sx, sxx, sxxx],
            [sxx, sxxx, sxxxx]
        ],
        [sy, sxy, sxxy],
        3
    )
    return lambda xi: a + b * xi + c * xi ** 2, a, b, c

# Кубическая аппроксимация: модель y = a + b*x + c*x^2 + d*x^3
def cubic_approximation(xs, ys, n):
    # Моменты до порядка 6
    sx = sum(xs)
    sxx = sum(x ** 2 for x in xs)
    sxxx = sum(x ** 3 for x in xs)
    sxxxx = sum(x ** 4 for x in xs)
    sxxxxx = sum(x ** 5 for x in xs)
    sxxxxxx = sum(x ** 6 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxxy = sum(x ** 2 * y for x, y in zip(xs, ys))
    sxxxy = sum(x ** 3 * y for x, y in zip(xs, ys))

    # Система 4x4
    a, b, c, d = solve_sle(
        [
            [n, sx, sxx, sxxx],
            [sx, sxx, sxxx, sxxxx],
            [sxx, sxxx, sxxxx, sxxxxx],
            [sxxx, sxxxx, sxxxxx, sxxxxxx]
        ],
        [sy, sxy, sxxy, sxxxy],
        4
    )
    return lambda xi: a + b * xi + c * xi ** 2 + d * xi ** 3, a, b, c, d

# Экспоненциальная аппроксимация: модель y = A * exp(b*x)
def exponential_approximation(xs, ys, n):
    # Преобразуем задачу: ln(y) = ln(A) + b*x
    ys_log = [log(y) for y in ys]
    # Линейная аппроксимация для (x, ln y)
    _, a_log, b = linear_approximation(xs, ys_log, n)
    A = exp(a_log)
    return lambda xi: A * exp(b * xi), A, b

# Логарифмическая аппроксимация: модель y = a + b*ln(x)
def logarithmic_approximation(xs, ys, n):
    # Преобразуем x: ln(x)
    xs_log = [log(x) for x in xs]
    # Линейная аппроксимация для (ln x, y)
    _, a, b = linear_approximation(xs_log, ys, n)
    return lambda xi: a + b * log(xi), a, b

# Степенная аппроксимация: модель y = A * x^b
def power_approximation(xs, ys, n):
    # Преобразуем: ln(y) = ln(A) + b*ln(x)
    xs_log = [log(x) for x in xs]
    ys_log = [log(y) for y in ys]
    _, a_log, b = linear_approximation(xs_log, ys_log, n)
    A = exp(a_log)
    return lambda xi: A * xi ** b, A, b

# ==================== МЕТРИКИ КАЧЕСТВА МОДЕЛЕЙ ====================

def compute_pearson_correlation(x, y, n):
    """Коэффициент корреляции Пирсона: линейная связь двух переменных"""
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sqrt(
        sum((xi - mean_x) ** 2 for xi in x) *
        sum((yi - mean_y) ** 2 for yi in y)
    )
    return num / den


def compute_mean_squared_error(x, y, fi, n):
    """Среднеквадратичное отклонение ошибок RMSE"""
    return sqrt(sum((fi(xi) - yi) ** 2 for xi, yi in zip(x, y)) / n)


def compute_measure_of_deviation(x, y, fi, n):
    """Сумма квадратов отклонений S - мера отклонения"""
    return sum((fi(xi) - yi) ** 2 for xi, yi in zip(x, y))


def compute_coefficient_of_determination(xs, ys, fi, n):
    """Коэффициент детерминации R^2"""
    mean_pred = sum(fi(xi) for xi in xs) / n
    ss_res = sum((yi - fi(xi)) ** 2 for xi, yi in zip(xs, ys))
    ss_tot = sum((yi - mean_pred) ** 2 for yi in ys)
    return 1 - ss_res / ss_tot

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_str_content_of_func(func):
    """Извлечение текстового выражения лямбда-функции для вывода"""
    source_line = inspect.getsourcelines(func)[0][0]
    return source_line.split('lambda xi:')[-1].strip()


def draw_plot(x, y):
    """Отображение исходных точек на графике"""
    plt.scatter(x, y, label="Вводные точки")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Приближение разными методами")
    plt.show()


def draw_func(func, name, x, dx=0.001):
    """Построение линии аппроксимирующей функции на интервале"""
    start, end = x[0] - 0.1, x[-1] + 0.1
    xs_plot, ys_plot = [], []
    t = start
    while t <= end:
        xs_plot.append(t)
        ys_plot.append(func(t))
        t += dx
    plt.plot(xs_plot, ys_plot, label=name)

# Функция run реализует полный цикл исследования:
# - выбирает лучшую модель по RMSE
# - выводит метрики и формулы
# - строит графики всех моделей

def run(functions, x, y, n):
    best_mse = float('inf')
    best_func = None
    mses = []

    for approximation, name in functions:
        try:
            # Получаем функцию и ее коэффициенты
            fi, *coeffs = approximation(x, y, n)
            # Вычисляем метрики качества
            s = compute_measure_of_deviation(x, y, fi, n)
            mse = compute_mean_squared_error(x, y, fi, n)
            r2 = compute_coefficient_of_determination(x, y, fi, n)

            # Обновляем лучшую модель по MSE
            if mse <= best_mse:
                best_mse = mse
                best_func = name
                mses.append((mse, name))

            # Рисуем график этой модели
            draw_func(fi, name, x)

            # Выводим информацию по модели
            print(f"{name} функция:")
            print(f"  Функция: f(x) = {get_str_content_of_func(fi)}")
            # print(f"  X: {x}")
            # print(f"  Y: {y}")
            # print(f"  fi: {[round(fi(val), 3) for val in x]}")
            print(f"  Коэф.: {coeffs}")
            print(f"  RMSE = {mse:.5f}, R^2 = {r2:.5f}")

            # Для линейной модели выводим корреляцию Пирсона
            if approximation == linear_approximation:
                r = compute_pearson_correlation(x, y, n)
                print(f"  Pearson r = {r:.5f}")

        except Exception as e:
            print(f"Ошибка аппроксимации {name}: {e}")
        print('-' * 40)

    # Отображаем лучшую модель
    print(f"Лучшая модель: {best_func}")
    # Отображаем исходные точки поверх всех графиков
    draw_plot(x, y)

# Функции чтения данных из файла и консоли с проверкой ошибок
def read_data_from_file(filename):
    try:
        x, y = [], []
        with open(filename) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
        return x, y, None
    except IOError as err:
        return None, None, f"Ошибка чтения файла: {err}"


def read_data_from_input():
    # Ввод точек до команды 'quit'
    x, y = [], []
    while True:
        line = input()
        if line.strip().lower() == 'quit':
            break
        parts = line.strip().split()
        if len(parts) == 2:
            x.append(float(parts[0]))
            y.append(float(parts[1]))
        else:
            print("Неправильный формат точки, пропускаем.")
    return x, y

# Точка входа: выбор источника данных и запуск анализа
def main():
    # Выбор ввода: файл или консоль
    while True:
        mode = input("Ввод из файла (f) или с клавиатуры (t)? ")
        if mode == 'f':
            fname = input("Имя файла: ")
            x, y, err = read_data_from_file(fname)
            if err:
                print(err)
                continue
            break
        elif mode == 't':
            print("Введите точки в формате 'x y', 'quit' для завершения:")
            x, y = read_data_from_input()
            break
        else:
            print("Введите 'f' или 't'.")

    n = len(x)
    # Выбор функций в зависимости от положительности данных
    functions = [(linear_approximation, "Линейная"),
                 (quadratic_approximation, "Квадратичная"),
                 (cubic_approximation, "Кубическая")]
    # Добавляем экспоненциальную, логарифмическую или степенную, если можно
    if all(val > 0 for val in y):
        functions.append((exponential_approximation, "Экспоненциальная"))
    if all(val > 0 for val in x):
        functions.append((logarithmic_approximation, "Логарифмическая"))
        if all(val > 0 for val in y):
            functions.append((power_approximation, "Степенная"))

    # Выбор вывода
    out_mode = input("Вывод в файл (f) или терминал (t)? ")
    if out_mode == 'f':
        sys.stdout = open('out.txt', 'w')

    # Запуск анализа
    run(functions, x, y, n)

if __name__ == '__main__':
    main()
    print("Конец работы программы.")