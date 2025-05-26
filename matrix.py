# Решение системы линейных уравнений методом Крамера для n=2,3,4

def calc_det2(A):
    """Определитель 2x2 матрицы"""
    return A[0][0]*A[1][1] - A[0][1]*A[1][0]


def solve2(A, B):
    """Решает систему 2 уравнений методом Крамера"""
    det = calc_det2(A)
    # Подстановка B в первый и второй столбцы
    det1 = calc_det2([[B[r], A[r][1]] for r in range(2)])
    det2 = calc_det2([[A[r][0], B[r]] for r in range(2)])
    return det1/det, det2/det


def calc_det3(A):
    """Определитель 3x3 матрицы"""
    pos = A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
    neg = A[0][2]*A[1][1]*A[2][0] + A[0][1]*A[1][0]*A[2][2] + A[0][0]*A[1][2]*A[2][1]
    return pos - neg


def solve3(A, B):
    """Решает систему 3 уравнений методом Крамера"""
    det = calc_det3(A)
    det1 = calc_det3([[B[r], A[r][1], A[r][2]] for r in range(3)])
    det2 = calc_det3([[A[r][0], B[r], A[r][2]] for r in range(3)])
    det3 = calc_det3([[A[r][0], A[r][1], B[r]] for r in range(3)])
    return det1/det, det2/det, det3/det


def calc_det4(A):
    """Рекурсивный расчет определителя 4x4 через разложение по первой строке"""
    res = 0
    sign = 1
    for c in range(4):
        # Формируем минор без строки 0 и столбца c
        minor = [row[:c] + row[c+1:] for row in A[1:]]
        res += sign * A[0][c] * calc_det3(minor)
        sign *= -1
    return res


def solve4(A, B):
    """Решает систему 4 уравнений методом Крамера"""
    det = calc_det4(A)
    dets = []
    for c in range(4):
        # Заменяем столбец c на B
        M = [row.copy() for row in A]
        for r in range(4): M[r][c] = B[r]
        dets.append(calc_det4(M))
    return tuple(d/det for d in dets)


def solve_sle(A, B, n):
    """Выбор метода решения СЛАУ по размерности n"""
    if n == 2:
        return solve2(A, B)
    elif n == 3:
        return solve3(A, B)
    elif n == 4:
        return solve4(A, B)
    else:
        raise ValueError(f"Неподдерживаемая размерность: {n}")
