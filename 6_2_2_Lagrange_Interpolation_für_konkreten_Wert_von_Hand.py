import numpy as np


def lagrange_int(x, y, x_int):
    return np.sum([y[i] * np.prod([(x_int - x[j] * 1.0) / (x[i] - x[j]) for j in range(x.shape[0]) if j != i]) for i in range(x.shape[0])])


x = np.array([0, 2500, 5000, 10000])  # xi
y = np.array([1013, 747, 540, 226])   # yi
x_int = 3750  # Zu interpolierende Stelle

n = x.shape[0] - 1

print('Interpolationspolynom vom Grad {}:'.format(n))
print('Pn(x) = ∑[i = 0 .. {}] yi * ℓi(x)'.format(n))
sum_out = ''
for i in range(n + 1):
    sum_out += '{} * ℓ{}(x)'.format(y[i], i)
    if i < n:
        sum_out += ' + '

print('Pn(x) = {} mit x = {}\n'.format(sum_out, x_int))

print('Lagrange-Polynome ℓi:')
print('ℓi(x) = ∏[j = 0 .. n, j ≠ i](x - xj)/(xi - xj)')


for i in range(n + 1):
    prod_out_sym = ''
    prod_out_con = ''
    prod_res = 1

    for j in range(n + 1):
        last = j == n or (i == n and j == n-1)
        if j == i:
            continue

        prod_out_sym += '(x - x{})/(x{} - x{})'.format(j, i, j)
        prod_out_con += '({} - {})/({} - {})'.format(x_int, x[j], x[i], x[j])
        prod_res *= (x_int - x[j])/(x[i] - x[j])

        if not last:
            prod_out_sym += ' * '
            prod_out_con += ' * '

    print('ℓ{}(x) = {} = {} = {}'.format(i, prod_out_sym, prod_out_con, prod_res))

y_int = lagrange_int(x, y, x_int)

print('\nEinsetzen in Pn(x) = {} = {}'.format(sum_out, y_int))

print("f(" + str(x_int) + ") = " + str(y_int))
