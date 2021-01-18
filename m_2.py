def f(x, y):
    return (6 * x ** 6) + (2 * x ** 4 * y ** 2) + (10 * x ** 2) + (6 * x * y) + (10 * y ** 2) - (6 * x) + 4

def df_dx(x, y):
    return 6 * 6 * x ** 5 + 2 * 4 * x ** 3 * y ** 2 + 10 * 2 * x + 6 * y - 6

def df_dy(x, y):
    return 2 * x ** 4 * 2 * y + 6 * x + 20 * y

steps = []
cur_x, cur_y = 1.0, 1.0
num_iter = 100000
lr = 0.00001

for it in range(num_iter):
    steps.append([cur_x, cur_y, f(cur_x, cur_y)])

    grad_x = df_dx(cur_x, cur_y)
    grad_y = df_dy(cur_x, cur_y)

    cur_x -= lr * grad_x
    cur_y -= lr * grad_y

print(steps[-1][-1])
