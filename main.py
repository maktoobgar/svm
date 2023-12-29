import math
import sys
import numpy as np
import scipy.optimize as opt


# All data
points = [
    [np.array([1, 0], dtype=np.float64), -1.0],
    [np.array([0, 1], dtype=np.float64), -1.0],
    [np.array([0, -1], dtype=np.float64), -1.0],
    [np.array([-1, 0], dtype=np.float64), -1.0],
    [np.array([3, -1], dtype=np.float64), 1.0],
    [np.array([3, 1], dtype=np.float64), 1.0],
    [np.array([6, 1], dtype=np.float64), 1.0],
    [np.array([6, -1], dtype=np.float64), 1.0],
]


# Define the function to minimize
def dual(u):
    sum = 0.0
    for i in range(len(points)):
        sum += u[i]
    sum_in_sum = 0.0
    for i in range(len(points)):
        for j in range(len(points)):
            sum_in_sum += (
                points[i][1]
                * points[j][1]
                * u[i]
                * u[j]
                * np.dot(points[i][0], points[j][0])
            )
    return -(sum - 0.5 * sum_in_sum)


def con(u):
    sum = 0.0
    for i in range(len(points)):
        sum += points[i][1] * u[i]
    return sum


cons = [{"type": "eq", "fun": con}]

# Find the minimum value of the function
result = opt.minimize(
    dual,
    x0=[100 for _ in range(len(points))],
    bounds=[(0, 500) for _ in range(len(points))],
    # method="BFGS",
    constraints=cons,
)

mu = result.x
print(f"mu = {mu}")
w = np.array([0 for _ in range(len(points[0][0]))], dtype=np.float64)

for i in range(len(mu)):
    w += points[i][1] * mu[i] * points[i][0]

print(f"w = {w}")

chosen_mu = 0.0
chosen_mu_index = -1

for i in range(len(points)):
    if mu[i] > chosen_mu:
        chosen_mu = mu[i]
        chosen_mu_index = i


b = -(np.dot(w, points[chosen_mu_index][0]) - points[chosen_mu_index][1])

print(f"{round(w[0], 5)}x + {round(w[1], 5)}y + {round(b, 5)}")
