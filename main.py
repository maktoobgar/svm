import numpy as np
import scipy.optimize as opt


# Define the dual svm function
def svm_dual(u, points):
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


# Svm condition function
def svm_con(u, points):
    sum = 0.0
    for i in range(len(points)):
        sum += points[i][1] * u[i]
    return sum


# Svm solver
def svm_solver(data):
    # Prepare the functions for optimization
    svm_con_lambda = lambda x: svm_con(x, data)
    svm_cons = [{"type": "eq", "fun": svm_con_lambda}]
    svm_dual_lambda = lambda x: svm_dual(x, data)

    # Optimize
    result = opt.minimize(
        svm_dual_lambda,
        x0=[100 for _ in range(len(data))],
        bounds=[(0, 500) for _ in range(len(data))],
        constraints=svm_cons,
    )

    mu = result.x
    w = np.array([0 for _ in range(len(data[0][0]))], dtype=np.float64)

    for i in range(len(mu)):
        w += data[i][1] * mu[i] * data[i][0]

    chosen_mu = 0.0
    chosen_mu_index = -1

    for i in range(len(data)):
        if mu[i] > chosen_mu:
            chosen_mu = mu[i]
            chosen_mu_index = i

    b = -(np.dot(w, data[chosen_mu_index][0]) - data[chosen_mu_index][1])

    return lambda x: 1 if (np.dot(w, x) + b) > 0 else -1


# Retrieves the attribute definitions
def get_attributes_definitions(addr: str) -> dict:
    attributes = {}
    with open(addr, "r") as file:
        lines = file.readlines()
        for line_index in range(len(lines)):
            line = lines[line_index]
            i = -1 if line_index == len(lines) - 1 else 0
            data = line.strip().split(" ")
            name = data[0]
            elements = data[1].split(",")
            attributes[name] = {}
            for element in elements:
                attributes[name][element] = i
                if i == -1:
                    i = 1
                    continue
                i += 1
    return attributes


# Retrieves and defines data from attributes_definitions
def get_data(attributes_definitions: dict, addr: str) -> []:
    points = []
    with open(addr, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = []
            output = -100
            split = line.strip().split(",")
            i = 0
            for key in attributes_definitions:
                if split[i] == "?":
                    data = []
                    break
                if i == len(attributes_definitions) - 1:
                    output = attributes_definitions[key][split[i]]
                else:
                    data.append(attributes_definitions[key][split[i]])
                i += 1
            if len(data) > 0:
                points.append([np.array(data, dtype=np.float64), output])
    return points


def calculate_test(data, svm) -> float:
    right = 0
    for single in data:
        if single[1] == svm(single[0]):
            right += 1
    return right / len(data)


def main():
    # Prepare all
    attributes_definitions = get_attributes_definitions("files/attributes.txt")
    data = get_data(attributes_definitions, "files/train_small.txt")
    svm = svm_solver(data)
    test_data = get_data(attributes_definitions, "files/test.txt")
    print(
        f"Trained on {len(data)} Data.\nTested on {len(test_data)} Data.\nAccuracy: {round(calculate_test(test_data, svm)*100, 2)}%"
    )


if __name__ == "__main__":
    main()
