import numpy as np
import random


def morris(n_input: int, values: list, x_star: np.array = None, D_star: np.array = None) -> np.array:
    k: int = n_input
    p: int = len(values)
    delta: float = p / (2 * (p - 1))
    B: np.array = np.tri(k+1, k, -1)
    J: np.array = np.ones((k+1, k))
    P_star: np.array = np.identity(k)

    if x_star is None:
        x_star = np.random.choice(values, n_input)
    if D_star is None:
        D_star = np.diag(np.random.choice([1, -1], k))

    return (J * x_star + (delta / 2) * ((2 * B - J) @ D_star + J)) @ P_star


def get_D_star(k: int) -> np.array:
    return


def main():
    np.random.seed(42)
    n_input: int = 2
    values: list = [0, 1 / 3, 2 / 3, 1]

    # Test/Debug:
    x_star: np.array = np.array([0, 1 / 3])
    D_star: np.array = np.array([[1, 0], [0, -1]])

    print(morris(n_input, values, x_star, D_star))


if __name__ == "__main__":
    main()
