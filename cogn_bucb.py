from scipy.stats import beta
import numpy as np


def BUCB(total_time, k, C, X, T):
    def Q(i, t):
        a = X[i] + 1
        b = T[i] - X[i] + 1
        # print(a, b)
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 2000)
        return np.quantile(x, 1 - (1 / t))

    Qs = [(0, 0)] * C
    for c in range(C):
        Qs[c] = (c, Q(c, total_time))
    sorted_Qs = sorted(Qs, key=lambda t: t[1], reverse=True)
    return sorted_Qs[k][0]
