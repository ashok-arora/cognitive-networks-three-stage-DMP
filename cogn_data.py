import numpy as np
from tqdm import tqdm


def data_generation(length, file_name, probability):
    # Generate data
    a = []
    for p in tqdm(probability):
        a.append(np.random.binomial(n=1, p=p, size=[length]))
    aa = np.stack(a, axis=1)
    with open(file_name, "wb") as f:
        np.savetxt(
            f,
            aa.astype(int),
            delimiter=",",
            fmt="%i",
            newline="\n",
            header="",
            footer="",
            comments="# ",
        )


d = {
    "case1": [0.20, 0.30, 0.80, 0.70, 0.50, 0.10, 0.60, 0.40],
    "case2": [0.15, 0.45, 0.05, 0.65, 0.25, 0.85, 0.35, 0.75],
}

for case, probability in d.items():
    data_generation(10000, f"cogn-input-{case}.csv", probability)
