import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (
        x.mean(),
        np.percentile(boots, 100 * alpha / 2),
        np.percentile(boots, 100 * (1 - alpha / 2)),
    )

# In the order of SVM, LR, KNN, RF, DT, NB             
violations_D1_P0 = [6, 7, 5, 6, 5, 3]
violations_D1_P1 = [2, 1, 2, 2, 5, 2]
violations_D1_P2 = [2, 2, 4, 2, 5, 3]
violations_D1_P3 = [6, 2, 7, 5, 5, 2]
violations_D1_P4 = [5, 2, 6, 6, 2, 2]
violations_D1_P5 = [6, 7, 6, 7, 5, 3]

violations_D2_P0 = [6, 5, 6, 6, 2, 5]
violations_D2_P1 = [6, 2, 0, 1, 5, 2]
violations_D2_P2 = [2, 1, 2, 2, 1, 0]
violations_D2_P3 = [2, 3, 3, 2, 2, 3]
violations_D2_P4 = [3, 3, 5, 3, 3, 3]
violations_D2_P5 = [6, 6, 5, 6, 2, 5]

violations_D3_P0 = [7, 8, 5, 7, 5, 8]
violations_D3_P1 = [5, 4, 4, 4, 5, 8]
violations_D3_P2 = [2, 4, 2, 4, 4, 8]
violations_D3_P3 = [6, 6, 4, 5, 4, 8]
violations_D3_P4 = [5, 6, 5, 6, 4, 8]
violations_D3_P5 = [7, 8, 5, 7, 5, 1] # <- worth noting

violations_D4_P0 = [8, 8, 8, 7, 8, 6]
violations_D4_P1 = [1, 3, 6, 7, 6, 1]
violations_D4_P2 = [1, 4, 5, 6, 7, 2]
violations_D4_P3 = [4, 7, 7, 8, 8, 8]
violations_D4_P4 = [4, 8, 7, 8, 8, 8]
violations_D4_P5 = [8, 8, 8, 7, 8, 6]

violations_D5_P0 = [1, 7, 6, 8, 3, 7]
violations_D5_P1 = [1, 7, 4, 5, 2, 5]
violations_D5_P2 = [1, 7, 4, 6, 0, 7]
violations_D5_P3 = [2, 2, 6, 6, 5, 6]
violations_D5_P4 = [2, 1, 1, 6, 1, 4]
violations_D5_P5 = [1, 7, 6, 8, 3, 7]

violations_D6_P0 = [2, 5, 8, 2, 2, 8]
violations_D6_P1 = [2, 1, 0, 2, 2, 6]
violations_D6_P2 = [2, 4, 5, 1, 5, 7]
violations_D6_P3 = [8, 1, 2, 2, 2, 5]
violations_D6_P4 = [6, 2, 2, 2, 2, 7]
violations_D6_P5 = [2, 5, 8, 2, 2, 8]

viol = {
    "D1": {"P0": violations_D1_P0, "P1": violations_D1_P1, "P2": violations_D1_P2, "P3": violations_D1_P3, "P4": violations_D1_P4, "P5": violations_D1_P5},
    "D2": {"P0": violations_D2_P0, "P1": violations_D2_P1, "P2": violations_D2_P2, "P3": violations_D2_P3, "P4": violations_D2_P4, "P5": violations_D2_P5},
    "D3": {"P0": violations_D3_P0, "P1": violations_D3_P1, "P2": violations_D3_P2, "P3": violations_D3_P3, "P4": violations_D3_P4, "P5": violations_D3_P5},
    "D4": {"P0": violations_D4_P0, "P1": violations_D4_P1, "P2": violations_D4_P2, "P3": violations_D4_P3, "P4": violations_D4_P4, "P5": violations_D4_P5},
    "D5": {"P0": violations_D5_P0, "P1": violations_D5_P1, "P2": violations_D5_P2, "P3": violations_D5_P3, "P4": violations_D5_P4, "P5": violations_D5_P5},
    "D6": {"P0": violations_D6_P0, "P1": violations_D6_P1, "P2": violations_D6_P2, "P3": violations_D6_P3, "P4": violations_D6_P4, "P5": violations_D6_P5},
}

def bootstrap_pooled_reduction_ci(v0, v1, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    v0 = np.asarray(v0)
    v1 = np.asarray(v1)

    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(v0), size=len(v0), replace=True)
        r = 1 - v1[idx].sum() / v0[idx].sum()
        boots.append(r * 100)

    boots = np.asarray(boots)
    mean = 1 - v1.sum() / v0.sum()
    mean *= 100

    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return mean, lo, hi

def per_classifier_reductions(v0, v1):
    """
    v0, v1: lists length 6 (violations out of 8 per classifier)
    returns: np.array of per-classifier reduction rates in percent
            r_c = (1 - v1/v0) * 100
    """
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)

    # If any baseline v0==0, reduction is undefined; drop those classifiers for this cell.
    mask = v0 > 0
    r = (1 - (v1[mask] / v0[mask])) * 100.0
    return r

datasets = ["D1","D2","D3","D4","D5","D6"]
methods = ["P1","P2","P3","P4","P5"]

mean_mat = pd.DataFrame(index=datasets, columns=methods, dtype=float)
annot_mat = pd.DataFrame(index=datasets, columns=methods, dtype=object)

for d in datasets:
    for p in methods:
        r = per_classifier_reductions(viol[d]["P0"], viol[d][p])

        # If all v0 were zero (unlikely), r would be empty
        if len(r) == 0:
            mean_mat.loc[d, p] = np.nan
            annot_mat.loc[d, p] = "NA"
            continue

        # mean, lo, hi = bootstrap_ci(r, n_boot=10000, alpha=0.05, seed=0)
        mean, lo, hi = bootstrap_pooled_reduction_ci(viol[d]["P0"], viol[d][p])


        mean_mat.loc[d, p] = mean
        annot_mat.loc[d, p] = f"{mean:.0f}%\n[{lo:.0f}, {hi:.0f}]"

plt.figure(figsize=(9, 6))
ax = sns.heatmap(
    mean_mat,
    annot=annot_mat,
    fmt="",
    cmap="Greens",
    linewidths=0.5,
    cbar_kws={"label": "Mean reduction rate (%)"},
    vmin=np.nanmin(mean_mat.values),
    vmax=np.nanmax(mean_mat.values),
)

# ax.set_title("Reduction rates by dataset and preprocessing method\n(mean with 95% bootstrap CI across classifiers)")
ax.set_title("Reduction rates by dataset and preprocessing method\n(pooled reduction with 95% bootstrap CI over classifiers)")
ax.set_xlabel("Preprocessing method")
ax.set_ylabel("Dataset")
plt.tight_layout()
plt.show()


# Reduction rates (%), as given
reduction = pd.DataFrame(
    {
        "P1": [56, 47, 25, 47, 25, 52],
        "P2": [44, 73, 40, 44, 22, 11],
        "P3": [16, 50, 18, 7, 16, 26],
        "P4": [28, 33, 15, 4, 53, 22],
        "P5": [-6, 0, 18, 0, 0, 0],
    },
    index=["D1", "D2", "D3", "D4", "D5", "D6"]
)

# Annotation strings like "56%"
annot = reduction.applymap(lambda v: f"{v:.0f}%")

plt.figure(figsize=(8, 5))
ax = sns.heatmap(
    reduction,
    annot=annot,
    fmt="",
    cmap="Greens",
    linewidths=0.5,
    cbar_kws={"label": "Reduction rate (%)"},
    vmin=reduction.min().min(),  # includes negative values
    vmax=reduction.max().max(),
)

ax.set_title("Reduction rates by dataset and preprocessing method")
ax.set_xlabel("Preprocessing method")
ax.set_ylabel("Dataset")
plt.tight_layout()
plt.show()
