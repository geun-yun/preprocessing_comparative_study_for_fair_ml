import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from preprocess_data import get_data
import math

def optimise_emd(y, target_y, A, target_A, priv_val, alpha=0.05):
    mask = (y == target_y) & (A == target_A)
    removeable_idx = y.index[mask]
    k = len(removeable_idx)
    step = max(1, int(math.sqrt(k)))
    print("k: ", k, " Step: ", step)
    def emd_after(i):
        drop_idx = removeable_idx[:i]
        return compute_emd_p_value(y.drop(drop_idx), A.drop(drop_idx), priv_val, opt=True, alpha=alpha)
 
    k_high = k
    for i in range(0, k + 1, step):
        found = emd_after(i)
        print("i: ", i)
        if found:
            k_high = i
            break
    print("k_high: ", k_high)
    k_low = max(0, k_high - step)
    for i in range(k_low, k_high + 1):
        found = emd_after(i)
        if found:
            drop_idx = removeable_idx[:i]
            D, p = compute_emd_p_value(y.drop(drop_idx), A.drop(drop_idx), priv_val)
            print(D, p, i, step)
            return D, p, i, step

    print("Could not find")
    return None, None, None, step

def compute_emd(y, A, priv_val):
    P_y_cond_A = pd.crosstab(y, A, normalize='columns').sort_index()
    P_y_cond_A = P_y_cond_A[priv_val]
    P_y = y.value_counts(normalize=True).sort_index()
    # print(f"P(y|{A.name}={priv_val})\n", P_y_cond_A)
    # print("P(y)\n", P_y)
    # print(pd.crosstab(P_y_cond_A, P_y))
    dist = wasserstein_distance([0,1], [0,1], P_y_cond_A, P_y)
    return dist

def compute_emd_p_value(y, A, priv_val, n_perm=1000, random_seed=42, opt=False, alpha=0.05):
    D_obs = compute_emd(y, A, priv_val)
    rng = np.random.default_rng(random_seed)
    A_vals = A.to_numpy()
    extreme = 0
    for i in range(n_perm):
        A_perm = pd.Series(rng.permutation(A_vals), index=A.index, name=A.name)
        # print(A_perm)
        D_perm = compute_emd(y, A_perm, priv_val)
        if D_perm >= D_obs:
            extreme += 1

        if opt:
            p_min = extreme / n_perm
            p_max = (extreme + n_perm - (i+1)) / n_perm
            if p_min > alpha:
                return True
            elif p_max < alpha:
                return False

    p_val = extreme / n_perm
    return D_obs, p_val

#y_val, priv_val   1,1  0,0  1,0  0,1
# heart!            82  486  
# defafult!        370 1830
# income!         4454 6033
# coupon!          363  293
# compas!         1087  802
# cad              265   76
# liver!                      237  197
# diabetes                   1169 1147             


# heart (3656, 13): https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression
# liver (1700, 11): https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset
# default (30000, 24): https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
# income (30162, 13): https://archive.ics.uci.edu/dataset/2/adult https://www.kaggle.com/datasets/uciml/adult-census-income
# coupon (12079, 24): https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation
# compas (15615, 22): https://www.kaggle.com/datasets/danofer/compass

def emd_based_removal(dataset, df, sensitive_attribute, group_type, random_state):
    y = df['Target']
    if group_type == 'male':
        target_type = 1 if dataset != "liver" else 0
        match dataset:
            case "heart":
                best_n = 82
            case "liver":
                best_n = 197
            case "default":
                best_n = 370
            case "income":
                best_n = 4454
            case "coupon":
                best_n = 363
            case "compas":
                best_n = 1087

        removeable_idx = df.index[(y == target_type) & (df[sensitive_attribute] == 1)]
    elif group_type == 'female':
        target_type = 0 if dataset != "liver" else 1
        match dataset:
            case "heart":
                best_n = 486
            case "liver":
                best_n = 237
            case "default":
                best_n = 1830
            case "income":
                best_n = 6033
            case "coupon":
                best_n = 293
            case "compas":
                best_n = 802

        removeable_idx = df.index[(y == target_type) & (df[sensitive_attribute] == 0)]
    
    rng = np.random.default_rng(random_state)
    drop_idx = rng.choice(removeable_idx, size=best_n, replace=False)
    df = df.drop(index=drop_idx)
    return df

def dummy_data():
    df = pd.DataFrame({
    "Gender":    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
    "Target": [0,0,0,0,0,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1]
    })
    return df

df = get_data('compas')
y = df['Target']
A = df['Gender']
# print(compute_emd(y, A, 0))
# print(compute_emd_p_value(y, A, 0))
x = 1
print(optimise_emd(y, x, A, x, x, alpha=0.05))
# # for i in range(0, 13):
# #     mask = (y == 1) & (A == 1)
# #     removeable_idx = y.index[mask]
# #     drop_idx = removeable_idx[:i]
# #     D, p = compute_emd_p_value(y.drop(drop_idx), A.drop(drop_idx), 0)
# #     print("EMD: ", D, " p-val: ", p)


# compute_emd(y, A, 1)
# for a in ['Gender', 'Race']:
#     for priv_val in [0, 1]:
#         D_obs, p_val = compute_emd_p_value(y, A, priv_val)
#         print(f"{a}={priv_val}:")
#         print("EMD: ", D_obs)
#         print("p-value: ", p_val)

