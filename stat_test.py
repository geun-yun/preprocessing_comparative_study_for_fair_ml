import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal
from pingouin import welch_anova

EPS = 1e-10

def is_normality(x):
    x = np.asarray(x)
    if len(np.unique(x)) == 1:
        return -1
    _, p = shapiro(x)
    return p

def split_by_label(df, label, group_col="Group"):
    protected = df[group_col] == label
    unprotected = ~protected
    return protected, unprotected

def get_metric_arrays(df, priv_mask, unpriv_mask, col):
    a = df.loc[priv_mask, col].to_numpy()
    b = df.loc[unpriv_mask, col].to_numpy()

    return a, b

def ratio_arrays(df, prot_mask, unprot_mask, num_col, den_col, eps=EPS):
    a = (df.loc[prot_mask, num_col] / (df.loc[prot_mask, den_col] + eps)).to_numpy()
    b = (df.loc[unprot_mask, num_col] / (df.loc[unprot_mask, den_col] + eps)).to_numpy()
    return a, b

def compare_two_groups(a, b, alpha=0.05):
    p_a = is_normality(a)
    p_b = is_normality(b)

    if p_a > alpha and p_b > alpha:
        _, p_lv = levene(a, b)
        equal_var = p_lv > alpha
        stat, p = ttest_ind(a, b, equal_var=equal_var)
        return {"test": "ttest_ind", "equal_var": equal_var, "stat": stat, "is_unfair": (p <= 0.05), 
                "p": p, "p_levene": p_lv, "p_norm_a": p_a, "p_norm_b": p_b}
    else:
        stat, p = mannwhitneyu(a, b)
        return {"test": "mannwhitneyu", "stat": stat, "is_unfair": (p <= 0.05), 
                "p": p, "p_norm_a": p_a, "p_norm_b": p_b}

def perform_t_tests(df, model, label, tests_list, alpha=0.05, group_col="Group"):
    priv_mask, unpriv_mask = split_by_label(df, label, group_col)

    tpr_col = f"{model}_TPR"
    fpr_col = f"{model}_FPR"
    fnfp_col  = f"{model}_FN/FP"
    dp_col = f"{model}_DP"
    ber_col = f"{model}_BER"
    csp_col = f"{model}_CSP"
    ds_col = f"{model}_DS"

    for col in [tpr_col, fpr_col, fnfp_col, dp_col, ber_col, csp_col, ds_col]:
        col_a, col_b = get_metric_arrays(df, priv_mask, unpriv_mask, col)
        
        res = compare_two_groups(col_a, col_b, alpha=alpha)

        tests_list.append(pd.DataFrame([{
            "model": model,
            "group": label,
            "metric": col.rsplit("_", 1)[-1],  # TPR, FPR, FN/FP, DP, BER, CSP, DS
            **res
        }]))

        
