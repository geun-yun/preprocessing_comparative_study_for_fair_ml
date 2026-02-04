import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing, LFR, DisparateImpactRemover
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ttest_rel, ttest_1samp, wilcoxon, mannwhitneyu
from preprocess_data import get_data, legit_spec, print_distribution
from emd import optimise_emd, emd_based_removal
from stat_test import perform_t_tests

def build_models(seed):
    models = {
        'SVM': SVC(random_state=seed),
        'LR': LogisticRegression(random_state=seed),
        'KNN': KNeighborsClassifier(),
        'RF': RandomForestClassifier(random_state=seed),
        'DT': DecisionTreeClassifier(random_state=seed),
        'NB': GaussianNB()

    }
    return models

def to_bld(df, label_col, prot_col):
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df,
        label_names=[label_col],
        protected_attribute_names=[prot_col],
    )

def bld_to_df(bld):
    # returns (df, meta), only need df
    return bld.convert_to_dataframe()[0]

def calculate_metrics(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    fnfp = fn / (fp + 1e-10)
    demo_parity = float(np.mean(np.asarray(y_pred) == 1))
    ber = (fpr + fnr) / 2 if np.isfinite(fpr) and np.isfinite(fnr) else 0
    ds = (fp + fn) / (fp + tn + fn + tp)

    return tp, tn, fp, fn, tpr, tnr, fpr, fnr, fnfp, demo_parity, ber, ds

def run_k_fold(kf, models, X_data, y_data, dataset, df_group, group_label, results_list, oversample=False, undersample=False, random_state=42):
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_data, y_data), start = 1):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        if oversample:
            # If a fold happens to have only one class, SMOTE will fail.
            if np.unique(y_train).size < 2:
                print(f"[SMOTE skipped] fold={fold} group={group_label}: only one class in y_train")
            else:
                sm = SMOTE(random_state=random_state)
                X_train, y_train = sm.fit_resample(X_train, y_train)
        elif undersample:
            if np.unique(y_train).size < 2:
                print(f"[Undersampling skipped] fold={fold} group={group_label}: only one class in y_train")
            else:
                rus = RandomUnderSampler(sampling_strategy="auto", random_state=random_state)
                X_train, y_train = rus.fit_resample(X_train, y_train)

        fold_results = {'Fold': fold, 'Group': group_label}
        print(f'Processing fold {fold} for group {group_label}')
        for name, model, in models.items():
            print(f'Training and evaluating model: {name}')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            tp, tn, fp, fn, tpr, tnr, fpr, fnr, fnfp, demo_parity, ber, ds = calculate_metrics(y_test, y_pred)

            L_test = df_group.iloc[test_idx]["_L"].to_numpy()
            L1 = (L_test == 1)
            csp = float(np.mean(np.asarray(y_pred)[L1] == 1)) if L1.sum() > 0 else np.nan

            fold_results.update({
                f'{name}_Accuracy': accuracy, f'{name}_Precision': precision,
                f'{name}_Recall': recall, f'{name}_f1': f1,
                f'{name}_TP': tp, f'{name}_TN': tn,
                f'{name}_FP': fp, f'{name}_FN': fn,
                f'{name}_TPR': tpr, f'{name}_TNR': tnr,
                f'{name}_FPR': fpr, f'{name}_FNR': fnr,
                f'{name}_FN/FP': fnfp, f'{name}_DP': demo_parity,
                f'{name}_BER': ber, f'{name}_CSP': csp, f'{name}_DS': ds
            })
        results_list.append(pd.DataFrame([fold_results]))

    return results_list

def run(dataset, method, sensitive_attribute, k, normalise=False, 
        seed=42, vt_threshold=0.1, repair_level=1.0, K=None, Ax=None, Ay=None, Az=None):
    df = get_data(dataset)

    legit_col, bin_fn = legit_spec(dataset)
    if bin_fn is None:
        L_raw = (df[legit_col] == 1).astype(int)
    else:
        L_raw = bin_fn(df[legit_col]).astype(int)
    df["_L"] = L_raw

    # apply any df-level mitigation first
    if method == "emd_male":
        df = emd_based_removal(dataset, df, sensitive_attribute, "male", seed)
    elif method == "emd_female":
        df = emd_based_removal(dataset, df, sensitive_attribute, "female", seed)
    elif method in ("raw", "diremover", "lfr", "oversample", "undersample"):
        pass
    else:
        raise ValueError(f"Unknown method: {method}")
    
    y = df["Target"].to_numpy().astype(int)
    X_all = df.drop(columns=["Target", "_L"]).copy()

    if sensitive_attribute not in X_all.columns:
        raise KeyError(f"Sensitive attribute '{sensitive_attribute}' not found in columns: {X_all.columns.tolist()}")

    s = X_all[sensitive_attribute].to_numpy()
    X_feat = X_all.drop(columns=[sensitive_attribute])

    selector = VarianceThreshold(threshold=vt_threshold)
    X_sel = selector.fit_transform(X_feat)
    selected_cols = X_feat.columns[selector.get_support()]

    if normalise:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
    else:
        X_scaled = X_sel

    # Rebuild a dataframe for potential DIR
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_cols, index=df.index)
    X_scaled_df[sensitive_attribute] = s  # add back untouched
    X_scaled_df["_L"] = df["_L"].to_numpy()

    print("final shape: ", df.shape)
    
    # aif360
    if method in ["diremover", "lfr"]:
        work_df = X_scaled_df.copy()
        work_df["Target"] = y

        bld = to_bld(work_df, "Target", sensitive_attribute)
        if method == "lfr":
            unprivileged_groups = [{sensitive_attribute: 0}]
            privileged_groups   = [{sensitive_attribute: 1}]
            lfr = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups,
                      k=K, Ax=Ax, Ay=Ay, Az=Az, seed=seed)
            bld_rep = lfr.fit_transform(bld)
            rep_df = bld_to_df(bld_rep)
            rep_df["_L"] = work_df["_L"].to_numpy()
            print(rep_df.head(100))
            print(rep_df.shape)
        elif method == "diremover":
            dir_alg = DisparateImpactRemover(repair_level=repair_level, sensitive_attribute=sensitive_attribute)
            rep_df = bld_to_df(dir_alg.fit_transform(bld))
            rep_df["_L"] = work_df["_L"].to_numpy()

        # update df/X/y after repair
        df = rep_df.copy()
        y = df["Target"].to_numpy().astype(int)

        # training features: repaired non-sensitive features (drop Target + sensitive attr)
        X_final = df.drop(columns=["Target", "_L", sensitive_attribute]).to_numpy()

    else:
        # training features: scaled non-sensitive features (drop sensitive attr)
        X_final = X_scaled_df.drop(columns=[sensitive_attribute, "_L"]).to_numpy()

    # split into sensitive groups
    priv_mask = (df[sensitive_attribute].to_numpy() == 1)
    unpriv_mask = (df[sensitive_attribute].to_numpy() == 0)

    df_priv = df.loc[priv_mask].reset_index(drop=True)
    df_unpriv = df.loc[unpriv_mask].reset_index(drop=True)

    X_priv = X_final[priv_mask]
    X_unpriv = X_final[unpriv_mask]
    y_priv = y[priv_mask]
    y_unpriv = y[unpriv_mask]

    if len(y_priv) == 0 or len(y_unpriv) == 0:
        raise ValueError(f"Empty group after preprocessing: priv={len(y_priv)}, unpriv={len(y_unpriv)}")

    # run CV within each group
    models = build_models(seed)
    results_list = []
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    if method == "oversample":
        run_k_fold(kf, models, X_priv, y_priv, dataset, df_priv, "Male", results_list, oversample=True)
        run_k_fold(kf, models, X_unpriv, y_unpriv, dataset, df_unpriv, "Female", results_list, oversample=True)
    elif method == "undersample":
        run_k_fold(kf, models, X_priv, y_priv, dataset, df_priv, "Male", results_list, undersample=True)
        run_k_fold(kf, models, X_unpriv, y_unpriv, dataset, df_unpriv, "Female", results_list, undersample=True)
    else:
        run_k_fold(kf, models, X_priv, y_priv, dataset, df_priv, "Male", results_list)
        run_k_fold(kf, models, X_unpriv, y_unpriv, dataset, df_unpriv, "Female", results_list)

    results_df = pd.concat(results_list, ignore_index=True)
    if normalise:
        results_df.to_csv(f"final_results/normalised_{dataset}_{method}_performance_results.csv", index=False)
    else:
        results_df.to_csv(f"final_results/{dataset}_{method}_performance_results.csv", index=False)

    tests_list = []
    for model in models.keys():
        perform_t_tests(results_df, model, "Female", tests_list)

    tests_df = pd.concat(tests_list, ignore_index=True)
    eo_df = tests_df[tests_df["metric"].isin(["TPR", "FPR"])]
    eo_pivot = eo_df.pivot(index="model", columns="metric", values="is_unfair")
    eo_pivot["EO_violation"] = eo_pivot.any(axis=1)
    eo_violation_count = eo_pivot["EO_violation"].sum()
    print(eo_violation_count)
    total_unfair = tests_df["is_unfair"].sum() + eo_violation_count
    
    cols = list(tests_df.columns)
    j = cols.index("is_unfair")
    row = {c: np.nan for c in cols}
    row[cols[j-3]] = "EO violation count"
    row[cols[j-2]] = eo_violation_count
    row[cols[j-1]] = "Total violation count"
    row["is_unfair"] = total_unfair
    summary_row = pd.DataFrame([row], columns=cols)

    tests_df = pd.concat([tests_df, summary_row], ignore_index=True)
    lfr_parameters = f"_K{K}_Ax{Ax}_Ay{Ay}_Az{Az}" if method=="lfr" else ""
    dir_parameter = f"_{repair_level}" if method=="diremover" else ""
    if normalise:
        tests_df.to_csv(f"real_final_results/normalised_{dataset}_{method}_stat_tests_results{lfr_parameters}{dir_parameter}.csv", index=False)
    else:
        tests_df.to_csv(f"real_final_results/{dataset}_{method}_stat_tests_results{lfr_parameters}{dir_parameter}.csv", index=False)

def run_all():
    methods = ["raw", "emd_male", "emd_female", "diremover", "oversample", "undersample"]
    datasets = ["heart", "liver", "default", "income", "coupon", "compas"]
    for dataset in datasets:
        for method in methods:
            k = 10 if dataset == "heart" or dataset == "liver" else 20
            if method == "lfr":
                match dataset:
                    case "heart":
                        K, Ax, Ay, Az = 10, 0.005, 0.8, 50.0
                    case "liver":
                        K, Ax, Ay, Az = 10, 0.005, 0.8, 50.0
                    case "default":
                        K, Ax, Ay, Az = 8, 0.02, 0.8, 50.0
                    case "income":
                        K, Ax, Ay, Az = 8, 0.015, 0.8, 50.0
                    case "coupon":
                        K, Ax, Ay, Az = 6, 0.01, 0.9, 50.0
                    case "compas":
                        K, Ax, Ay, Az = 5, 0.01, 0.8, 40.0
                run(dataset, method, "Gender", k, K=K, Ax=Ax, Ay=Ay, Az=Az)
            else:
                run(dataset, method, "Gender", k)
                run(dataset, method, "Gender", k, normalise=True)

run_all()
