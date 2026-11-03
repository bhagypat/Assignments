import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, r2_score, mean_squared_error, mean_absolute_error)

warnings.filterwarnings("ignore")
DATA = "../data"
OUT = "../output"
os.makedirs(OUT, exist_ok=True)
SEED = 42

# name -> (csv file, target column, task, columns to drop, note)
# task: "cls" = classification, "reg" = regression
problems = [
    ("Supply Chain",   "supply_chain_dataset.csv",       "stockout_risk",          "cls", ["product_id"]),
    ("Supply Chain (projected stockout)", "supply_chain_dataset.csv", "projected_stockout", "cls", ["product_id", "stockout_risk"]),
    ("BFSI",           "bfsi_fraud_dataset.csv",         "is_fraud",               "cls", []),
    ("AdTech",         "adtech_ecommerce_dataset.csv",   "conversion",             "cls", []),
    ("Telecom",        "telecom_churn_dataset.csv",      "churn",                  "cls", []),
    ("Cybersecurity",  "cybersecurity_dataset.csv",      "intrusion_flag",         "cls", []),
    ("IT Infra",       "it_infrastructure_dataset.csv",  "system_failure_risk",    "cls", []),
    ("Healthcare",     "healthcare_dataset.csv",         "disease_risk",           "cls", []),
    ("HR",             "hr_analytics_dataset.csv",       "attrition_risk",         "cls", []),
    ("Construction (SVR)", "construction_dataset.csv",   "compressive_strength",   "reg", []),
    ("Energy (SVR)",   "energy_utilities_dataset.csv",   "energy_consumption_kwh", "reg", []),
    ("EdTech",         "edtech_dataset.csv",             "performance_level",      "cls", []),
    ("Automotive",     "automotive_adas_dataset.csv",    "driving_risk",           "cls", []),
]

def load(file, target, drop):
    df = pd.read_csv(f"{DATA}/{file}")
    if target == "projected_stockout":
        # my own label (NOT in the file): stock on hand will run out before a new delivery can arrive
        df[target] = (df["inventory_level"] < df["daily_sales"] * df["lead_time_days"]).astype(int)
    y = df[target]
    X = df.drop(columns=[target] + [c for c in drop if c in df.columns])
    return X, y

def preprocessor(X):
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)]), num, cat

def eda(X, y, num, cat, task):
    info = {"rows": len(X), "numeric": len(num), "categorical": len(cat), "missing": int(X.isna().sum().sum())}
    out = {}
    for c in num:                                  # share of values outside the 1.5 x IQR fences
        q1, q3 = X[c].quantile([.25, .75]); iqr = q3 - q1
        out[c] = float(((X[c] < q1 - 1.5 * iqr) | (X[c] > q3 + 1.5 * iqr)).mean())
    info["outlier_cols"] = {c: round(v, 3) for c, v in out.items() if v > 0.01}
    corr = X[num].apply(lambda s: np.corrcoef(s, y)[0, 1]).abs().sort_values(ascending=False)
    info["top_corr"] = {c: round(float(v), 2) for c, v in corr.head(3).items()}
    if task == "cls":
        info["positive_rate"] = round(float(y.mean()), 4); info["positives"] = int(y.sum())
    else:
        info.update(y_mean=round(float(y.mean()), 1), y_std=round(float(y.std()), 1))
    return info

def cls_grids():
    cw = [None, "balanced"]
    return {
        "linear": {"svc__C": [0.01, 0.1, 1], "svc__class_weight": cw},
        "rbf":    {"svc__C": [1, 10, 100], "svc__gamma": [0.01, 0.1, 1], "svc__class_weight": cw},
        "poly":   {"svc__C": [1, 10], "svc__degree": [2, 3], "svc__class_weight": cw},
    }

def reg_grids():
    return {
        "linear": {"svr__regressor__C": [0.01, 0.1, 1]},
        "rbf":    {"svr__regressor__C": [1, 10, 100], "svr__regressor__gamma": [0.01, 0.1, 1]},
        "poly":   {"svr__regressor__C": [1, 10], "svr__regressor__degree": [2, 3]},
    }

rows, eda_all = [], {}
for name, file, target, task, drop in problems:
    X, y = load(file, target, drop)
    pre, num, cat = preprocessor(X)
    eda_all[name] = eda(X, y, num, cat, task)
    if task == "cls":
        if y.sum() < 20:                          # the file's own supply-chain label has only 2 positives
            print(f"{name}: only {int(y.sum())} positive rows - cannot train or test a classifier, skipped")
            eda_all[name]["skipped"] = True
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        cv = StratifiedKFold(3, shuffle=True, random_state=SEED); scoring = "f1"
        grids = cls_grids()
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
        cv = KFold(3, shuffle=True, random_state=SEED); scoring = "r2"
        grids = reg_grids()

    for kernel in ["linear", "rbf", "poly"]:
        def make(params=None):
            kw = {"kernel": kernel}
            if kernel == "poly": kw.update(degree=3, coef0=1)
            if task == "cls":
                return make_pipeline(pre, SVC(**kw))
            return make_pipeline(pre, TransformedTargetRegressor(regressor=SVR(**kw), transformer=StandardScaler()))
        # the step name differs between SVC pipeline ("svc") and SVR pipeline ("transformedtargetregressor")
        steps = {"cls": "svc", "reg": "svr"}
        for stage in ["default", "tuned"]:
            model = make()
            best, cv_score = "C=1, gamma=scale", np.nan
            if stage == "tuned":
                grid = grids[kernel]
                if task == "cls":
                    pipe = make_pipeline(pre, SVC(kernel=kernel, **({"coef0": 1} if kernel == "poly" else {})))
                    pg = grid
                else:
                    pipe = make_pipeline(pre, TransformedTargetRegressor(
                        regressor=SVR(kernel=kernel, **({"coef0": 1} if kernel == "poly" else {})), transformer=StandardScaler()))
                    pg = {k.replace("svr__", "transformedtargetregressor__"): v for k, v in grid.items()}
                search = GridSearchCV(pipe, pg, cv=cv, scoring=scoring, n_jobs=1).fit(X_tr, y_tr)
                model, cv_score = search.best_estimator_, float(search.best_score_)
                best = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in search.best_params_.items())
            t = time.time(); model.fit(X_tr, y_tr); fit_t = time.time() - t
            t = time.time(); pred = model.predict(X_te); pred_t = time.time() - t
            est = model[-1] if task == "cls" else model[-1].regressor_
            n_sv = int(est.n_support_.sum()) if task == "cls" else int(len(est.support_))
            row = {"dataset": name, "task": task, "kernel": kernel, "stage": stage, "params": best,
                   "cv_score": round(cv_score, 4) if cv_score == cv_score else None,
                   "fit_s": round(fit_t, 2), "predict_s": round(pred_t, 3), "n_support": n_sv}
            if task == "cls":
                score = model.decision_function(X_te); tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
                row.update(accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred, zero_division=0),
                           recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), roc_auc=roc_auc_score(y_te, score),
                           TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), baseline_acc=max(y_te.mean(), 1 - y_te.mean()))
            else:
                row.update(r2=r2_score(y_te, pred), rmse=float(np.sqrt(mean_squared_error(y_te, pred))),
                           mae=mean_absolute_error(y_te, pred))
            rows.append(row)
            print(f"{name:34s} {kernel:6s} {stage:7s} " + (f"acc={row['accuracy']:.3f} f1={row['f1']:.3f} auc={row['roc_auc']:.3f}" if task == "cls" else f"r2={row['r2']:.3f} rmse={row['rmse']:.2f}") + f"  [{best}]")

res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUT}/svm_handson_results.csv", index=False)
json.dump(eda_all, open(f"{OUT}/svm_handson_eda.json", "w"), indent=1)

# chart: tuned test F1 (classification) by kernel
t = res[(res.stage == "tuned") & (res.task == "cls")].pivot(index="dataset", columns="kernel", values="f1")
t.plot.barh(figsize=(8, 5.5), color=["#2E5B8A", "#C0392B", "#7F8C8D"])
plt.xlabel("Test F1 score (tuned models)"); plt.title("SVM kernels compared on each classification dataset")
plt.legend(title="kernel", loc="center left", bbox_to_anchor=(1.01, 0.5)); plt.tight_layout(); plt.savefig(f"{OUT}/svm_handson_f1.png", dpi=110)
print("saved results")
