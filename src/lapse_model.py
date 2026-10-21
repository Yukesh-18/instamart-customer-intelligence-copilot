"""Part B tasks 4-8: lapse problem, leakage-free preprocessing, tuned models, class balance, interpretation, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/clean/products.csv")
seg = pd.read_csv("outputs/customer_segments.csv")          # used ONLY for the retention list, never as a feature

# ---------- Task 4: cohort, label, features (only orders BEFORE the cutoff) ----------
before = o[o["order_ts"] < CUTOFF].merge(p[["product_id", "category"]], on="product_id")
b_done = before[before["status"] == "completed"]
g = b_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().index[0]),
})
feat["cancel_rate"] = before.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean()).reindex(feat.index)
feat = feat.reset_index().merge(c[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat = feat.drop(columns="signup_date")

sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < END) & (o["status"] == "completed")]
feat["lapsed"] = (~feat["customer_id"].isin(sept["customer_id"])).astype(int)
print("cohort:", len(feat), "| lapsed:", int(feat.lapsed.sum()), f"({feat.lapsed.mean():.2%})")

NUM = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery", "cancel_rate", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = feat[NUM + CAT], feat["lapsed"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test sizes:", len(Xtr), len(Xte))
base_acc = (yte == 0).mean()
print("always-'not lapsed' baseline: accuracy %.3f, lapsed-class F1 %.3f" % (base_acc, f1_score(yte, np.zeros(len(yte)), zero_division=0)))

# ---------- Task 5: preprocessing inside a Pipeline (fit on TRAIN only) ----------
def make_pipe(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(drop="first"), CAT)])
    return Pipeline([("pre", pre), ("model", model)])

print("missing values in features:", int(X.isna().sum().sum()))
demo = make_pipe(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
i = NUM.index("frequency")
print("frequency mean learned from TRAIN: %.4f | full cohort mean: %.4f" % (demo.named_steps["pre"].named_transformers_["num"].mean_[i], X["frequency"].mean()))

# ---------- Task 6: baseline, grid searches, evaluation ----------
def report(name, model, Xt=Xte, yt=yte):
    pr = model.predict(Xt)
    tn, fp, fn, tp = confusion_matrix(yt, pr).ravel()
    print(f"{name}: TN={tn} FP={fp} FN={fn} TP={tp} | precision {precision_score(yt, pr, zero_division=0):.3f} "
          f"recall {recall_score(yt, pr, zero_division=0):.3f} F1 {f1_score(yt, pr, zero_division=0):.3f}")
    return f1_score(yt, pr, zero_division=0)

base = make_pipe(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
report("baseline LR", base)
lr_gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)), {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("LR best params:", lr_gs.best_params_, "| CV F1 %.3f" % lr_gs.best_score_)
f1_lr = report("tuned LR", lr_gs.best_estimator_)
rf_gs = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                     {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("RF best params:", rf_gs.best_params_, "| CV F1 %.3f" % rf_gs.best_score_)
f1_rf = report("tuned RF", rf_gs.best_estimator_)
selected = rf_gs.best_estimator_ if f1_rf >= f1_lr else lr_gs.best_estimator_
print("selected model:", "random forest" if selected is rf_gs.best_estimator_ else "logistic regression", "| test F1 %.3f" % max(f1_rf, f1_lr))

# ---------- Task 7: oversample the TRAINING set only ----------
rng_state = 42
maj, mino = Xtr[ytr == 0], Xtr[ytr == 1]
extra = mino.sample(n=len(maj) - len(mino), replace=True, random_state=rng_state)
Xbal = pd.concat([Xtr, extra]); ybal = pd.concat([ytr, pd.Series(1, index=extra.index)])
print("balanced train class counts:", ybal.value_counts().to_dict())
best_lr = make_pipe(LogisticRegression(max_iter=1000, C=lr_gs.best_params_["model__C"]))
report("best LR, original train", best_lr.fit(Xtr, ytr))
report("best LR, oversampled train", best_lr.fit(Xbal, ybal))

# ---------- Task 8: interpret, persist, retention list ----------
rf = rf_gs.best_estimator_
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 random-forest feature importances:\n", imp.head(5).round(3))
lrm = lr_gs.best_estimator_
coef = pd.Series(lrm.named_steps["model"].coef_[0], index=lrm.named_steps["pre"].get_feature_names_out())
print("\nTop 3 logistic coefficients by size:\n", coef.reindex(coef.abs().sort_values(ascending=False).index).head(3).round(3))

joblib.dump(selected, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
print("\nreloaded model reproduces predictions:", bool(np.array_equal(loaded.predict_proba(Xte), selected.predict_proba(Xte))))

test = feat.loc[Xte.index, ["customer_id"]].copy()
test["lapse_probability"] = loaded.predict_proba(Xte)[:, 1].round(4)
test["actually_lapsed"] = yte.values
test = test.merge(seg[["customer_id", "is_premium"]], on="customer_id", how="left")
prem = test[test["is_premium"] == 1].sort_values("lapse_probability", ascending=False)
top = prem.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
top.to_csv("outputs/retention_list.csv", index=False)
print("premium customers in test set:", len(prem), "| retention list rows:", len(top), "| of these actually lapsed:", int(top.actually_lapsed.sum()))