"""Part B tasks 1-3: RFM table, K-means selection, segment profiles, PCA, outputs/customer_segments.csv."""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

o = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = o[o["status"] == "completed"]

# Task 1: RFM table
g = done.groupby("customer_id")
rfm = pd.DataFrame({"recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum()}).reset_index()
print("customers:", len(rfm), "| recency median/min/max:", rfm.recency.median(), rfm.recency.min(), rfm.recency.max())
print("frequency mean/max: %.2f / %d | monetary mean/max: %.2f / %.2f" % (rfm.frequency.mean(), rfm.frequency.max(), rfm.monetary.mean(), rfm.monetary.max()))

# Task 2: K-means for K = 2..8
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\nK   inertia   silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k}   {km.inertia_:8.1f}   {silhouette_score(X, km.labels_):.3f}")

# Task 3: final model, profiles, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\nSegment profiles:\n", prof)
premium_id = prof["monetary"].idxmax()
rest = prof.drop(index=premium_id)
frequent_id = rest["frequency"].idxmax()
dormant_id = rest["recency"].idxmax()
names = {premium_id: "Premium Big-Spenders", frequent_id: "Frequent Everyday Shoppers", dormant_id: "Dormant Low-Value"}
names = {i: names.get(i, "Occasional Regulars") for i in prof.index}      # leftover segment
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = (rfm["segment_id"] == premium_id).astype(int)
pca = PCA(n_components=2).fit(X)
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_.round(3), "total", round(pca.explained_variance_ratio_.sum(), 3))
print("premium segment id:", premium_id)
print(names)
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("saved outputs/customer_segments.csv | premium customers:", int(rfm.is_premium.sum()))