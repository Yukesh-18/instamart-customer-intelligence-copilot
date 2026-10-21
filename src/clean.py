"""Part A tasks 3-5: clean customers and orders, write CSVs, store in SQLite. Every rule prints how many values it changed."""
import sqlite3
import os
import numpy as np
import pandas as pd

os.makedirs("data/clean", exist_ok=True)

# ---------------- customers ----------------
c = pd.read_csv("data/raw/customers.csv", dtype=str, keep_default_na=False)
print("== CUSTOMERS ==")
before = c["city"].copy()
c["city"] = c["city"].str.strip().str.title()
print("city: changed", int((before != c["city"]).sum()), "values ->", c["city"].nunique(), "distinct cities")

gmap = {"m": "male", "male": "male", "f": "female", "female": "female"}
g = c["gender"].str.strip().str.lower().map(gmap)
print("gender: blank -> unknown:", int(g.isna().sum()))
c["gender"] = g.fillna("unknown")

c["age"] = pd.to_numeric(c["age"], errors="coerce")
bad = (c["age"] < 10) | (c["age"] > 100)
print("age: impossible values set to missing:", int(bad.sum()))
c.loc[bad, "age"] = np.nan
n_missing = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
print("age: filled", n_missing, "missing values with the city median; still missing:", int(c["age"].isna().sum()))
c["signup_date"] = pd.to_datetime(c["signup_date"])

# ---------------- products ----------------
p = pd.read_csv("data/raw/products.csv")

# ---------------- orders ----------------
o = pd.read_csv("data/raw/orders.csv", dtype=str, keep_default_na=False)
print("\n== ORDERS ==")
print("raw rows:", len(o))
# i. timestamps
o["order_ts"] = pd.to_datetime(o["order_ts"])
# ii. exact duplicates
n = len(o); o = o.drop_duplicates(); print("ii.  exact duplicate rows removed:", n - len(o))
# iii. keep the latest record per order_id
n = len(o); o = o.sort_values("order_ts").drop_duplicates("order_id", keep="last"); print("iii. older pending records removed:", n - len(o))
# types
for col in ["quantity", "discount_pct"]:
    o[col] = pd.to_numeric(o[col])
# iv. amounts
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
o = o.merge(p[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o["order_amount"].isna()
o.loc[miss, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss]
print("iv.  unparseable amounts rebuilt from price:", int(miss.sum()))
# v. delivery minutes
o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
neg = o["delivery_minutes"] < 0
o.loc[neg, "delivery_minutes"] = np.nan
print("v.   negative delivery times set to missing:", int(neg.sum()))
done = o["status"] == "completed"
med = o[done].groupby("category")["delivery_minutes"].transform("median")
fill = done & o["delivery_minutes"].isna()
o.loc[fill, "delivery_minutes"] = med[fill]
print("v.   completed orders filled with category median:", int(fill.sum()))
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)

print("\nREFERENCE CHECK  rows:", len(o), "| completed:", int(done.sum()), "| cancelled:", int((o.status == "cancelled").sum()),
      "| completed revenue:", round(o.loc[o.status == "completed", "order_amount"].sum(), 2))

# ---------------- save CSVs ----------------
c.to_csv("data/clean/customers.csv", index=False)
p.to_csv("data/clean/products.csv", index=False)
o.to_csv("data/clean/orders.csv", index=False)

# ---------------- SQLite ----------------
cs, os_ = c.copy(), o.copy()
cs["signup_date"] = cs["signup_date"].dt.strftime("%Y-%m-%d")
os_["order_ts"] = os_["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
con = sqlite3.connect("data/instamart.db")
cs.to_sql("customers", con, if_exists="replace", index=False)
p.to_sql("products", con, if_exists="replace", index=False)
os_.to_sql("orders", con, if_exists="replace", index=False)
con.close()
print("saved data/clean/*.csv and data/instamart.db")
