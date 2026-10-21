"""Part A task 7: pandas cross-check of Q2 and exploratory analysis of order_amount."""
import pandas as pd
import numpy as np
import sqlite3

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
done = o[o.status == "completed"].merge(c[["customer_id", "city"]], on="customer_id").merge(p[["product_id", "category"]], on="product_id")

# (a) Q2 with pivot_table + margins, and groupby with named aggregation
pt = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("Pivot grand total:", round(pt.loc["All", "All"], 2), "(SQL total 2998668.80)")
rep = done.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")).round(2)
print(rep)

# (b) exploratory analysis of completed order_amount
a = done.order_amount
q1, q3 = a.quantile(.25), a.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {a.mean():.2f} | median {a.median():.2f} | skew {a.skew():.2f} | Q1 {q1:.2f} | Q3 {q3:.2f}")
print(f"IQR fences: lower {lo:.2f}, upper {hi:.2f} | outliers above: {(a > hi).sum()} | below: {(a < lo).sum()}")
print("\nCorrelation matrix:")
corr = done[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3))
