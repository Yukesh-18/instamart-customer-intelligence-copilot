"""Part A task 5: assert at least eight reference checkpoints from tasks 3 and 4 against data/instamart.db."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
c = pd.read_sql_query("SELECT * FROM customers", con)
o = pd.read_sql_query("SELECT * FROM orders", con)
p = pd.read_sql_query("SELECT * FROM products", con)
done, canc = o[o.status == "completed"], o[o.status == "cancelled"]

checks = [
    ("six clean cities", c.city.nunique() == 6),
    ("city counts", c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("no missing age", c.age.isna().sum() == 0),
    ("ages within 18-60", c.age.between(18, 60).all()),
    ("products rows", len(p) == 40),
    ("orders rows", len(o) == 12309),
    ("unique order ids", o.order_id.is_unique),
    ("completed rows", len(done) == 11110),
    ("cancelled rows", len(canc) == 1199),
    ("no pending", (o.status == "pending").sum() == 0),
    ("no missing amount", o.order_amount.isna().sum() == 0),
    ("no missing delivery (completed)", done.delivery_minutes.isna().sum() == 0),
    ("all cancelled delivery missing", canc.delivery_minutes.isna().sum() == 1199),
    ("completed revenue", round(done.order_amount.sum(), 2) == 2998668.80),
]
failed = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
assert not failed, f"FAILED: {failed}"
print("ALL CHECKPOINTS PASSED")
