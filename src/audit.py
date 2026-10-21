"""Part A task 2: audit the raw tables. Read everything as text so nothing is silently converted."""
import pandas as pd

def load(name):
    return pd.read_csv(f"data/raw/{name}.csv", dtype=str, keep_default_na=False)

def missing_table(df, title):
    miss = (df.apply(lambda c: c.str.strip() == "")).sum()
    out = pd.DataFrame({"dtype_in_file": "text", "missing": miss, "missing_pct": (miss / len(df) * 100).round(2)})
    print(f"\n=== {title}: shape={df.shape} ===")
    print(out)

cust, prod, orders, tickets = (load(n) for n in ["customers", "products", "orders", "tickets"])
for n, d in [("customers", cust), ("products", prod), ("orders", orders), ("tickets", tickets)]:
    missing_table(d, n)

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
deliv = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(cust["age"], errors="coerce")
print("\n=== Required audit figures ===")
print("orders rows                      :", len(orders))
print("exact duplicate rows             :", int(orders.duplicated().sum()))
print("rows with status pending         :", int((orders["status"] == "pending").sum()))
print("distinct order_id                :", orders["order_id"].nunique())
print("unparseable order_amount         :", int((amount_num.isna() & (orders["order_amount"].str.strip() != "")).sum()))
print("negative delivery_minutes        :", int((deliv < 0).sum()))
print("blank delivery_minutes           :", int((orders["delivery_minutes"].str.strip() == "").sum()))
print("customers blank age              :", int((cust["age"].str.strip() == "").sum()))
print("customers age outside 10-100     :", int(((age < 10) | (age > 100)).sum()))
print("distinct raw city spellings      :", cust["city"].nunique())
g = cust.loc[cust["gender"].str.strip() != "", "gender"]
print("distinct non-blank gender spells :", g.nunique(), sorted(g.unique()))
print("customers blank gender           :", int((cust["gender"].str.strip() == "").sum()))
