"""Part A task 6: run every query in sql/queries.sql from Python and print the figures the brief asks for."""
import re, sqlite3
import pandas as pd

sql_text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d)\s*$", sql_text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
con = sqlite3.connect("data/instamart.db")
R = {q: pd.read_sql_query(s, con) for q, s in queries.items()}

print("Q1 customers:", len(R["Q1"]))
print("Q2 rows:", len(R["Q2"]), "| sum:", round(R["Q2"].revenue.sum(), 2)); print(R["Q2"].head(1).to_string(index=False))
print("Q3 products:", len(R["Q3"]))
print("Q4 customers with no completed order:", len(R["Q4"]))
print("Q5 customers above average:", len(R["Q5"]), "| average:", R["Q5"].avg_total.iloc[0])
q6 = R["Q6"]
print("Q6 top customer per city:"); print(q6[q6.row_num == 1].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
q7 = R["Q7"]
print("Q7 days:", len(q7), "| last cumulative:", q7.cumulative_revenue.iloc[-1])
