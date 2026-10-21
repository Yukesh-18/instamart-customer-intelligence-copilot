# Instamart Customer Intelligence & Support Copilot

A four-part capstone: a trusted analytical database (Part A), customer segments and a lapse model (Part B), a safe and cost-aware support-ticket triage pipeline (Part C), and a policy RAG assistant with a refund agent (Part D). All data is synthetic and fictional, and all amounts are in Indian Rupees (INR). Every language-model call goes through the supplied offline `mock_llm.py`, so no API key, account or network connection is needed.

## Overview

| Part | What it builds | Main outputs |
|---|---|---|
| A | Cleaned data, SQLite database, 7 SQL queries, pandas cross-check | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | RFM table, K-means segments, lapse prediction model | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Guard-railed ticket triage (privacy, retries, validation, token budget) | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Policy RAG assistant and a tool-using refund agent | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

`generate_data.py` and `mock_llm.py` are the supplied files and are unmodified.

## Reproduce

Run these from a fresh clone (Windows PowerShell shown; on macOS/Linux use `source .venv/bin/activate`):

1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python src/audit.py`
6. `python src/clean.py`
7. `python src/validate.py`
8. `python src/run_queries.py`
9. `python src/pandas_check.py`

(Steps for Parts B, C and D will be added here as they are built.)

Step 4 must end with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`, and step 7 must print `ALL CHECKPOINTS PASSED`.

## Part A: Data Foundation

### Audit of the raw data
| Table | Shape | Notes |
|---|---|---|
| customers | 1200 x 5 | age blank 61 (5.08%), gender blank 30 (2.50%) |
| products | 40 x 4 | no missing values |
| orders | 13084 x 9 | delivery_minutes blank 2541 (19.42%) |
| tickets | 164 x 5 | order_id blank 8, text blank 4 |

| Audit figure | Value |
|---|---|
| raw orders rows | 13084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12309 |
| unparseable order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2541 |
| customers: blank age / age outside 10-100 | 61 / 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct gender spellings / blank gender | 10 / 30 |

### Cleaning rules and what each changed
**Customers:** city stripped and title-cased (476 values changed, 6 cities left); gender mapped to male/female/unknown (30 blanks became unknown); 6 impossible ages set to missing; 67 missing ages filled with the median age of the customer's city; signup_date converted to a datetime.

**Orders (in this order):**
1. order_ts converted to a datetime.
2. 119 exact duplicate rows removed.
3. 656 older `pending` records removed (kept the latest record per order_id).
4. 112 unparseable amounts rebuilt as round(price x quantity x (1 - discount/100), 2). This is lower than the 120 in the audit because the audit counted before duplicates were removed.
5. 105 negative delivery times set to missing. 774 missing delivery times on completed orders were filled with the median of that order's product category. Cancelled orders keep it missing.

Result: 12,309 orders (11,110 completed, 1,199 cancelled); completed revenue INR 2,998,668.80.

### SQL results (`sql/queries.sql`)
| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows summing to INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 |
| Q6 window functions | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows where RANK differs from ROW_NUMBER |
| Q7 running total | 183 days; final cumulative INR 2,998,668.80 |

RANK and ROW_NUMBER differ when two customers in the same city have exactly the same total spend: RANK gives them the same rank, ROW_NUMBER still gives them different numbers.

### Exploratory analysis (completed orders)
This is six months of completed orders from six cities, with amounts in INR. Order amounts are strongly right-skewed: the mean is 269.91 against a median of 135.00, and skewness is 4.61 (Q1 64.80, Q3 297.00). There are 897 outliers above the upper IQR fence of 645.30 and none below the lower fence. The strongest relationship is order_amount with discount_pct at -0.095, then quantity at +0.087, while delivery_minutes is essentially unrelated (-0.003). All correlations are weak, and correlation alone does not establish causation.

Per-city check (pandas `groupby`): Bengaluru total 523,643.40, mean 276.47, count 1,894; Mumbai total 479,058.50, mean 297.00, count 1,613. The pivot table grand total equals the SQL total.

## Part B: Customer Segments and Lapse Prediction

_In progress._

## Part C: Support Ticket Triage with Guardrails

_In progress._

## Part D: Policy RAG Assistant and Refund Agent

_In progress._

## Limitations and Next Steps

_To be completed after Parts B to D._