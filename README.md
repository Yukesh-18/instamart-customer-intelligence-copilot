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

### RFM and K-means
Snapshot date 2025-10-01, completed orders only: 1,175 customers (recency median 19 days, range 0-180; frequency mean 9.46, max 33; monetary mean INR 2,552.06, max INR 14,766.55). The three columns were standardised with `StandardScaler`.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Inertia always falls as K grows, because more centres mean every point is closer to one, so K=8 is lower than K=2 by construction and inertia alone cannot pick K. I chose **K=4**: the elbow bends there and the silhouette peaks there at 0.526, and four segments are easy to explain to a marketing team.

### Segment profiles
| Segment | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|
| Premium Big-Spenders (Premium) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to new electronics, priority support |
| Frequent Everyday Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | Bundles and a repeat-order subscription |
| Occasional Regulars | 436 | 28.86 | 3.77 | 1,057.62 | Reminder nudges and small coupons |
| Dormant Low-Value | 182 | 109.04 | 2.20 | 685.99 | A one-time win-back offer |

PCA on the scaled RFM matrix: explained variance 0.634 and 0.218, total 0.852. Segment numbers are only labels with no order. The Premium segment is the one with the highest mean monetary value. Output: `outputs/customer_segments.csv`.

### Lapse problem
Cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff: 1,156 customers, 443 lapsed (38.32%). A customer is lapsed (1) if they have no completed order from 2025-09-01 to 2025-09-30. Features use only orders before the cutoff: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender and tenure. Segment labels were **not** used as features. Split 80/20, stratified, `random_state=42`: 924 train and 232 test rows. The always-"not lapsed" baseline has accuracy 0.616 and lapsed-class F1 0.

### Leakage control
No encoder, scaler or resampler saw test rows. Encoding and scaling live inside a scikit-learn `Pipeline`, fitted on the training rows only. The scaler learned a `frequency` mean of 8.1147 from the training rows, while the full-cohort mean is 8.0285. They differ because the scaler must not know anything about the test rows. No feature column has a missing value.

### Models
| Model | Best params | CV F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Baseline logistic regression | default | - | 0.615 | 0.629 | 0.622 |
| Tuned logistic regression | C=100 | 0.644 | 0.615 | 0.629 | 0.622 |
| Tuned random forest | depth 8, leaf 10 | 0.689 | 0.652 | 0.674 | 0.663 |

Test confusion matrix for the random forest (TN/FP/FN/TP): 111 / 32 / 29 / 60. I selected the **random forest** (F1 0.663, above the 0.55 requirement). [Write 3-5 sentences in your own words here on the cost of missing a lapsing customer versus sending an offer to a loyal one.]

### Class-balance experiment
Over-sampling the lapsed class in the training set only (570 rows per class) changed the best logistic regression on the original, unbalanced test set: recall rose from 0.629 to 0.865, precision slipped from 0.615 to 0.592, and F1 went from 0.622 to 0.703. Recall moved most. For the retention team this means they catch far more of the customers who are really leaving, at the price of a few more offers sent to customers who would have stayed.

### Interpretation and files
Top 5 random-forest importances: frequency 0.293, monetary 0.262, recency 0.120, cancel_rate 0.062, avg_delivery 0.051. Top logistic coefficients: frequency (-1.627, more orders lowers lapse risk), favourite category Electronics (-0.923) and city Delhi (-0.364). The saved model reloaded with `joblib.load` reproduces the original predictions exactly. `outputs/retention_list.csv` lists the 10 test-set Premium customers with the highest predicted lapse probability; 1 of the 10 actually lapsed.

The Premium flag comes from whole-window RFM (including the September window that defines the label), so it is a post-hoc label. The retention list is illustrative and is not a leakage-free evaluation (see Limitations).

## Part C: Support Ticket Triage with Guardrails

_In progress._

## Part D: Policy RAG Assistant and Refund Agent

_In progress._

## Limitations and Next Steps

_To be completed after Parts B to D._