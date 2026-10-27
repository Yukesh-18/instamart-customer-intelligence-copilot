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

All 164 tickets are processed in file order through `mock_llm.py` by `src/triage.py`.

### Input guardrails and redaction
Empty or whitespace-only tickets get status `skipped_empty` (4 tickets) and tickets above `MAX_INPUT_TOKENS = 400` estimated tokens get `flagged_long` (4 tickets, about 734 tokens each); neither is ever sent to the model. Before any prompt is built, these regular expressions are applied in this order: email `[\w.+-]+@[\w-]+\.[\w.]+` becomes `[EMAIL]`, phone `\b[6-9]\d{9}\b` becomes `[PHONE]`, and order id `ORD-\d{6}` becomes `[ORDER_ID]`.

| Substitution | Count |
|---|---|
| Emails | 55 |
| Phone numbers | 91 |
| Order ids | 132 |
| Total | 278 |

Re-applying the three patterns to the redacted text finds 0 matches. An assertion runs before every model call and fails if any prompt still contains an email, phone number or order id, and the user message is exactly the redacted ticket text.

### Prompts
The zero-shot prompt contains the word CLASSIFY, the four categories (billing, delivery, technical, fraud) and asks for JSON `{"category": "...", "reason": "..."}`. The few-shot prompt is the same plus a line `EXAMPLES:` followed by four labelled examples, one of them fraud.

### Retries
A wrapper allows up to 3 retries on `TransientLLMError`, waiting `BASE_DELAY * 2**n * random.uniform(0.9, 1.15)` seconds with `BASE_DELAY = 0.05`, and logs one line per retry. A full run had 19 transient errors, every one recovered by a first retry (retry number 0), with waits between 0.048 and 0.057 seconds; no ticket ended with a permanent error. Example log line: `retry ticket=T0002 retry_number=0 wait=0.0556s error=503 server busy`. `mock_llm.reset()` is called before each full run.

### Output validation
A reply is accepted as JSON if its category is one of the four allowed values (`classified_json`). If it is not valid JSON, I take the lowercase words with `re.findall(r"[a-z]+", text.lower())` and accept it if exactly one allowed category word is present (`classified_keyword`). Everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review. I never guess.

| Status | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot
| Prompt | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126/147) | 8,932 | 60.8 |
| Few-shot | 147 | 100% (147/147) | 21,100 | 143.5 |

Zero-shot confusion matrix (rows true, columns predicted):

| | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

[Write 3-4 sentences in your own words: every error is a fraud ticket predicted as billing (11) or a technical ticket predicted as billing (10)... wait, check the matrix and describe it exactly. Then say the few-shot prompt costs about 2.4 times the tokens per ticket and is worth it here because fraud tickets are the most expensive to miss.]

### Token budget and routing
Before each model call the running total of `usage.total_tokens` is compared with `TOKEN_BUDGET`; if it is already at or above the budget, no call is made and the ticket gets `budget_exhausted` and goes to human review. With the zero-shot prompt and `TOKEN_BUDGET = 6000`, 6,023 tokens were spent and **51 tickets** were `budget_exhausted` (`outputs/triage_budget_demo.csv`).

The final run (`outputs/triage_results.csv`, 164 rows) uses the few-shot prompt with `TOKEN_BUDGET = 100000`. Routing: skipped, long, invalid or budget-exhausted tickets go to `human_review` (NORMAL); fraud goes to `security_team` (HIGH); customers in the Premium segment from Part B go to `premium_desk` (HIGH); everyone else goes to `support_queue` (NORMAL). A customer not in the segment file counts as not premium.

| Route | Tickets |
|---|---|
| support_queue | 102 |
| security_team | 34 |
| human_review | 17 |
| premium_desk | 11 |

### What is checked and what is not
The pipeline checks personal data before every call, token counts and output format. A human still has to review the 17 `human_review` tickets, and the category of the 12 `classified_keyword` tickets was recovered from free text, not read from clean JSON.

## Part D: Policy RAG Assistant and Refund Agent

### Chunking, retrieval and grounding
The five policy files are split into chunks of 40 words with an overlap of 10 (stepping 30 words), stopping after the first chunk that reaches the last word: 15 chunks in total (refund 4, delivery 3, escalation 3, payment 3, privacy 2). Retrieval uses `TfidfVectorizer` fitted on the chunk texts and `cosine_similarity`, and `retrieve(query, k=3)` returns the source, score and text. For `refund policy late delivery` the top three chunks all come from `refund_policy.txt`.

[Write 2 sentences: TF-IDF weights a word by how often it is in a chunk (TF) and how rare it is across chunks (IDF). One limitation: it matches words, not meaning, so two phrasings of the same idea can score low.]

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing POLICY ASSISTANT, and a user message of the form `CONTEXT:` newline, the three chunks each prefixed with `[source file name]` on its own line, newline `QUESTION:` newline, the question.

| Evaluation | Overlap 10 | Overlap 0 |
|---|---|---|
| Answerable (12): expected doc in top 3 | 12/12 | 12/12 |
| Answerable (12): expected fact in answer | 12/12 | 10/12 |
| Unanswerable (2): fallback "I could not find..." | 2/2 | 2/2 |

With overlap 0 two answers break. For "What delivery time makes a completed order eligible for a full refund?" the answer came back as `receives a full refund of the order amount.` and lost "more than 45 minutes". The sentence is split across the boundary between two chunks, so the fact sits in a different chunk from the rest of the sentence. The 10-word overlap repeats those words in both chunks.

### Tools and agent
Five tools are described in `TOOLS_SPEC` (JSON-schema style) and run through `TOOL_MAP`: `get_order` (reads `data/instamart.db`), `search_policy` (uses my retrieval), `calculate_refund` (100% of the order amount if delivery took more than 45 minutes, otherwise 0), `issue_refund` and `escalate_to_human` (each appends a timestamped line with the tool name to `outputs/refund_ledger.jsonl`). Both ledger tools skip a second write for the same order, so a repeated tool call cannot refund twice.

`run_agent` is a ReAct loop (max 6 iterations, token limit 6000, time limit 10 s). The final status is the word before the colon in the final answer.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1037 |
| S2 | REFUND | 135.00 | 5 | 1037 |
| S3 | ESCALATE | 1999.00 | 5 | 1024 |
| S4 | DENIED | 0.00 | 4 | 717 |
| S5 | DENIED | 0.00 | 3 | 421 |
| S6 | NOT_FOUND | 0.00 | 2 | 128 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3168 |

### Stopping safely
Fallbacks return the steps so far and a short human-handoff summary.
- S7 hits the iteration limit (`FALLBACK_MAX_ITERATIONS`, 6 model calls, 3168 tokens, under the 6000 limit).
- Demo, S1 with `token_limit=200`: `FALLBACK_TOKEN_LIMIT` after 438 tokens; tools run before stopping: `get_order`, `search_policy`; the tool requested by the call that broke the limit was not run.
- Demo, S1 with `time_limit_seconds=0`: `FALLBACK_TIMEOUT` with 0 model calls.

### Crash and resume
After every tool observation, a checkpoint with the run id, scenario id, full messages, tokens and executed tools is written to `state/`. For S1 I raised a simulated crash right after the second tool execution, then called `resume_agent`.

| Phase | Tools executed |
|---|---|
| Before the crash | get_order, search_policy |
| After resume | calculate_refund, issue_refund |

Four executions in total, none repeated, and the final status is REFUND with INR 128.25.

### ReAct versus plan-and-execute
[Write at most 250 words. Include these measured ReAct numbers: S1 5 model calls, S5 3 calls, S6 2 calls. Make clear that the plan-and-execute numbers are estimates, since I did not build that agent. See the notes I gave you.]

## Limitations and Next Steps

**Limitations**
- **The Premium flag is a post-hoc, whole-window label.** It comes from RFM computed over the whole window, including September, which is the month that defines the lapse label. So the retention list is illustrative and is not a leakage-free evaluation. The lapse model's own features use only orders before 2025-09-01, and segment labels were not used as features.
- **The data is synthetic.** Customers were generated from three simple behaviour types, so the clean segments and the 100% few-shot accuracy are easier than real data would be.
- **The language model is a mock.** `mock_llm.py` matches keywords and fails on a fixed schedule, so the accuracy and token numbers show how the pipeline behaves and do not predict a real model's quality.
- **Retrieval matches words, not meaning.** TF-IDF can miss a question phrased with different words, and the 40-word chunks can split a sentence (shown by the zero-overlap run).
- **Only 12 policy questions were evaluated, and the test set for the lapse model has 232 rows,** so small differences between models (for example F1 0.62 versus 0.66) are not strong evidence.
- **Human review is still needed** for the tickets routed to `human_review`, for refunds above INR 1000, and for every `FALLBACK_*` outcome.

**Next steps**
- Check the lapse model with a time-based split and add a probability-threshold choice based on the rupee cost of a missed customer versus a wasted offer.
- Replace the keyword mock with a real model behind the same `chat` interface, and re-measure accuracy and cost per ticket.
- Use better retrieval (sentence-aware chunks, or embeddings) and a larger question set.
- Add a human-approval queue and an audit dashboard for the refund ledger.