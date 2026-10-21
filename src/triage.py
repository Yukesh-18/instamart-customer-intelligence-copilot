"""Part C: safe, resilient, checked and cost-aware support-ticket triage built around the supplied mock_llm.py."""
import json, os, random, re, sys, time
from collections import Counter
import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm
from mock_llm import TransientLLMError, est_tokens

os.makedirs("outputs", exist_ok=True)
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# ---------- Task 2: redaction patterns (applied in this order) ----------
PATTERNS = [("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
            ("PHONE", r"\b[6-9]\d{9}\b", "[PHONE]"),
            ("ORDER_ID", r"ORD-\d{6}", "[ORDER_ID]")]

def redact(text, counts=None):
    for name, pat, repl in PATTERNS:
        text, n = re.subn(pat, repl, text)
        if counts is not None:
            counts[name] += n
    return text

# ---------- Task 3: the two prompts ----------
ZERO_SHOT = ("You CLASSIFY customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "duplicate charge"}
Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized payment"}
Ticket: The rider has not arrived and delivery is late. -> {"category": "delivery", "reason": "late delivery"}
Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}"""

# ---------- Input guard on every outgoing prompt ----------
def assert_clean(messages):
    for m in messages:
        for _, pat, _ in PATTERNS:
            assert not re.search(pat, m["content"]), "personal data found in a prompt sent to the model"
        assert m["content"].strip() != "", "empty message sent to the model"

# ---------- Task 4: retries with exponential backoff and jitter ----------
def call_with_retry(messages, ticket_id, stats, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            line = f"retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s error={e}"
            log.append(line)
            if stats['verbose']: print(line)
            stats["retries"] += 1
            stats["errors"] += 1
            time.sleep(wait)

# ---------- Task 5: validate the answer ----------
def validate(content):
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("category") in CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", None            # valid JSON but category outside the list
    except (json.JSONDecodeError, TypeError):
        found = [w for w in re.findall(r"[a-z]+", (content or "").lower()) if w in CATEGORIES]
        if len(set(found)) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", None

# ---------- the pipeline ----------
def run(tickets, system_prompt, budget=None, label=""):
    mock_llm.reset()
    random.seed(0)
    stats, log, total, results = Counter(), [], 0, []
    stats['verbose'] = (label == 'zero-shot')
    pii_counts = Counter()
    for _, t in tickets.iterrows():
        text = t["text"]
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"], "category": None, "tokens_used": 0}
        if text.strip() == "":                                                    # guard (a)
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:                                 # guard (b)
            row["status"] = "flagged_long"
        elif budget is not None and total >= budget:                              # Task 7 budget guard
            row["status"] = "budget_exhausted"
        else:
            clean = redact(text, pii_counts)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
            assert_clean(messages)
            resp = call_with_retry(messages, t["ticket_id"], stats, log)
            used = resp.usage.total_tokens
            total += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate(resp.choices[0].message.content)
        results.append(row)
    return pd.DataFrame(results), total, stats, log

if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", dtype=str, keep_default_na=False)
    seg = pd.read_csv("outputs/customer_segments.csv")

    # Task 1 figures
    est = tickets["text"].map(est_tokens)
    print("empty tickets:", int((tickets.text.str.strip() == "").sum()), "| long tickets:", int((est > MAX_INPUT_TOKENS).sum()),
          "| long ticket estimate:", int(est.max()), "tokens")

    # Task 2 figures
    counts = Counter()
    redacted = tickets["text"].map(lambda s: redact(s, counts))
    print("substitutions: emails", counts["EMAIL"], "| phones", counts["PHONE"], "| order ids", counts["ORDER_ID"], "| total", sum(counts.values()))
    left = sum(len(re.findall(p, s)) for s in redacted for _, p, _ in PATTERNS)
    print("patterns still found after redaction:", left)

    # Task 6: zero-shot vs few-shot
    summary = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        res, total, stats, log = run(tickets, prompt, label=name)
        res = res.merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
        ok = res[res.status.isin(["classified_json", "classified_keyword"])]
        acc = (ok.category == ok.true_category).mean()
        summary[name] = dict(classified=len(ok), accuracy=acc, total_tokens=total, per_ticket=total / len(ok))
        print(f"\n--- {name} ---\nstatuses: {res.status.value_counts().to_dict()}")
        print(f"transient errors: {stats['errors']} | retries: {stats['retries']}")
        print(f"classified: {len(ok)} | accuracy: {acc:.1%} ({int((ok.category == ok.true_category).sum())}/{len(ok)}) | total tokens: {total} | tokens/classified ticket: {total/len(ok):.1f}")
        if name == "zero-shot":
            print("confusion matrix (rows = true, columns = predicted):")
            print(pd.crosstab(ok.true_category, ok.category))
            open("outputs/retry_log.txt", "w").write("\n".join(log) + "\n")
            waits = [float(re.search(r"wait=([\d.]+)s", l).group(1)) for l in log]
            print("retry waits: min %.4f max %.4f (allowed 0.0450-0.0575 for retry 0)" % (min(waits), max(waits)))
            
    # Task 7a: budget demo (zero-shot, budget 6000)
    res, total, stats, _ = run(tickets, ZERO_SHOT, budget=6000, label="budget")
    res.to_csv("outputs/triage_budget_demo.csv", index=False)
    print("\n--- budget demo (TOKEN_BUDGET=6000, zero-shot) ---")
    print("budget_exhausted tickets:", int((res.status == "budget_exhausted").sum()), "| tokens spent:", total)

    # Task 7b: final routing (few-shot, budget 100000)
    res, total, stats, _ = run(tickets, FEW_SHOT, budget=100000)
    res = res.merge(seg[["customer_id", "segment_name", "is_premium"]], on="customer_id", how="left")
    def route(r):
        if r.status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            return "human_review", "NORMAL"
        if r.category == "fraud":
            return "security_team", "HIGH"
        if r.is_premium == 1:
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"
    res[["route", "priority"]] = res.apply(lambda r: pd.Series(route(r)), axis=1)
    res["segment_name"] = res["segment_name"].fillna("")
    res["category"] = res["category"].fillna("")
    cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    res[cols].to_csv("outputs/triage_results.csv", index=False)
    print("\n--- final routing (few-shot, budget 100000) ---")
    print("rows:", len(res), "| statuses:", res.status.value_counts().to_dict())
    print("routes:", res.route.value_counts().to_dict(), "| total tokens:", total)