"""Part D tasks 5-8: tools, ReAct refund agent, safe stopping, checkpoint and resume."""
import csv, glob, json, os, re, sqlite3, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())
import mock_llm
from rag import retrieve

DB, LEDGER, RUNS, STATE_DIR = "data/instamart.db", "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl", "state"
os.makedirs("outputs", exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)

class SimulatedCrash(Exception):
    pass

# ---------- Task 5: tools ----------
def get_order(order_id):
    con = sqlite3.connect(DB)
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    """Refund rules: more than 45 minutes -> 100% of the order amount, otherwise 0."""
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0}

def _ledger_write(tool, order_id, refund_inr):
    """Append one line to the ledger; if this tool already recorded this order, do not write it twice."""
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            e = json.loads(line)
            if e["tool"] == tool and e["order_id"] == order_id:
                return {"refund_inr": e["refund_inr"]}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return _ledger_write("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger_write("escalate_to_human", order_id, refund_inr)

def _spec(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}}

TOOLS_SPEC = [
    _spec("get_order", "Read one order from the database by order id.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the best matching chunks.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund amount in INR from the order amount and delivery minutes.",
          {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund of up to INR 1000.", {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead.", {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

SYSTEM_AGENT = ("You are a refund AGENT for a quick-commerce support desk. Use the tools get_order, search_policy, "
                "calculate_refund, issue_refund and escalate_to_human. Finish with one line that starts with REFUND:, "
                "ESCALATE:, DENIED: or NOT_FOUND:.")

# ---------- Task 8: checkpoints ----------
def save_checkpoint(st):
    json.dump({k: st[k] for k in ("run_id", "scenario_id", "user_message", "messages", "tokens_used", "llm_calls",
                                  "steps", "executed", "limits")}, open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w"), indent=1)

def _fallback(st, status):
    names = [s["tool"] for s in st["steps"]]
    return finish(st, status, f"Handing over to a human agent. Summary: request '{st['user_message']}'; "
                              f"tools run so far: {', '.join(names) or 'none'}; stopped because {status}.")

def finish(st, status, final):
    return {"scenario_id": st["scenario_id"], "status": status, "final_answer": final, "llm_calls": st["llm_calls"],
            "tokens_used": st["tokens_used"], "steps": st["steps"], "executed": st["executed"]}

# ---------- Tasks 6, 7: the ReAct loop ----------
def _loop(st, crash_after=None):
    lim = st["limits"]
    t0 = time.monotonic()
    while st["llm_calls"] < lim["max_iterations"]:
        if time.monotonic() - t0 >= lim["time_limit_seconds"]:                     # 7c: wall-clock limit, checked first
            return _fallback(st, "FALLBACK_TIMEOUT")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > lim["token_limit"]:                                 # 7b: stop before running the requested tool
            return _fallback(st, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:                                                     # final answer
            final = msg.content
            return finish(st, final.split(":")[0].strip(), final)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        obs = TOOL_MAP[tc.name](**tc.arguments)
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        st["executed"].append(tc.name)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        save_checkpoint(st)                                                        # Task 8: after every observation
        if crash_after is not None and len(st["executed"]) == crash_after:
            raise SimulatedCrash(f"simulated crash after tool execution {crash_after}")
    return _fallback(st, "FALLBACK_MAX_ITERATIONS")                                # 7a

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, crash_after=None, run_id=None):
    mock_llm.reset()
    st = {"run_id": run_id or scenario_id, "scenario_id": scenario_id, "user_message": user_message,
          "messages": [{"role": "system", "content": SYSTEM_AGENT}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "steps": [], "executed": [],
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}}
    return _loop(st, crash_after)

def resume_agent(run_id):
    st = json.load(open(os.path.join(STATE_DIR, f"{run_id}.json")))
    return _loop(st)

def refund_of(result):
    for s in result["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0

if __name__ == "__main__":
    for p in [LEDGER, RUNS, "outputs/agent_demos.jsonl"] + glob.glob(os.path.join(STATE_DIR, "*.json")):
        if os.path.exists(p):
            os.remove(p)
    scenarios = list(csv.DictReader(open("data/raw/agent_scenarios.csv", encoding="utf-8")))
    print(f"{'id':3} {'status':26} {'expected':9} {'refund':>8} {'expected':>9} {'calls':>5} {'tokens':>6}")
    for sc in scenarios:
        r = run_agent(sc["message"], sc["scenario_id"])
        with open(RUNS, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: r[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
        print(f"{sc['scenario_id']:3} {r['status']:26} {sc['expected_outcome']:9} {refund_of(r):8.2f} {float(sc['expected_refund_inr']):9.2f} {r['llm_calls']:5d} {r['tokens_used']:6d}")

    s1 = scenarios[0]["message"]
    d1 = run_agent(s1, "S1", token_limit=200, run_id="S1-token-demo")
    d2 = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    print("\nstop-safely demos:  S1 token_limit=200 ->", d1["status"], f"(tokens {d1['tokens_used']}, tools run: {d1['executed']})",
          "|  S1 time_limit_seconds=0 ->", d2["status"], f"(model calls {d2['llm_calls']})")

    try:
        run_agent(s1, "S1", run_id="S1-crash-demo", crash_after=2)
    except SimulatedCrash as e:
        print("\ncrash demo:", e)
    done = json.load(open(os.path.join(STATE_DIR, "S1-crash-demo.json")))
    print("tools executed before the crash:", done["executed"])
    res = resume_agent("S1-crash-demo")
    print("after resume -> status:", res["status"], "| refund INR", refund_of(res), "| all tools executed:", res["executed"])
    print("each tool exactly once:", len(res["executed"]) == len(set(res["executed"])) == 4)
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| state files:", sorted(os.path.basename(p) for p in glob.glob("state/*.json")))