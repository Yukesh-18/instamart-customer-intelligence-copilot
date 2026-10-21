"""mock_llm.py -- deterministic, offline stand-in for a hosted LLM chat API (no key, no network, no cost).
Interface mirrors a hosted chat API:  response = chat(messages, tools=None)
    response.choices[0].message.content / .tool_calls ; response.choices[0].finish_reason ; response.usage
Three behaviours are selected by a marker word in the FIRST (system) message:
    "CLASSIFY"  -> support-ticket classifier       "POLICY ASSISTANT" -> grounded policy answerer
    "AGENT"     -> refund agent that requests tools (ReAct style)"""
import json, math, re
from types import SimpleNamespace as NS

class TransientLLMError(Exception):
    """Simulated temporary failure (timeout / server busy). Retrying the same call succeeds."""

_state = {"calls": 0, "ok": 0}      # call counters that drive the simulated misbehaviour

def reset():
    """Restart the counters. Call once before each independent batch run."""
    _state["calls"] = 0; _state["ok"] = 0

CATS = ["billing", "delivery", "technical", "fraud"]
KW = {"fraud": {"unauthorized", "fraud", "hacked", "stolen"},
      "billing": {"charged", "invoice", "bill", "refund", "deducted", "payment"},
      "technical": {"app", "crash", "login", "otp", "error", "freezes"},
      "delivery": {"arrived", "rider", "delivery", "delivered", "late"}}

def est_tokens(text):
    return math.ceil(len((text or "").split()) * 1.3)

def _words(t):
    return set(re.findall(r"[a-z]+", (t or "").lower()))

def _resp(content=None, tool_calls=None, finish="stop", messages=()):
    prompt = sum(est_tokens(m.get("content") or json.dumps(m.get("tool_calls", ""), default=str)) for m in messages)
    comp = est_tokens(content or json.dumps(tool_calls or "", default=str))
    msg = NS(role="assistant", content=content, tool_calls=[NS(**tc) for tc in (tool_calls or [])])
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=prompt, completion_tokens=comp, total_tokens=prompt + comp))

def _classify(messages):
    system, user = messages[0]["content"], messages[-1]["content"]
    _state["calls"] += 1
    if _state["calls"] % 9 == 0:                                 # every 9th call fails once
        raise TransientLLMError("503 server busy")
    _state["ok"] += 1
    order = ["fraud", "technical", "billing", "delivery"] if "EXAMPLES:" in system else ["billing", "fraud", "technical", "delivery"]
    w = _words(user)
    cat = next((c for c in order if KW[c] & w), "technical")
    if _state["ok"] % 13 == 0:                                   # chatty, non-JSON answer
        return _resp(f"The customer appears to have a {cat}-related problem.", messages=messages)
    if _state["ok"] % 17 == 0:                                   # JSON, but category outside the allowed list
        return _resp(json.dumps({"category": "shipping", "reason": "parcel movement"}), messages=messages)
    return _resp(json.dumps({"category": cat, "reason": "keyword match"}), messages=messages)

STOP = {"the", "a", "an", "is", "of", "to", "for", "in", "on", "how", "what", "when", "who", "are", "do", "does",
        "i", "my", "be", "can", "after", "from", "it", "or", "and", "at", "by", "which", "why", "if", "than", "that"}

def _stems(t):
    return {w[:5] for w in re.findall(r"[a-z]+", (t or "").lower()) if w not in STOP}

def _policy(messages):
    """Answers with the context sentence that best matches the question (rare words weigh more)."""
    user = messages[-1]["content"]
    m = re.search(r"CONTEXT:\s*(.*?)\s*QUESTION:\s*(.*)", user, re.S)
    context, question = (m.group(1), m.group(2)) if m else ("", user)
    sents = [s.strip() for s in re.split(r"(?<=[.])\s+", re.sub(r"\[[^\]]*\.txt\]", " ", context)) if s.strip()]
    qs = _stems(question)
    df = {q: sum(q in _stems(s) for s in sents) for q in qs}
    best, best_score = None, 0.0
    for s in sents:
        sc = sum(1.0 / df[q] for q in qs & _stems(s))
        if sc > best_score:
            best, best_score = s, sc
    if best is None:
        return _resp("I could not find that information in the policy documents.", messages=messages)
    return _resp(best, messages=messages)

def _agent(messages, tools):
    user = next(m["content"] for m in messages if m["role"] == "user")
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    call = lambda n, a: _resp(None, [{"id": f"call_{len(tool_msgs)+1}", "name": n, "arguments": a}], "tool_calls", messages)
    if "[LOOP]" in user:
        return call("search_policy", {"query": "refund policy"})
    if not tool_msgs:
        m = re.search(r"ORD-\d{6}", user)
        return call("get_order", {"order_id": m.group(0)}) if m else _resp("NEEDS_INFO: please share the order id.", messages=messages)
    last = tool_msgs[-1]; data = json.loads(last["content"]); name = last["name"]
    if name == "get_order":
        if "error" in data:
            return _resp("NOT_FOUND: no such order.", messages=messages)
        return call("search_policy", {"query": "refund policy late delivery"})
    order = json.loads(next(m for m in tool_msgs if m["name"] == "get_order")["content"])
    if name == "search_policy":
        if order["status"] != "completed":
            return _resp("DENIED: only completed orders can be refunded.", messages=messages)
        return call("calculate_refund", {"order_amount": order["order_amount"], "delivery_minutes": order["delivery_minutes"]})
    if name == "calculate_refund":
        amt = data["refund_inr"]
        if amt == 0:
            return _resp("DENIED: delivery was within 45 minutes.", messages=messages)
        if amt > 1000:
            return call("escalate_to_human", {"order_id": order["order_id"], "refund_inr": amt})
        return call("issue_refund", {"order_id": order["order_id"], "refund_inr": amt})
    if name == "issue_refund":
        return _resp(f"REFUND: INR {data['refund_inr']} issued.", messages=messages)
    if name == "escalate_to_human":
        return _resp("ESCALATE: refund above INR 1000 sent to a human lead.", messages=messages)
    return _resp("DENIED: unable to process.", messages=messages)

def chat(messages, tools=None):
    head = messages[0]["content"].upper()
    if "CLASSIFY" in head:
        return _classify(messages)
    if "POLICY ASSISTANT" in head:
        return _policy(messages)
    if "AGENT" in head:
        return _agent(messages, tools)
    raise ValueError("first message must name one of: CLASSIFY, POLICY ASSISTANT, AGENT")
