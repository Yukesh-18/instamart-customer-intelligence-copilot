"""Part D tasks 1-4: chunking, TF-IDF retrieval, grounded answering and evaluation."""
import glob, os, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
import mock_llm

# ---------- Task 1: chunking ----------
def chunk_words(words, size=40, overlap=10):
    """Windows of `size` words, moving forward by size - overlap; stop after the first chunk that reaches the last word."""
    step, chunks, start = size - overlap, [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            return chunks
        start += step

def build_chunks(size=40, overlap=10, folder="data/policies"):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        for i, text in enumerate(chunk_words(words, size, overlap)):
            rows.append({"source": os.path.basename(path), "chunk_id": i, "text": text})
    return rows

# ---------- Task 2: TF-IDF retrieval ----------
class Retriever:
    def __init__(self, size=40, overlap=10):
        self.chunks = build_chunks(size, overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(sims[i]), 4), "text": self.chunks[i]["text"]} for i in top]

RETRIEVER = Retriever()
def retrieve(query, k=3):
    return RETRIEVER.retrieve(query, k)

# ---------- Task 3: grounded answer ----------
SYSTEM_POLICY = ("You are a POLICY ASSISTANT. Answer only from the supplied context. If the context does not contain "
                 "the information, say that you could not find it in the policy documents.")

def answer(question, retriever=None, k=3):
    hits = (retriever or RETRIEVER).retrieve(question, k)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_POLICY},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    text = mock_llm.chat(messages).choices[0].message.content
    return text, [h["source"] for h in hits]

# ---------- Task 4: evaluation ----------
def evaluate(retriever, label):
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    ans, una = ev[ev.expected_doc != "NONE"], ev[ev.expected_doc == "NONE"]
    doc_hits = fact_hits = fallback = 0
    rows, failures = [], []
    for _, r in ans.iterrows():
        text, sources = answer(r.question, retriever)
        d, f = r.expected_doc in sources, r.expected_fact in text
        doc_hits += d; fact_hits += f
        rows.append((r.question, r.expected_doc, d, f))
        if not (d and f):
            failures.append((r.question, r.expected_fact, text, sources))
    for _, r in una.iterrows():
        text, _ = answer(r.question, retriever)
        fallback += text.startswith("I could not find")
    print(f"\n=== {label} ===")
    print(f"answerable ({len(ans)}): expected doc retrieved {doc_hits}/{len(ans)} | expected fact in answer {fact_hits}/{len(ans)}")
    print(f"unanswerable ({len(una)}): fallback answers {fallback}/{len(una)}")
    return rows, failures

if __name__ == "__main__":
    chunks = build_chunks()
    counts = pd.Series([c["source"] for c in chunks]).value_counts().sort_index()
    print("chunks per document:", counts.to_dict(), "| total:", len(chunks))
    print("\nTop 3 for 'refund policy late delivery':")
    for h in retrieve("refund policy late delivery"):
        print(f"  {h['source']}  score={h['score']}")
    rows, _ = evaluate(RETRIEVER, "overlap 10")
    print(pd.DataFrame(rows, columns=["question", "expected_doc", "doc_retrieved", "fact_in_answer"]).to_string(index=False))
    _, failures = evaluate(Retriever(40, 0), "overlap 0")
    for q, fact, text, sources in failures:
        print("\nFAILS:", q, "\n  expected fact:", fact, "\n  answer:", text, "\n  sources:", sources)