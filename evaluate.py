from pathlib import Path
import argparse
import pandas as pd
from src.knowledge_base import KnowledgeBase

parser = argparse.ArgumentParser(description="Evaluate customer-support FAQ retrieval")
parser.add_argument("--mode", choices=["tfidf", "semantic", "hybrid", "auto"], default="tfidf")
args = parser.parse_args()

BASE = Path(__file__).parent
kb = KnowledgeBase(str(BASE / "data" / "faq.csv"))
tests = pd.read_csv(BASE / "data" / "evaluation_queries.csv")
rows=[]
for _, row in tests.iterrows():
    result = kb.search(row["query"], top_k=3, mode=args.mode)
    top1 = result[0] if result else {}
    top3_ids = [x.get("policy_id") for x in result]
    rows.append({
        "query": row["query"], "expected": row["expected_policy_id"],
        "retrieved": top1.get("policy_id", ""), "score": top1.get("score", 0.0),
        "top1_correct": top1.get("policy_id") == row["expected_policy_id"],
        "top3_correct": row["expected_policy_id"] in top3_ids,
        "retrieval_mode": top1.get("retrieval_mode", "None"),
    })
report=pd.DataFrame(rows)
print(report.to_string(index=False))
print()
print(f"Requested mode: {args.mode}")
print(f"Actual mode: {report['retrieval_mode'].mode().iloc[0] if len(report) else 'N/A'}")
print(f"Top-1 accuracy: {report['top1_correct'].mean():.1%} ({report['top1_correct'].sum()}/{len(report)})")
print(f"Top-3 accuracy: {report['top3_correct'].mean():.1%} ({report['top3_correct'].sum()}/{len(report)})")
if kb.last_retrieval_error:
    print(f"Last semantic retrieval warning: {kb.last_retrieval_error}")
