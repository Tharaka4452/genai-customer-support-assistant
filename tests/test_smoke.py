from pathlib import Path
from src.knowledge_base import KnowledgeBase
from src.llm_service import fallback_result


def _kb():
    base = Path(__file__).resolve().parents[1]
    return KnowledgeBase(str(base / "data" / "faq.csv"))


def test_dataset_and_retrieval():
    kb = _kb()
    assert len(kb.df) == 200
    assert kb.policy_count == 50
    retrieved = kb.search("I forgot my password and need to reset it", top_k=3, mode="tfidf")
    assert retrieved[0]["policy_id"] == "FAQ001"


def test_policy_escalation_guardrail():
    kb = _kb()
    retrieved = kb.search("I want to delete my account", top_k=3, mode="tfidf")
    result = fallback_result("I want to delete my account", retrieved)
    assert retrieved[0]["policy_id"] == "FAQ004"
    assert result["needs_human"] is True


def test_duplicate_charge_escalates():
    kb = _kb()
    retrieved = kb.search("My debit card was charged twice for one order", top_k=3, mode="tfidf")
    result = fallback_result("My debit card was charged twice for one order", retrieved)
    assert retrieved[0]["policy_id"] == "FAQ023"
    assert result["urgency"] == "High"
    assert result["needs_human"] is True


def test_normal_delivery_does_not_escalate():
    kb = _kb()
    retrieved = kb.search("How many days is standard delivery?", top_k=3, mode="tfidf")
    result = fallback_result("How many days is standard delivery?", retrieved)
    assert retrieved[0]["policy_id"] == "FAQ009"
    assert result["needs_human"] is False
