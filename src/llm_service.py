import json
import os
import re
from typing import Dict, List

NEGATIVE_WORDS = {
    "angry", "terrible", "bad", "worst", "frustrated", "annoyed", "upset",
    "disappointed", "unacceptable", "hate", "complaint", "missing", "broken"
}
URGENT_WORDS = {
    "urgent", "immediately", "asap", "fraud", "stolen", "charged twice", "duplicate charge",
    "security", "hacked", "compromised", "unauthorized", "legal", "emergency", "privacy"
}


def _rule_based_sentiment(text: str) -> str:
    lowered = text.lower()
    return "Negative" if any(word in lowered for word in NEGATIVE_WORDS) else "Neutral"


def _rule_based_urgency(text: str) -> str:
    lowered = text.lower()
    return "High" if any(word in lowered for word in URGENT_WORDS) else "Normal"


def fallback_result(query: str, retrieved: List[Dict]) -> Dict:
    top = retrieved[0] if retrieved else None
    confidence = float(top.get("score", 0.0)) if top else 0.0
    threshold = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.08"))

    if top and confidence >= threshold:
        answer = top["answer"]
        category = top["category"]
        intent = top.get("intent", "unknown")
    else:
        answer = (
            "I could not find a reliable answer in the approved support knowledge base. "
            "Please contact a human support agent for assistance."
        )
        category = "Other"
        intent = "unsupported"

    urgency = _rule_based_urgency(query)
    sentiment = _rule_based_sentiment(query)
    policy_escalation = bool(top) and str(top.get("escalation_required", "No")).strip().lower() == "yes"
    low_confidence = confidence < threshold
    needs_human = urgency == "High" or policy_escalation or low_confidence

    reasons = []
    if policy_escalation:
        reasons.append("the matched business policy requires human review")
    if urgency == "High":
        reasons.append("the message contains a high-risk or urgent signal")
    if low_confidence:
        reasons.append("retrieval confidence is below the safety threshold")
    reason = "; ".join(reasons) if reasons else "the request matches an approved low-risk support policy"

    return {
        "answer": answer,
        "category": category,
        "intent": intent,
        "sentiment": sentiment,
        "urgency": urgency,
        "confidence": round(confidence, 4),
        "needs_human": needs_human,
        "reason": reason,
        "mode": "Deterministic grounded fallback",
    }


def _extract_json(text: str) -> Dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def generate_support_response(query: str, retrieved: List[Dict]) -> Dict:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    deterministic = fallback_result(query, retrieved)
    if not api_key:
        return deterministic

    context_blocks = []
    for item in retrieved:
        context_blocks.append(
            f"[Policy {item['policy_id']} / Variant {item['faq_id']}]\n"
            f"Category: {item['category']}\nIntent: {item.get('intent','')}\n"
            f"Approved answer: {item['answer']}\n"
            f"Policy escalation required: {item.get('escalation_required','No')}\n"
            f"Priority: {item.get('priority','Normal')}\nRisk type: {item.get('risk_type','general')}\n"
            f"Retrieval confidence: {item['score']}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No relevant approved policy was retrieved."

    prompt = f"""
You are an AI Customer Support Assistant for an e-commerce business. Use only the approved support context below.

Rules:
- Never invent company policy, refund amounts, dates, guarantees, or account facts.
- If context is insufficient, clearly recommend a human agent.
- A policy marked 'Policy escalation required: Yes' MUST result in needs_human=true.
- Fraud, unauthorized access, privacy complaints, duplicate charges, unresolved refunds, and missing-delivered parcels require human review.
- Keep the customer-facing answer concise, professional, and empathetic without overpromising.
- Classify sentiment as Positive, Neutral, or Negative and urgency as High or Normal.

Customer message:
{query}

Approved context:
{context}

Return ONLY one valid JSON object with exactly these fields:
{{
  "answer": "string",
  "category": "string",
  "intent": "string",
  "sentiment": "Positive|Neutral|Negative",
  "urgency": "High|Normal",
  "needs_human": true,
  "reason": "short explanation"
}}
""".strip()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        parsed = _extract_json(response.text or "")

        # Deterministic policy guardrail overrides the model if business rules require escalation.
        if deterministic["needs_human"]:
            parsed["needs_human"] = True
            if deterministic["reason"] not in str(parsed.get("reason", "")):
                parsed["reason"] = f"{parsed.get('reason','').strip()} Guardrail: {deterministic['reason']}.".strip()

        parsed.setdefault("category", deterministic["category"])
        parsed.setdefault("intent", deterministic["intent"])
        parsed.setdefault("sentiment", deterministic["sentiment"])
        parsed.setdefault("urgency", deterministic["urgency"])
        parsed["confidence"] = deterministic["confidence"]
        parsed["mode"] = f"Grounded Gemini LLM: {model}"
        return parsed
    except Exception as exc:
        deterministic["mode"] = "Deterministic fallback after Gemini API error"
        deterministic["reason"] = (
            f"{deterministic['reason']}. API failure handled safely ({type(exc).__name__})."
        )
        return deterministic
