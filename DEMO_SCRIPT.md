# Short Demonstration Script

## 1. Introduction

“GenAI Customer Support Assistant with RAG and Analytics is a customer-support prototype. Instead of sending a customer message directly to an LLM, the system first retrieves relevant support policies. The retrieved evidence is then used to generate a grounded response, classify the ticket, and decide whether human review is required.”

Mention that the dataset contains **200 labeled question records mapped to 50 core policies**.

## 2. Normal automated case

Enter:

```text
How many business days does standard delivery usually take?
```

Explain the output:

- Delivery category
- normal urgency
- no human escalation
- standard-delivery policy retrieved
- evidence panel shows the support knowledge used for the answer

## 3. High-risk escalation case

Enter:

```text
My debit card shows the same order twice. This is urgent.
```

Explain:

- Payments category
- high urgency
- duplicate-payment policy requires human review
- the deterministic guardrail keeps human escalation enabled even if an LLM classification is different

Alternative security example:

```text
Someone changed details on my account and I think it was hacked.
```

## 4. RAG explanation

Open the evidence panel and Architecture tab.

Explain that TF-IDF is retained as a reproducible offline baseline. When a Gemini API key is available, Gemini semantic embeddings can retrieve similar meanings even when the wording differs. Hybrid mode combines the lexical and semantic signals.

## 5. Evaluation

Open the Evaluation tab and `EVALUATION_RESULTS.txt`.

Explain:

- 60 held-out/noisy evaluation queries
- Top-1 TF-IDF baseline = 65.0%
- Top-3 TF-IDF baseline = 86.7%
- the imperfect lexical baseline motivates semantic retrieval

Do not claim semantic/hybrid benchmark numbers unless they were actually generated with a valid Gemini API key.

## 6. Analytics

After running several examples, open the Analytics tab and show:

- interaction count
- escalation rate
- average confidence
- category and urgency charts
- retrieval mode
- recent interaction table

## Closing line

“The main learning from this project is that a useful GenAI support application needs grounding, evaluation, guardrails and observability, not only an LLM prompt.”
