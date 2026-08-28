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

Explain that TF-IDF is retained as a reproducible lexical baseline. The advanced semantic layer uses local Latent Semantic Analysis (TruncatedSVD) to create dense concept vectors without consuming embedding API quota. Hybrid mode combines lexical ranking with the local semantic signal, while Gemini is used only to generate the grounded customer-facing response.

## 5. Evaluation

Open the Evaluation tab and `EVALUATION_RESULTS.txt`.

Explain:

- 60 held-out/noisy evaluation queries
- TF-IDF baseline: Top-1 **65.0%**, Top-3 **86.7%**
- Local LSA semantic-only: Top-1 **53.3%**, Top-3 **81.7%**
- Hybrid TF-IDF + LSA: Top-1 **65.0%**, Top-3 **86.7%**
- explain that the small synthetic FAQ dataset favors the tuned lexical baseline, while LSA adds a quota-free semantic signal and the hybrid design remains robust

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
