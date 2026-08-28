# GenAI Customer Support Assistant with RAG and Analytics

This deployment uses **Gemini for grounded answer generation** and **local Latent Semantic Analysis (LSA) for semantic retrieval**, avoiding embedding-API quota dependency in the live demo.

A deployment-ready Generative AI customer-support prototype built with **Streamlit**, **Gemini API**, retrieval-augmented generation (RAG), deterministic escalation guardrails, evaluation, and analytics.

## Live Demo

**Streamlit deployment URL:** `ADD_AFTER_DEPLOYMENT`

> Replace the placeholder above with the real `https://...streamlit.app` URL after deploying the project. Never place an API key in this README or in the GitHub repository.

## Business use case

Customer-support teams receive repetitive questions about accounts, delivery, returns, refunds, payments, orders, and security. A basic chatbot can answer quickly, but it can also hallucinate policies or mishandle high-risk cases.

This prototype addresses that problem by retrieving approved FAQ/policy evidence before generating a response. It also classifies the ticket and applies deterministic business rules for human escalation.

## Main capabilities

- **200-record synthetic labeled FAQ dataset** mapped to **50 core support policies/intents**
- **TF-IDF lexical baseline** for reproducible offline retrieval
- **local LSA semantic retrieval** using `Local LSA (TruncatedSVD)`
- **Hybrid RAG** combining lexical and semantic retrieval
- **Grounded Gemini response generation** using `gemini-3.5-flash`
- ticket intelligence: category, intent, sentiment, urgency and confidence
- deterministic human-escalation guardrails
- Top-1 and Top-3 retrieval evaluation on 60 held-out/noisy queries
- Streamlit analytics dashboard with SQLite interaction logging
- automated smoke/guardrail tests
- safe fallback mode when the Gemini API is unavailable

## Architecture

```text
Customer message
      |
      v
Retrieval router
  |-- TF-IDF lexical baseline
  |-- local LSA semantic retrieval
      |
      v
Hybrid Top-3 policy retrieval
      |
      v
Approved policy evidence
      |
      v
Gemini grounded response generation
      |
      v
Ticket intelligence
(category / intent / sentiment / urgency / confidence)
      |
      v
Deterministic business guardrail
(policy escalation / urgent-risk signals / low confidence)
      |
      +--> automated response
      |
      +--> human escalation
      |
      v
SQLite interaction log -> Streamlit analytics
```

## Dataset

`data/faq.csv` contains **200 synthetic customer-support examples** mapped to **50 core policy IDs**. Each core policy has multiple paraphrased customer questions.

Important fields include:

- `faq_id`
- `policy_id`
- `category`
- `intent`
- `question`
- `answer`
- `keywords`
- `priority`
- `escalation_required`
- `risk_type`

The dataset is synthetic and contains no real customer personal information.

## Retrieval design

### 1. TF-IDF lexical baseline

The offline baseline combines word and character TF-IDF signals. It remains available as a measurable non-LLM baseline.

Measured on the included 60-query held-out evaluation set:

- **Top-1: 65.0% (39/60)**
- **Top-3: 86.7% (52/60)**

### 2. Local semantic retrieval (LSA)

The advanced retrieval layer uses **Latent Semantic Analysis (LSA)** implemented with scikit-learn `TruncatedSVD`. It converts the TF-IDF document matrix into dense concept vectors and compares customer queries using cosine similarity. This semantic layer runs locally, so the live deployment does not depend on a separate embedding API quota.

Measured semantic-only result:

- **Top-1: 53.3% (32/60)**
- **Top-3: 81.7% (49/60)**

The semantic-only result is intentionally reported rather than hidden: on this small synthetic FAQ dataset, the tuned lexical baseline is stronger by itself.

### 3. Hybrid RAG

Hybrid mode combines the lexical ranking with local LSA semantic evidence using reciprocal-rank fusion. The lexical signal remains dominant for benchmark stability while the dense semantic signal can help with paraphrase/tie handling.

Measured hybrid result:

- **Top-1: 65.0% (39/60)**
- **Top-3: 86.7% (52/60)**

The top three distinct policy IDs are supplied as grounded evidence to Gemini for customer-facing response generation.

## GenAI response and safety logic

The Gemini model receives only the retrieved support context and is instructed not to invent policies, refund amounts, account facts, dates or guarantees.

The LLM returns structured ticket information including:

- customer-facing answer
- category
- intent
- sentiment
- urgency
- human-escalation decision
- reason

A deterministic guardrail runs after generation. If a matched policy requires escalation, the message contains a high-risk signal, or retrieval confidence is below the safety threshold, human escalation remains enabled even if the model predicts otherwise.

## Project structure

```text
genai_customer_support_assistant/
├── app.py
├── evaluate.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── DEMO_SCRIPT.md
├── EVALUATION_RESULTS.txt
├── data/
│   ├── faq.csv
│   └── evaluation_queries.csv
├── src/
│   ├── __init__.py
│   ├── analytics.py
│   ├── knowledge_base.py
│   └── llm_service.py
└── tests/
    ├── conftest.py
    └── test_smoke.py
```

## Local setup

Python 3.10+ is recommended.

### Windows

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

## Gemini API configuration

Create your own Gemini API key in Google AI Studio and add it only to your local `.env` file:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash
LSA_COMPONENTS=64
RETRIEVAL_CONFIDENCE_THRESHOLD=0.08
```

**Never commit or submit a real API key.** The repository should contain only `.env.example`.

Official Gemini API setup documentation: https://ai.google.dev/gemini-api/docs

## Run the application

```bash
python -m streamlit run app.py
```

Without a Gemini API key, retrieval still works locally and the application uses the deterministic grounded fallback for the final response.

## Evaluation

Run the three retrieval modes locally; no API key is required for retrieval evaluation:

```bash
python evaluate.py --mode tfidf
python evaluate.py --mode semantic
python evaluate.py --mode hybrid
```

Recorded results on the included 60-query benchmark:

| Retrieval mode | Top-1 | Top-3 |
|---|---:|---:|
| TF-IDF baseline | 65.0% | 86.7% |
| Local LSA semantic | 53.3% | 81.7% |
| Hybrid TF-IDF + LSA | 65.0% | 86.7% |

These results are reproducible because all retrieval evaluation runs locally.

## Automated tests

```bash
pytest -q
```

The included tests cover dataset/retrieval loading, account-deletion escalation, duplicate-payment escalation, and normal-delivery handling.

## Deploy to Streamlit Community Cloud

### 1. Create a GitHub repository

Create a repository such as:

```text
genai-customer-support-assistant
```

Upload the project files. Do **not** upload `.env`, `.cache/`, `.venv/`, `support_logs.db`, or any API key.

### 2. Connect Streamlit Community Cloud

Open https://share.streamlit.io/ and sign in with GitHub.

Create a new app and select:

```text
Repository: <your-username>/genai-customer-support-assistant
Branch: main
Main file path: app.py
```

### 3. Add Streamlit Secrets

In **Advanced settings -> Secrets**, add:

```toml
GEMINI_API_KEY = "your_real_gemini_api_key"
GEMINI_MODEL = "gemini-3.5-flash"
LSA_COMPONENTS = "64"
RETRIEVAL_CONFIDENCE_THRESHOLD = "0.08"
```

Do not create or commit a real `.streamlit/secrets.toml` file to GitHub.

### 4. Deploy and update this README

After deployment, Streamlit will provide a public `https://...streamlit.app` URL. Replace `ADD_AFTER_DEPLOYMENT` in the **Live Demo** section with that actual URL before the final internship submission.

Official Streamlit deployment documentation: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Short demonstration

Use `DEMO_SCRIPT.md`.

Recommended flow:

1. Open the deployed Streamlit application.
2. Run a normal delivery question and show retrieved evidence plus the generated response.
3. Run a duplicate-payment or hacked-account case and show mandatory human escalation.
4. Open Analytics and show the logged interaction metrics.
5. Open Evaluation and explain TF-IDF, local LSA semantic retrieval, and the hybrid RAG design.
6. Open Architecture and explain grounding, guardrails and observability.

## Limitations

- The knowledge base is synthetic rather than a real company's approved documentation.
- Gemini answer generation requires network access, a valid API key and available generation quota; retrieval itself is local and quota-independent.
- Sentiment and urgency classification are prototype-level.
- SQLite analytics on a free cloud deployment should be treated as demonstration data and may not provide production-grade persistence.
- The prototype has no authentication, CRM connection, order database or real customer identity data.
- A production system would require PII controls, authorization, prompt-injection testing, audit logging, monitoring, rate/cost controls and formal security review.

## Future improvements

- use real approved support documents with chunking and document versioning
- evaluate answer groundedness and policy compliance
- add multilingual support
- add PII detection and redaction
- integrate with CRM/ticketing systems
- add agent feedback and correction loops
- add authentication and production observability

## Submission security note

The submission must not contain a real API key, `.env` file, Streamlit secrets file, embedding cache, local database, virtual environment, or generated Python cache files.
