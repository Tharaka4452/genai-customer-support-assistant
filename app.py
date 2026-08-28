import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.analytics import AnalyticsStore
from src.knowledge_base import KnowledgeBase
from src.llm_service import generate_support_response

load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "faq.csv"
DB_PATH = BASE_DIR / "support_logs.db"

st.set_page_config(page_title="GenAI Customer Support Assistant", page_icon="🤖", layout="wide")


@st.cache_resource
def load_kb():
    return KnowledgeBase(str(DATA_PATH))


@st.cache_resource
def load_store():
    return AnalyticsStore(str(DB_PATH))


kb = load_kb()
store = load_store()
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 GenAI Customer Support Assistant with RAG and Analytics")
st.caption("Hybrid RAG • grounded LLM replies • policy guardrails • human escalation • evaluation • analytics")

with st.sidebar:
    st.header("System configuration")
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    has_key = kb.has_api_key
    st.write(f"**Dataset:** {len(kb.df)} labeled FAQ records")
    st.write(f"**Core policies/intents:** {kb.policy_count}")
    st.write(f"**LLM:** `{model}`")
    st.write(f"**Embeddings:** `{embedding_model}`")
    if has_key:
        st.success("Gemini API key detected")
        if st.button("Test embedding API", use_container_width=True):
            ok, detail = kb.embedding_healthcheck()
            if ok:
                st.success(detail)
            else:
                st.error("Embedding API test failed")
                st.code(detail, language=None)
    else:
        st.warning("No API key: deterministic TF-IDF demo mode")

    mode_label = st.selectbox(
        "Retrieval strategy",
        ["Auto hybrid", "TF-IDF baseline", "Gemini embeddings", "Hybrid"],
        index=0,
        help="Auto uses hybrid retrieval with an API key and safely falls back to TF-IDF otherwise.",
    )
    mode_map = {
        "Auto hybrid": "auto", "TF-IDF baseline": "tfidf",
        "Gemini embeddings": "semantic", "Hybrid": "hybrid",
    }
    retrieval_mode = mode_map[mode_label]

    st.divider()
    st.markdown("**Pipeline**")
    st.write("Customer query → Retrieval → Approved policies → LLM → Rule guardrail → Escalation → Analytics")


tab_chat, tab_dashboard, tab_eval, tab_data, tab_about = st.tabs([
    "💬 AI Assistant", "📊 Analytics", "🧪 Evaluation", "🗂 Dataset", "🧠 Architecture"
])

with tab_chat:
    left, right = st.columns([2, 1])
    with left:
        query = st.text_area(
            "Customer message",
            placeholder="Example: My bank shows two charges for one order and I need help urgently.",
            height=120,
        )
        c1, c2 = st.columns([1, 4])
        submit = c1.button("Analyze & Reply", type="primary", use_container_width=True)
        clear = c2.button("Clear session history")
        if clear:
            st.session_state.history = []
            st.rerun()

        if submit:
            if not query.strip():
                st.warning("Enter a customer message first.")
            else:
                with st.spinner("Retrieving approved knowledge and generating support analysis..."):
                    retrieved = kb.search(query, top_k=3, mode=retrieval_mode)
                    result = generate_support_response(query, retrieved)
                    store.log(query, result, retrieved)
                st.session_state.history.insert(0, {"query": query, "result": result, "retrieved": retrieved})

        if st.session_state.history:
            item = st.session_state.history[0]
            result = item["result"]
            retrieved = item["retrieved"]
            top = retrieved[0] if retrieved else {}

            st.subheader("AI customer-facing response")
            st.info(result.get("answer", "No answer generated."))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Category", result.get("category", "Other"))
            m2.metric("Intent", result.get("intent", "unknown"))
            m3.metric("Sentiment", result.get("sentiment", "Neutral"))
            m4.metric("Urgency", result.get("urgency", "Normal"))
            m5.metric("Confidence", f"{float(result.get('confidence', 0.0)):.2f}")

            if result.get("needs_human"):
                st.error("⚠️ Human escalation required — " + result.get("reason", ""))
            else:
                st.success("✅ Automated handling allowed — " + result.get("reason", ""))

            st.caption(
                f"Response mode: {result.get('mode','Unknown')} | "
                f"Retrieval: {top.get('retrieval_mode','Unknown')} | "
                f"Top policy: {top.get('policy_id','N/A')}"
            )
            if kb.last_retrieval_error:
                st.warning("Semantic retrieval was unavailable; the safe TF-IDF fallback was used.")
                with st.expander("Embedding diagnostic"):
                    st.code(kb.last_retrieval_error, language=None)

            with st.expander("Evidence: retrieved approved policies", expanded=True):
                for source in retrieved:
                    st.markdown(
                        f"**{source['policy_id']} · {source['category']} · confidence {source['score']:.3f}**"
                    )
                    st.write(source["question"])
                    st.caption(source["answer"])
                    st.write(
                        f"Policy escalation: **{source['escalation_required']}** | "
                        f"Lexical: {source['lexical_score']:.3f} | "
                        f"Semantic: {source['semantic_score'] if source['semantic_score'] is not None else 'N/A'}"
                    )

        if len(st.session_state.history) > 1:
            st.subheader("Recent session history")
            for old in st.session_state.history[1:6]:
                with st.expander(old["query"][:100]):
                    st.write(old["result"].get("answer", ""))

    with right:
        st.subheader("Demo scenarios")
        prompts = [
            "How many business days does standard delivery usually take?",
            "My debit card shows the same order twice. This is urgent.",
            "I opened the box already. Can I still return it?",
            "Someone changed details on my account and I think it was hacked.",
            "My order failed but my bank still shows a pending payment.",
            "The refund amount is smaller than I expected.",
        ]
        for p in prompts:
            st.code(p, language=None)
        st.info("For the presentation, show one normal automated case and one policy-mandated human escalation case.")

with tab_dashboard:
    df = store.dataframe()
    if df.empty:
        st.info("Run a few demo interactions first. Analytics will appear here.")
    else:
        total = len(df)
        escalations = int(df["needs_human"].fillna(0).sum())
        negative = int((df["sentiment"] == "Negative").sum())
        avg_conf = float(df["confidence"].fillna(0).mean())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Interactions", total)
        k2.metric("Escalation rate", f"{(escalations / total):.1%}")
        k3.metric("Negative tickets", negative)
        k4.metric("Avg confidence", f"{avg_conf:.2f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Tickets by category")
            counts = df["category"].fillna("Other").value_counts().rename_axis("category").to_frame("tickets")
            st.bar_chart(counts)
        with right:
            st.subheader("Urgency distribution")
            urgency = df["urgency"].fillna("Unknown").value_counts().rename_axis("urgency").to_frame("tickets")
            st.bar_chart(urgency)

        st.subheader("Retrieval modes used")
        modes = df["retrieval_mode"].fillna("Unknown").value_counts().rename_axis("mode").to_frame("interactions")
        st.bar_chart(modes)

        st.subheader("Recent ticket intelligence")
        show_cols = [
            "created_at", "query", "category", "intent", "sentiment", "urgency",
            "needs_human", "confidence", "source_policy", "retrieval_mode", "response_mode"
        ]
        st.dataframe(df[[c for c in show_cols if c in df.columns]], use_container_width=True, hide_index=True)
        st.download_button(
            "Download interaction log (CSV)", df.to_csv(index=False).encode("utf-8"),
            file_name="customer_support_interactions.csv", mime="text/csv"
        )
        if st.button("Reset analytics data"):
            store.clear()
            st.rerun()

with tab_eval:
    eval_path = BASE_DIR / "data" / "evaluation_queries.csv"
    eval_df = pd.read_csv(eval_path)
    st.subheader("Retrieval benchmark")
    st.write(
        f"The benchmark contains **{len(eval_df)} held-out/noisy queries**. "
        "The expected label is the core policy ID, not an exact training question variant."
    )
    sample = eval_df.head(10).copy()
    st.dataframe(sample, use_container_width=True, hide_index=True)
    st.code("python evaluate.py --mode tfidf", language="bash")
    st.code("python evaluate.py --mode hybrid  # requires GEMINI_API_KEY", language="bash")
    st.caption("This design makes the baseline-vs-semantic comparison reproducible during the demo.")

with tab_data:
    st.subheader("Synthetic labeled support dataset")
    st.write(
        f"**{len(kb.df)} records**, **{kb.policy_count} core policies/intents**, "
        f"across **{kb.df['category'].nunique()} categories**. Each policy has multiple paraphrased customer questions."
    )
    summary = kb.df.groupby("category").agg(records=("faq_id", "count"), policies=("policy_id", "nunique")).sort_values("records", ascending=False)
    st.dataframe(summary, use_container_width=True)
    st.dataframe(
        kb.df[["faq_id", "policy_id", "category", "question", "priority", "escalation_required"]].head(50),
        use_container_width=True, hide_index=True
    )

with tab_about:
    st.subheader("Advanced RAG architecture")
    st.code("""
Customer Message
      ↓
Retrieval Router
  ├─ TF-IDF lexical baseline
  └─ Gemini semantic embeddings
      ↓
Hybrid Retrieval + Top-3 Policy Evidence
      ↓
Grounded LLM (approved context only)
      ↓
Structured Ticket Intelligence
(category • intent • sentiment • urgency • confidence)
      ↓
Deterministic Business Guardrail
(policy escalation • risky keywords • low confidence)
      ↓
Automated Reply OR Human Escalation
      ↓
SQLite Observability Log → Streamlit Analytics
    """.strip(), language=None)

    st.subheader("Why this is stronger than a basic chatbot")
    st.markdown("""
- **Baseline + advanced comparison:** TF-IDF remains as a measurable lexical baseline.
- **Semantic RAG:** Gemini embeddings can retrieve paraphrases that do not share the same keywords.
- **Hybrid scoring:** lexical and semantic evidence are combined instead of relying on one signal.
- **Policy-aware safety:** escalation labels from the dataset are enforced after LLM generation.
- **Grounded answers:** the model is instructed to use only retrieved approved policy content.
- **Observability:** confidence, retrieval mode, source policy, and ticket intelligence are logged.
- **Reproducible evaluation:** a separate held-out query set measures Top-1 and Top-3 retrieval accuracy.
    """)

    st.warning(
        "Prototype only: the dataset is synthetic. A production system would require real approved company documents, "
        "PII controls, authentication, audit logging, prompt-injection defenses, monitoring, and CRM/ticketing integration."
    )
