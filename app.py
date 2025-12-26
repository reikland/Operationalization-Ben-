import asyncio
import json
import os
import re
import textwrap
from pathlib import Path
import os
import re
import textwrap
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (
    DecompositionResult,
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (
    QuestionOperationalizer,
)


def clean_indents(s: str) -> str:
    return textwrap.dedent(s).strip()


QUESTION_EXAMPLES_PATH = (
    Path(__file__).parent
    / "forecasting_tools/agents_and_tools/question_generators/q3_q4_quarterly_questions.json"
)


def ensure_question_examples_file() -> Path:
    """Ensure the operationalizer's example JSON file exists locally.

    The upstream package expects to read ``q3_q4_quarterly_questions.json`` from a
    relative path. When the package is installed without bundled data files, this
    fails. We try to copy the file from the installed package; if that is not
    available (e.g., minimal installation), we fall back to a small, local sample
    so the app can still start.
    """

    if QUESTION_EXAMPLES_PATH.exists():
        return QUESTION_EXAMPLES_PATH

    QUESTION_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        import importlib.resources as resources

        package_resource = resources.files(
            "forecasting_tools.agents_and_tools.question_generators"
        ).joinpath(QUESTION_EXAMPLES_PATH.name)
        with package_resource.open("r", encoding="utf-8") as src:
            QUESTION_EXAMPLES_PATH.write_text(src.read(), encoding="utf-8")
            return QUESTION_EXAMPLES_PATH
    except Exception:
        # Fallback minimal content to keep the operationalizer working even when
        # the upstream data file is missing. The structure mirrors the expected
        # list-of-dicts shape used by DataOrganizer.
        fallback_examples = [
            {
                "question_text": "Will the fallback example file load correctly?",
                "resolution_criteria": "The app reads the local JSON without errors.",
                "resolution_date": "2024-12-31",
                "extra_context": "Local stub provided when upstream data is unavailable.",
            }
        ]
        QUESTION_EXAMPLES_PATH.write_text(
            json.dumps(fallback_examples, indent=2), encoding="utf-8"
        )
        return QUESTION_EXAMPLES_PATH


# Make sure the quarterly question examples are available before any downstream
# imports try to read them. This protects against environments where
# ``forecasting_tools`` is installed without bundled data files.
ensure_question_examples_file()


# -----------------------------
# Async helper
# -----------------------------
def run_async(coro):
    """Run an async coroutine from Streamlit safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# -----------------------------
# Topic parsing helpers
# -----------------------------
_SPLIT_RE = re.compile(r"[;\n,\|]+")


def parse_topics_cell(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        raw = x
    else:
        s = str(x).strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                import json

                j = json.loads(s)
                raw = j if isinstance(j, list) else _SPLIT_RE.split(s)
            except Exception:
                raw = _SPLIT_RE.split(s)
        else:
            raw = _SPLIT_RE.split(s)

    out: List[str] = []
    for t in raw:
        tt = str(t).strip()
        if tt:
            out.append(tt)
    return out


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# -----------------------------
# Convert Pydantic objects to dict safely
# -----------------------------
def to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    # pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return to_dict(obj.model_dump())
    # pydantic v1
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return to_dict(obj.dict())
    return str(obj)


# -----------------------------
# Build research pack for operationalizer
# -----------------------------
def build_related_research(decomp: DecompositionResult, q_obj: Any) -> str:
    qd = to_dict(q_obj)
    return clean_indents(
        f"""
        Decomposer general research & approach:
        {decomp.general_research_and_approach}

        Proto-question (from decomposer):
        - Title: {qd.get("question_or_idea_text")}
        - Resolution process: {qd.get("resolution_process")}
        - Expected resolution date: {qd.get("expected_resolution_date")}
        - Background information: {qd.get("background_information")}
        - Past resolution / base rate: {qd.get("past_resolution")}
        - Other information: {qd.get("other_information")}
        """
    ).strip()


# -----------------------------
# Env vars (robuste pour LiteLLM/OpenRouter)
# -----------------------------
def uses_openrouter(*models: str) -> bool:
    return any((m or "").strip().lower().startswith("openrouter/") for m in models)


def configure_env(
    provider: str,
    api_key: str,
    decomposer_model: str,
    operationalizer_model: str,
    asknews_key: str,
    perplexity_key: str,
) -> None:
    if api_key:
        # Certains chemins lisent OPENAI_API_KEY même si provider=openrouter
        os.environ["OPENAI_API_KEY"] = api_key

        if provider == "OpenRouter" or uses_openrouter(decomposer_model, operationalizer_model):
            # LiteLLM (provider openrouter) attend généralement OPENROUTER_API_KEY
            os.environ["OPENROUTER_API_KEY"] = api_key

            # Optionnel mais pratique
            os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
            os.environ["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENROUTER_BASE_URL", None)
            os.environ.pop("OPENAI_BASE_URL", None)

    if asknews_key:
        os.environ["ASKNEWS_API_KEY"] = asknews_key

    if perplexity_key:
        os.environ["PERPLEXITY_API_KEY"] = perplexity_key
        os.environ["PPLX_API_KEY"] = perplexity_key


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Topics CSV → 20 proto-questions → operationalize",
    layout="wide",
)
st.title("Topics CSV → 20 proto-questions → operationalized questions (forecasting-tools)")

with st.sidebar:
    st.header("LLM provider")

    provider = st.selectbox("Provider (OpenAI-compatible)", ["OpenRouter", "OpenAI"], index=0)
    llm_api_key = st.text_input("API key", type="password")

    st.divider()
    st.caption("Optional keys for deep-mode tools:")
    asknews_key = st.text_input("ASKNEWS_API_KEY (optional)", type="password")
    perplexity_key = st.text_input("PERPLEXITY_API_KEY (optional)", type="password")

    st.divider()
    mode = st.selectbox("Decomposer mode", ["fast", "deep"], index=0)
    total_questions = st.number_input("Total proto-questions", min_value=1, max_value=100, value=20, step=1)

    st.divider()
    decomposer_model = st.text_input(
        "Decomposer model",
        value="openrouter/google/gemini-2.5-pro-preview" if mode == "deep" else "openrouter/perplexity/sonar",
    )
    operationalizer_model = st.text_input(
        "Operationalizer model",
        value="openrouter/perplexity/sonar-reasoning-pro",
    )

    additional_context = st.text_area(
        "Additional context (optional)",
        value="Generate high-VOI, resolvable forecasting questions aligned with the provided topics.",
        height=120,
    )

# Configure env (fix for OpenRouter auth)
configure_env(
    provider=provider,
    api_key=llm_api_key,
    decomposer_model=decomposer_model,
    operationalizer_model=operationalizer_model,
    asknews_key=asknews_key,
    perplexity_key=perplexity_key,
)

with st.sidebar:
    st.divider()
    st.caption(f"OPENROUTER_API_KEY set: {bool(os.environ.get('OPENROUTER_API_KEY'))}")
    st.caption(f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    st.caption(f"OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL')}")


# -----------------------------
# CSV input
# -----------------------------
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if not uploaded:
    st.info("Charge un CSV avec une colonne contenant des topics (ex: 'topics').")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, sep=";")

st.subheader("CSV preview")
st.dataframe(df.head(30), use_container_width=True)

topic_col = st.selectbox(
    "Topic column",
    options=list(df.columns),
    index=(list(df.columns).index("topics") if "topics" in df.columns else 0),
)

all_topics: List[str] = []
for cell in df[topic_col].tolist():
    all_topics.extend(parse_topics_cell(cell))
topics = unique_preserve_order(all_topics)

st.write(f"Topics uniques détectés: **{len(topics)}**")
with st.expander("Voir les topics"):
    st.write(topics)

if not llm_api_key:
    st.warning("Renseigne une API key dans la sidebar pour exécuter la génération.")
    st.stop()

run_btn = st.button("Generate proto-questions and operationalize", type="primary")


# -----------------------------
# Pipeline
# -----------------------------
async def run_pipeline() -> pd.DataFrame:
    topics_block = "\n".join([f"- {t}" for t in topics[:300]])  # guardrail
    fuzzy = clean_indents(
        f"""
        Please generate forecasting proto-questions grounded in the following topics (from a CSV).

        Topics:
        {topics_block}
        """
    ).strip()

    decomposer = QuestionDecomposer()

    if mode == "deep":
        decomp = await decomposer.decompose_into_questions_deep(
            fuzzy_topic_or_question=fuzzy,
            related_research=None,
            additional_context=additional_context,
            number_of_questions=int(total_questions),
            model=decomposer_model,
        )
    else:
        decomp = await decomposer.decompose_into_questions_fast(
            fuzzy_topic_or_question=fuzzy,
            related_research=None,
            additional_context=additional_context,
            number_of_questions=int(total_questions),
            model=decomposer_model,
        )

    ensure_question_examples_file()
    operationalizer = QuestionOperationalizer(model=operationalizer_model)

    sem = asyncio.Semaphore(4)

    async def operationalize_one(q_obj):
        async with sem:
            title = q_obj.question_or_idea_text
            related = build_related_research(decomp, q_obj)
            sq = await operationalizer.operationalize_question(
                question_title=title,
                related_research=related,
                additional_context=additional_context,
            )
            return q_obj, sq

    pairs = await asyncio.gather(*[operationalize_one(q) for q in decomp.decomposed_questions])

    rows: List[Dict[str, Any]] = []
    for q_obj, sq in pairs:
        qd = to_dict(q_obj)
        sd = to_dict(sq)

        row: Dict[str, Any] = {
            "proto_question_title": qd.get("question_or_idea_text"),
            "decomp_resolution_process": qd.get("resolution_process"),
            "decomp_expected_resolution_date": qd.get("expected_resolution_date"),
            "decomp_background_information": qd.get("background_information"),
            "decomp_past_resolution": qd.get("past_resolution"),
            "decomp_other_information": qd.get("other_information"),
        }

        if isinstance(sd, dict):
            for k, v in sd.items():
                row[f"op_{k}"] = v
        else:
            row["op_question"] = sd

        rows.append(row)

    return pd.DataFrame(rows)


if run_btn:
    if not topics:
        st.error("Aucun topic détecté.")
        st.stop()

    with st.spinner("Running decomposer + operationalizer..."):
        out_df = run_async(run_pipeline())

    st.subheader("Results")
    st.dataframe(out_df, use_container_width=True)

    st.download_button(
        "Download results as CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="operationalized_questions.csv",
        mime="text/csv",
    )
