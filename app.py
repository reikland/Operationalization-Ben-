import asyncio
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# =============================================================================
# 0) Avoid shadowing installed package:
#    - Put the "relative file expected upstream" into a cache directory
#    - chdir into that cache directory so the relative open() works
# =============================================================================

APP_DIR = Path(__file__).resolve().parent

mac_cache_root = Path.home() / "Library" / "Caches"
cache_root = mac_cache_root if mac_cache_root.exists() else (Path.home() / ".cache")
CACHE_DIR = cache_root / "forecasting-tools-streamlit-app"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(CACHE_DIR)

EXAMPLES_REL = Path(
    "forecasting_tools/agents_and_tools/question_generators/q3_q4_quarterly_questions.json"
)
EXAMPLES_PATH = Path.cwd() / EXAMPLES_REL


def ensure_question_examples_file() -> Path:
    if EXAMPLES_PATH.exists():
        return EXAMPLES_PATH

    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        import importlib.resources as resources

        pkg_file = (
            resources.files("forecasting_tools.agents_and_tools.question_generators")
            .joinpath(EXAMPLES_REL.name)
        )
        EXAMPLES_PATH.write_text(pkg_file.read_text(encoding="utf-8"), encoding="utf-8")
        return EXAMPLES_PATH
    except Exception:
        fallback_examples = [
            {
                "question_text": "Will the fallback example file load correctly?",
                "resolution_criteria": "The app reads the local JSON without errors.",
                "resolution_date": "2026-12-31",
                "extra_context": "Local stub provided when upstream data is unavailable.",
            }
        ]
        EXAMPLES_PATH.write_text(json.dumps(fallback_examples, indent=2), encoding="utf-8")
        return EXAMPLES_PATH


ensure_question_examples_file()

# =============================================================================
# 1) Import forecasting_tools after patching examples file
# =============================================================================
from forecasting_tools.agents_and_tools.question_generators.question_decomposer import (  # noqa: E402
    DecompositionResult,
    QuestionDecomposer,
)
from forecasting_tools.agents_and_tools.question_generators.question_operationalizer import (  # noqa: E402
    QuestionOperationalizer,
)

# =============================================================================
# Helpers
# =============================================================================


def clean_indents(s: str) -> str:
    return textwrap.dedent(s).strip()


def run_async(coro):
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


def to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        return to_dict(obj.model_dump())
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return to_dict(obj.dict())
    return str(obj)


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


def configure_env(provider: str, api_key: str, asknews_key: str, perplexity_key: str) -> None:
    for k in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
    ]:
        os.environ.pop(k, None)

    if api_key:
        if provider == "OpenRouter":
            base = "https://openrouter.ai/api/v1"
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENROUTER_API_KEY"] = api_key
            os.environ["OPENAI_BASE_URL"] = base
            os.environ["OPENAI_API_BASE"] = base
            os.environ["OPENROUTER_BASE_URL"] = base
        else:
            os.environ["OPENAI_API_KEY"] = api_key

    if asknews_key:
        os.environ["ASKNEWS_API_KEY"] = asknews_key
    if perplexity_key:
        os.environ["PERPLEXITY_API_KEY"] = perplexity_key
        os.environ["PPLX_API_KEY"] = perplexity_key


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Topics CSV → proto-questions → operationalize", layout="wide")
st.title("Topics CSV → proto-questions → operationalized questions (forecasting-tools)")

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
        value="openrouter/perplexity/sonar" if mode == "fast" else "openrouter/google/gemini-2.5-pro-preview",
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

if provider == "OpenAI" and (llm_api_key or "").startswith("sk-or-"):
    st.sidebar.error("Provider=OpenAI mais la clé ressemble à une clé OpenRouter (sk-or-v1...).")
    st.stop()

configure_env(provider, llm_api_key, asknews_key, perplexity_key)

with st.sidebar:
    st.divider()
    st.caption(f"Script dir: {APP_DIR}")
    st.caption(f"CWD (cache): {Path.cwd()}")
    st.caption(f"Examples JSON exists: {EXAMPLES_PATH.exists()}")
    st.caption(f"OPENROUTER_API_KEY set: {bool(os.environ.get('OPENROUTER_API_KEY'))}")
    st.caption(f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")
    st.caption(f"OPENAI_API_BASE: {os.environ.get('OPENAI_API_BASE')}")
    st.caption(f"OPENAI_BASE_URL: {os.environ.get('OPENAI_BASE_URL')}")

# =============================================================================
# CSV input
# =============================================================================

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

# =============================================================================
# Pipeline
# =============================================================================


async def run_pipeline() -> pd.DataFrame:
    topics_block = "\n".join([f"- {t}" for t in topics[:300]])
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
