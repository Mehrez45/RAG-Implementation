import streamlit as st

from src.app.runners import RunResult, build_runner
from src.app.runtime import build_runtime


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="S",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading local model and vector index...")
def get_runtime():
    return build_runtime()


def render_result(result: RunResult) -> None:
    st.subheader("Answer")
    st.write(result.answer)

    col1, col2, col3 = st.columns(3)
    col1.metric("Mode", result.mode.title())
    col2.metric("Route", result.route or "unknown")
    col3.metric("Revisions", result.revision_count)

    if result.route_reason:
        st.caption(f"Route reason: {result.route_reason}")

    if result.review_feedback:
        st.subheader("Review Feedback")
        st.write(result.review_feedback)

    if result.failure_reason:
        st.error(result.failure_reason)


st.title("RAG Assistant")
st.caption("Run the existing vanilla or agentic pipeline behind a simple Streamlit UI.")

with st.sidebar:
    st.header("Run Mode")
    mode = st.radio(
        "Choose a pipeline",
        options=["agentic", "vanilla"],
        format_func=str.title,
    )
    st.markdown(
        "This app loads the local GGUF model and FAISS index from the repo."
    )

with st.form("query_form"):
    query = st.text_area(
        "Ask a question",
        placeholder="What does the document say about ...?",
        height=140,
    )
    submitted = st.form_submit_button("Run Query")

if submitted:
    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Enter a question before running the pipeline.")
    else:
        try:
            runtime = get_runtime()
            runner = build_runner(mode, runtime)
            with st.spinner(f"Running the {mode} pipeline..."):
                result = runner.run(cleaned_query)
        except Exception as exc:
            st.exception(exc)
        else:
            render_result(result)
