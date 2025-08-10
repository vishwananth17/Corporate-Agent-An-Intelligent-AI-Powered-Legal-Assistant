import io
import os
import json
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from pathlib import Path


from document_checker import (
    RAGIndex,
    analyze_documents,
    REQUIRED_DOCUMENTS_BY_PROCESS,
    test_llm_connectivity,
    get_current_llm_config,
)
from adgm_reference_tools import download_official_references


def ensure_dirs() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("adgm_reference/reference_docs", exist_ok=True)
    os.makedirs("sample_docs", exist_ok=True)


def _load_envs() -> None:
    # Load .env from script directory and current working directory
    env_candidates = [Path(__file__).parent / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        try:
            load_dotenv(dotenv_path=env_path, override=False)
        except Exception:
            pass


def main() -> None:
    _load_envs()
    ensure_dirs()

    st.set_page_config(page_title="Corporate Agent – ADGM Legal Assistant", layout="wide")
    st.title("Corporate Agent – ADGM Legal Assistant")
    st.write(
        "Upload ADGM-related .docx documents. The agent will check completeness, detect red flags, add inline comments, and provide a summary report using RAG for ADGM references."
    )

    openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
    with st.expander("Environment & References", expanded=not openai_key_present):
        st.markdown(
            "- Set `OPENAI_API_KEY` in `.env` to enable RAG and drafting of suggestions.\n"
            "- Place ADGM reference PDFs or text files in `adgm_reference/reference_docs`.\n"
        )
        if not openai_key_present:
            st.warning("OPENAI_API_KEY missing. RAG and suggestions will be limited or disabled.")

    # RAG Index controls
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Rebuild RAG Index"):
            rag = RAGIndex.from_reference_folder(
                folder_path="adgm_reference/reference_docs",
                index_path="adgm_reference/index_openai.pkl",
            )
            st.success(f"Indexed {len(rag.documents)} chunks from reference docs.")

        if st.button("Fetch official ADGM links"):
            saved = download_official_references(
                json_path="adgm_reference/official_links.json",
                dest_folder="adgm_reference/reference_docs",
            )
            st.success(f"Fetched {len(saved)} references. You can now Rebuild RAG Index.")

        if st.button("Test LLM connectivity"):
            cfg = get_current_llm_config()
            st.write("Current LLM config:")
            st.json(cfg)
            diag = test_llm_connectivity()
            if diag.get("ok"):
                st.success("LLM connectivity OK")
            else:
                st.error("LLM connectivity failed")
            st.json(diag)

    # File upload
    uploaded_files = st.file_uploader(
        "Upload one or more .docx files",
        type=["docx"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Analyze Documents"):
        # Load or build index if present
        rag_index = None
        if os.path.exists("adgm_reference/index_openai.pkl"):
            try:
                rag_index = RAGIndex.load("adgm_reference/index_openai.pkl")
            except Exception:
                rag_index = None

        with st.spinner("Analyzing documents..."):
            input_docs: List[Dict[str, Any]] = []
            for f in uploaded_files:
                input_docs.append({
                    "filename": f.name,
                    "bytes": f.read(),
                })

            run_llm_json = st.checkbox("Run full LLM JSON analysis per file (uses API credits)", value=False)
            result = analyze_documents(input_docs, rag_index, run_llm_json=run_llm_json)

        # Display results
        st.subheader("Checklist & Summary")
        st.json(result["summary"])  # structured JSON summary
        # Save JSON to outputs and provide download
        summary_path = os.path.join("outputs", "analysis_summary.json")
        with open(summary_path, "w", encoding="utf-8") as jf:
            json.dump(result["summary"], jf, ensure_ascii=False, indent=2)
        st.download_button(
            label="Download summary JSON",
            data=json.dumps(result["summary"], ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="analysis_summary.json",
            mime="application/json",
        )

        st.subheader("Reviewed Files")
        for reviewed in result["reviewed_files"]:
            st.markdown(f"**{reviewed['filename']}**")
            st.json(reviewed["issues"])  # per-file issues
            if "llm_json" in reviewed:
                st.markdown("LLM JSON analysis:")
                st.json(reviewed["llm_json"])
            st.download_button(
                label="Download reviewed .docx",
                data=reviewed["bytes"],
                file_name=reviewed["output_filename"],
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        st.markdown("---")
        st.markdown("### Required Documents by Process")
        for process, reqs in REQUIRED_DOCUMENTS_BY_PROCESS.items():
            st.markdown(f"**{process}**")
            st.write("- " + "\n- ".join(reqs))


if __name__ == "__main__":
    main()


