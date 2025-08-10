import io
import os
import re
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
try:
    # Optional: available in openai>=1.x
    from openai import APIStatusError  # type: ignore
except Exception:  # pragma: no cover
    APIStatusError = Exception  # fallback
from typing import cast

_LOCAL_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
_USE_LOCAL_EMBEDDINGS = _LOCAL_EMBEDDING_MODEL_NAME.lower().startswith("local-")

from doc_utils import (
    extract_docx_text,
    annotate_docx_with_issues,
)


# -----------------------------
# Checklists and heuristics
# -----------------------------

REQUIRED_DOCUMENTS_BY_PROCESS: Dict[str, List[str]] = {
    "Company Incorporation": [
        "Articles of Association",
        "Memorandum of Association",
        "Board Resolution",
        "Shareholder Resolution",
        "Incorporation Application Form",
        "UBO Declaration Form",
        "Register of Members and Directors",
        "Change of Registered Address Notice",
    ],
    "SPV Incorporation": [
        "Articles of Association",
        "Memorandum of Association",
        "Board Resolution",
        "Shareholder Resolution",
        "Incorporation Application Form",
        "UBO Declaration Form",
        "Register of Members and Directors",
    ],
    "LLC Incorporation": [
        "Articles of Association",
        "Memorandum of Association",
        "Board Resolution",
        "Shareholder Resolution",
        "Incorporation Application Form",
        "UBO Declaration Form",
        "Register of Members and Directors",
        "Change of Registered Address Notice",
    ],
    # Extend with other processes as needed
    "Licensing": [
        "License Application Form",
        "Business Plan",
        "Financial Statements",
        "Risk & Compliance Policy",
    ],
}


DOC_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "Articles of Association": ["articles of association", "aoa"],
    "Memorandum of Association": ["memorandum of association", "memorandum", "moa", "mou"],
    "Board Resolution": ["board resolution"],
    "Shareholder Resolution": ["shareholder resolution", "shareholders' resolution"],
    "Incorporation Application Form": ["incorporation application", "application for incorporation"],
    "UBO Declaration Form": ["ubo", "ultimate beneficial owner"],
    "Register of Members and Directors": ["register of members", "register of directors"],
    "Change of Registered Address Notice": ["change of registered address", "registered address"],
    # Additional categories
    "Employment Contract": ["employment contract", "employee", "employer"],
    "Data Protection Policy": ["data protection", "privacy", "dpr 2021"],
    "Compliance Policy": ["compliance policy", "risk policy", "anti-money laundering", "aml"],
}


# -----------------------------
# RAG Index
# -----------------------------

EMBEDDINGS_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini")


def _load_envs_once() -> None:
    # Load from both script directory and CWD
    for p in [Path(__file__).parent / ".env", Path.cwd() / ".env"]:
        try:
            load_dotenv(dotenv_path=p, override=False)
        except Exception:
            pass


def _get_openai_client() -> Optional[OpenAI]:
    _load_envs_once()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL")
    try:
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
    except Exception:
        return None


@dataclass
class RAGIndex:
    documents: List[str]
    embeddings: np.ndarray
    doc_ids: List[str]
    model_name: str = EMBEDDINGS_MODEL

    @staticmethod
    def _embed_texts(texts: List[str]) -> np.ndarray:
        load_dotenv()
        if _USE_LOCAL_EMBEDDINGS:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                m = SentenceTransformer(model_name)
                vecs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                return vecs.astype(np.float32)
            except Exception:
                return np.zeros((len(texts), 384), dtype=np.float32)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Return zero embeddings to keep code functioning without key
            return np.zeros((len(texts), 1536), dtype=np.float32)
        client = _get_openai_client()
        if client is None:
            return np.zeros((len(texts), 1536), dtype=np.float32)
        # Batch in reasonable sizes
        batch_size = 64
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = client.embeddings.create(model=RAGIndex.model_name, input=batch)
                all_embeddings.extend([d.embedding for d in resp.data])
            except Exception:
                # If embeddings endpoint unsupported (e.g., non-OpenAI provider), fallback zeros
                return np.zeros((len(texts), 1536), dtype=np.float32)
        return np.array(all_embeddings, dtype=np.float32)

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for p in paragraphs:
            if current_len + len(p) + 1 > max_chars:
                if current:
                    chunks.append("\n".join(current))
                # start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = chunks[-1][-overlap:]
                    current = [overlap_text, p]
                    current_len = len(overlap_text) + len(p) + 1
                else:
                    current = [p]
                    current_len = len(p)
            else:
                current.append(p)
                current_len += len(p) + 1
        if current:
            chunks.append("\n".join(current))
        return chunks

    @classmethod
    def from_reference_folder(cls, folder_path: str, index_path: str) -> "RAGIndex":
        from pypdf import PdfReader

        texts: List[str] = []
        doc_ids: List[str] = []

        from docx import Document as DocxDocument
        for root, _, files in os.walk(folder_path):
            for fname in files:
                fpath = os.path.join(root, fname)
                base_id = os.path.relpath(fpath, folder_path)
                try:
                    if fname.lower().endswith(".pdf"):
                        reader = PdfReader(fpath)
                        txt = "\n".join(page.extract_text() or "" for page in reader.pages)
                    elif fname.lower().endswith(".docx"):
                        d = DocxDocument(fpath)
                        txt = "\n".join(p.text for p in d.paragraphs)
                    elif fname.lower().endswith((".txt", ".md")):
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            txt = f.read()
                    else:
                        # skip other formats
                        continue
                    for chunk in cls._chunk_text(txt):
                        texts.append(chunk)
                        doc_ids.append(base_id)
                except Exception:
                    continue

        if not texts:
            # Ensure at least one fallback doc
            texts = [
                "ADGM Companies Regulations 2020 overview. Jurisdiction for corporate disputes is ADGM Courts. Incorporation requires AoA, MoA, Board and Shareholder resolutions, UBO declaration, application form, and registers."
            ]
            doc_ids = ["fallback.txt"]

        embeddings = cls._embed_texts(texts)
        index = cls(documents=texts, embeddings=embeddings, doc_ids=doc_ids)
        with open(index_path, "wb") as f:
            pickle.dump(index, f)
        return index

    @classmethod
    def load(cls, index_path: str) -> "RAGIndex":
        with open(index_path, "rb") as f:
            return pickle.load(f)

    def search(self, query: str, k: int = 4) -> List[Tuple[str, str, float]]:
        q_emb = self._embed_texts([query])[0]
        if np.allclose(q_emb, 0.0):
            # No key scenario, return top docs arbitrarily
            results = []
            for i, doc in enumerate(self.documents[:k]):
                results.append((self.doc_ids[i], doc, 0.0))
            return results
        # cosine similarity
        A = self.embeddings
        B = q_emb
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B) + 1e-8)
        sims = A_norm @ B_norm
        idxs = np.argsort(-sims)[:k]
        return [(self.doc_ids[i], self.documents[i], float(sims[i])) for i in idxs]


# -----------------------------
# LLM helper
# -----------------------------

def llm_suggest_fix(query: str, retrieved: List[Tuple[str, str, float]]) -> Tuple[str, str]:
    """Return (suggestion, citation) using OpenAI if available; otherwise heuristic text."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        # Fallback suggestion
        return (
            "Specify ADGM Courts as governing jurisdiction and ensure required ADGM documents are included.",
            "ADGM Companies Regulations 2020 (general guidance)"
        )

    client = _get_openai_client()
    if client is None:
        return (
            "Specify ADGM Courts as governing jurisdiction and ensure required ADGM documents are included.",
            "ADGM Companies Regulations 2020 (general guidance)"
        )
    context_blocks = []
    for doc_id, text, score in retrieved:
        context_blocks.append(f"Source: {doc_id}\n{(text[:1200] + '...') if len(text) > 1200 else text}")
    system = (
        "You are an assistant ensuring compliance with ADGM Companies Regulations. "
        "Use the provided context to propose a concise fix and cite the most relevant source and, if possible, specific article."
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Context:\n" + "\n\n".join(context_blocks) +
                "\n\nIssue: " + query +
                "\n\nRespond with: Suggestion (1-2 sentences). Then a short citation like 'Per ADGM ... Art. X'."
            ),
        },
    ]
    try:
        resp = client.chat.completions.create(model=LLM_CHAT_MODEL, messages=messages, temperature=0.2)
        text = resp.choices[0].message.content.strip()
    except APIStatusError as e:  # e.g., 402 Insufficient Balance
        return (
            "Specify ADGM Courts as governing jurisdiction and ensure required ADGM documents are included.",
            "ADGM Companies Regulations 2020 (general guidance)"
        )
    except Exception:
        return (
            "Specify ADGM Courts as governing jurisdiction and ensure required ADGM documents are included.",
            "ADGM Companies Regulations 2020 (general guidance)"
        )
    # naive split to suggestion and citation
    if "Per ADGM" in text:
        parts = text.split("Per ADGM", 1)
        suggestion = parts[0].strip()
        citation = "Per ADGM" + parts[1].strip()
        return suggestion, citation
    return text, "ADGM reference"


PROMPT_TEMPLATE = (
    "You are an expert legal AI assistant specializing in Abu Dhabi Global Market (ADGM) corporate law.\n\n"
    "Your task is to review the given corporate legal document and perform the following:\n\n"
    "1. Identify the document type from these categories:\n"
    "   - Articles of Association (AoA)\n"
    "   - Memorandum of Association (MoA)\n"
    "   - Board Resolution\n"
    "   - Shareholder Resolution\n"
    "   - Incorporation Application Form\n"
    "   - UBO Declaration Form\n"
    "   - Register of Members and Directors\n"
    "   - Change of Registered Address Notice\n\n"
    "2. Check if the document complies with ADGM legal requirements for company incorporation.\n\n"
    "3. Detect and list any red flags or issues such as:\n"
    "   - Missing or invalid clauses\n"
    "   - Incorrect jurisdiction references (e.g., referencing UAE federal courts instead of ADGM)\n"
    "   - Ambiguous or non-binding language\n"
    "   - Missing signatories or improper formatting\n"
    "   - Non-compliance with ADGM templates\n\n"
    "4. Suggest contextual inline comments referencing specific ADGM laws or regulations (e.g., \"Per ADGM Companies Regulations 2020, Article 6...\").\n\n"
    "5. Summarize the compliance status and indicate missing documents based on the incorporation checklist if multiple documents are provided.\n\n"
    "6. Return your analysis as a JSON object with these keys:\n"
    "   - process: e.g., \"Company Incorporation\"\n"
    "   - document_type: identified document type\n"
    "   - compliance_status: \"Compliant\" or \"Non-Compliant\"\n"
    "   - issues_found: integer count\n"
    "   - missing_documents: list of any required documents missing (if applicable)\n"
    "   - issues: list of detected issues, each with:\n"
    "       * section (e.g., \"Clause 3.1\")\n"
    "       * issue (short description)\n"
    "       * severity (\"High\", \"Medium\", \"Low\")\n"
    "       * suggestion (recommended fix)\n"
    "       * citation (relevant ADGM law or regulation)\n\n"
    "Document content:\n\n"
    "\"\"\"\n{document_text}\n\"\"\"\n\n"
    "Provide only the JSON response as your output."
)


def get_current_llm_config() -> Dict[str, Any]:
    """Return current LLM-related configuration for diagnostics."""
    load_dotenv()
    return {
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "chat_model": os.getenv("LLM_CHAT_MODEL", LLM_CHAT_MODEL),
        "embedding_model": os.getenv("EMBEDDING_MODEL", EMBEDDINGS_MODEL),
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
    }


def test_llm_connectivity() -> Dict[str, Any]:
    """Try listing models and doing a tiny chat call. Returns diagnostic info instead of raising."""
    info: Dict[str, Any] = {"ok": False}
    cfg = get_current_llm_config()
    info.update({"config": cfg})
    client = _get_openai_client()
    if client is None:
        info["error"] = "No API client (missing key or invalid base_url)."
        return info
    # Try models.list
    try:
        models = client.models.list()
        info["models_count"] = len(getattr(models, "data", []) or [])
        # Optional: show a few model IDs
        info["models_sample"] = [m.id for m in (models.data or [])[:5]]
    except Exception as e:  # pragma: no cover
        info["models_error"] = str(e)
    # Try a ping chat call
    try:
        resp = client.chat.completions.create(
            model=cfg.get("chat_model") or LLM_CHAT_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=5,
        )
        info["chat_reply"] = resp.choices[0].message.content
        info["ok"] = True
    except Exception as e:  # pragma: no cover
        info["chat_error"] = str(e)
    return info


def llm_analyze_document(document_text: str, rag: Optional[RAGIndex], process_hint: str, missing_documents: List[str]) -> Optional[Dict[str, Any]]:
    client = _get_openai_client()
    if client is None:
        return None

    # Retrieve context from RAG if available
    retrieved = rag.search("ADGM incorporation requirements and document checks") if rag else []
    context_text = "\n\n".join([t for _, t, _ in retrieved])
    # Prepend context to document
    combined_text = ("Context from ADGM references:\n" + context_text + "\n\n" + document_text)[:16000]

    user_prompt = PROMPT_TEMPLATE.format(document_text=combined_text)
    messages = [
        {"role": "system", "content": "You are a legal compliance assistant specialized in ADGM regulations."},
        {"role": "user", "content": user_prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=LLM_CHAT_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1200,
        )
        content = resp.choices[0].message.content
        # try to parse JSON
        try:
            obj = json.loads(content)
            # augment with hints if missing
            if "process" not in obj:
                obj["process"] = process_hint
            if "missing_documents" not in obj and missing_documents:
                obj["missing_documents"] = missing_documents
            return obj
        except Exception:
            return {"raw": content}
    except Exception:
        return None


# -----------------------------
# Document analysis
# -----------------------------

def identify_document_type(text: str, filename: Optional[str] = None) -> str:
    t = text.lower()
    # 1) content-based keywords
    for doc_type, kws in DOC_TYPE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return doc_type
    # 2) filename-based hints
    if filename:
        fn = os.path.splitext(os.path.basename(filename))[0].lower()
        for doc_type, kws in DOC_TYPE_KEYWORDS.items():
            if any(kw in fn for kw in kws):
                return doc_type
        # common shorthands
        if "aoa" in fn:
            return "Articles of Association"
        if "moa" in fn or "mou" in fn:
            return "Memorandum of Association"
        if "board" in fn and "resolution" in fn:
            return "Board Resolution"
        if "shareholder" in fn and "resolution" in fn:
            return "Shareholder Resolution"
    # 3) heuristic fallbacks
    if "articles" in t and "association" in t:
        return "Articles of Association"
    if "memorandum" in t:
        return "Memorandum of Association"
    return "Unknown"


def detect_process(doc_types: List[str]) -> str:
    # Simple heuristic: if any formation document present -> Incorporation
    formation_docs = set(REQUIRED_DOCUMENTS_BY_PROCESS["Company Incorporation"])
    if any(dt in formation_docs for dt in doc_types):
        return "Company Incorporation"
    if any("license" in dt.lower() for dt in doc_types):
        return "Licensing"
    return "Company Incorporation"


def run_red_flag_checks(text: str, doc_type: str, rag: Optional[RAGIndex]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    # 1) Jurisdiction check
    if re.search(r"jurisdiction|governing law|courts", text, flags=re.I):
        if not re.search(r"ADGM|Abu Dhabi Global Market", text, flags=re.I):
            retrieved = rag.search("ADGM jurisdiction clause requirement") if rag else []
            suggestion, citation = llm_suggest_fix(
                "Jurisdiction clause should specify ADGM Courts.", retrieved
            )
            issues.append({
                "section": "Jurisdiction",
                "issue": "Jurisdiction clause does not specify ADGM",
                "severity": "High",
                "suggestion": suggestion,
                "citation": citation,
                "pattern": "jurisdiction",
            })

    # 2) Ambiguous language
    if re.search(r"\bmay\b|best efforts|endeavour to|endeavor to|as appropriate", text, flags=re.I):
        retrieved = rag.search("ADGM enforceable clause clarity") if rag else []
        suggestion, citation = llm_suggest_fix(
            "Ambiguous or non-binding language should be tightened.", retrieved
        )
        issues.append({
            "section": "Clarity",
            "issue": "Ambiguous or non-binding language",
            "severity": "Medium",
            "suggestion": suggestion,
            "citation": citation,
            "pattern": "ambiguous",
        })

    # 3) Missing signatory block
    if not re.search(r"Signed by|Authorised signatory|Authorized signatory|Signature", text, flags=re.I):
        retrieved = rag.search("ADGM signing requirements corporate documents") if rag else []
        suggestion, citation = llm_suggest_fix(
            "Document missing signatory section.", retrieved
        )
        issues.append({
            "section": "Signatures",
            "issue": "Missing signatory section",
            "severity": "High",
            "suggestion": suggestion,
            "citation": citation,
            "pattern": "signature",
        })

    # 4) ADGM template reference
    if re.search(r"Dubai Courts|UAE Federal Courts|onshore UAE", text, flags=re.I):
        retrieved = rag.search("ADGM vs UAE federal jurisdiction distinctions") if rag else []
        suggestion, citation = llm_suggest_fix(
            "Incorrect jurisdiction referenced; should be ADGM.", retrieved
        )
        issues.append({
            "section": "Jurisdiction",
            "issue": "Incorrect jurisdiction reference",
            "severity": "High",
            "suggestion": suggestion,
            "citation": citation,
            "pattern": "uae federal",
        })

    # Optional: doc-type specific checks
    if doc_type == "Articles of Association":
        if not re.search(r"shares|share capital|directors", text, flags=re.I):
            retrieved = rag.search("AoA required provisions ADGM") if rag else []
            suggestion, citation = llm_suggest_fix(
                "AoA should include provisions on share capital and directors.", retrieved
            )
            issues.append({
                "section": "AoA",
                "issue": "Potential missing core provisions (shares/directors)",
                "severity": "Medium",
                "suggestion": suggestion,
                "citation": citation,
                "pattern": "aoa core",
            })

    # 5) Dates and effective date presence
    if not re.search(r"Date:|Dated this|Effective date|Commencement", text, flags=re.I):
        retrieved = rag.search("ADGM document dating requirements") if rag else []
        suggestion, citation = llm_suggest_fix(
            "Document should include clear execution/Effective date.", retrieved
        )
        issues.append({
            "section": "Execution",
            "issue": "Missing execution or effective date",
            "severity": "Low",
            "suggestion": suggestion,
            "citation": citation,
            "pattern": "date",
        })

    # 6) Share capital details for AoA/MoA
    if doc_type in ("Articles of Association", "Memorandum of Association"):
        if not re.search(r"share capital|authorized capital|issued shares|nominal value", text, flags=re.I):
            retrieved = rag.search("ADGM share capital requirements") if rag else []
            suggestion, citation = llm_suggest_fix(
                "Include share capital details (authorized/issued, nominal value).", retrieved
            )
            issues.append({
                "section": "Capital",
                "issue": "Share capital details not found",
                "severity": "Medium",
                "suggestion": suggestion,
                "citation": citation,
                "pattern": "share capital",
            })

    return issues


def analyze_documents(
    uploaded_docs: List[Dict[str, Any]], rag_index: Optional[RAGIndex] = None, run_llm_json: bool = False
) -> Dict[str, Any]:
    # Load or build RAG index if not provided
    if rag_index is None and os.path.exists("adgm_reference/index_openai.pkl"):
        try:
            rag_index = RAGIndex.load("adgm_reference/index_openai.pkl")
        except Exception:
            rag_index = None

    texts: List[str] = []
    file_infos: List[Dict[str, Any]] = []
    for doc in uploaded_docs:
        content = doc["bytes"]
        filename = doc["filename"]
        text = extract_docx_text(io.BytesIO(content))
        texts.append(text)
        doc_type = identify_document_type(text, filename)
        file_infos.append({
            "filename": filename,
            "doc_type": doc_type,
            "text": text,
        })

    doc_types = [fi["doc_type"] for fi in file_infos]
    process = detect_process(doc_types)

    # Checklist verification
    required = REQUIRED_DOCUMENTS_BY_PROCESS.get(process, [])
    present = set(dt for dt in doc_types if dt != "Unknown")
    missing = [d for d in required if d not in present]

    reviewed_files: List[Dict[str, Any]] = []
    total_issues = 0
    for fi in file_infos:
        issues = run_red_flag_checks(fi["text"], fi["doc_type"], rag_index)
        total_issues += len(issues)
        # Annotate docx
        revised_bytes = annotate_docx_with_issues(io.BytesIO(uploaded_docs[file_infos.index(fi)]["bytes"]), issues)
        out_name = os.path.splitext(fi["filename"])[0] + "__reviewed.docx"
        out_path = os.path.join("outputs", out_name)
        with open(out_path, "wb") as f:
            f.write(revised_bytes)
        record: Dict[str, Any] = {
            "filename": fi["filename"],
            "output_filename": out_name,
            "issues": issues,
            "bytes": revised_bytes,
        }
        if run_llm_json:
            llm_json = llm_analyze_document(fi["text"], rag_index, process, missing)
            if llm_json is not None:
                record["llm_json"] = llm_json
                # save per-file llm analysis
                json_name = os.path.splitext(fi["filename"])[0] + "__llm_analysis.json"
                json_path = os.path.join("outputs", json_name)
                try:
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(llm_json, jf, ensure_ascii=False, indent=2)
                    record["llm_json_file"] = json_name
                except Exception:
                    pass
        reviewed_files.append(record)

    summary: Dict[str, Any] = {
        "process": process,
        "documents_uploaded": len(uploaded_docs),
        "required_documents": len(required),
        "documents_present_types": list(present),
        "missing_documents": missing,
        "issues_found_total": total_issues,
    }

    return {
        "summary": summary,
        "reviewed_files": reviewed_files,
    }


