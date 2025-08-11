Corporate Agent – ADGM Legal Assistant

This project is a small, focused demo of an AI-assisted legal reviewer for Abu Dhabi Global Market (ADGM) documentation. It accepts .docx files, checks them against ADGM-oriented checklists, flags common issues, inserts contextual comments in the document, and produces a concise JSON summary. Retrieval-Augmented Generation (RAG) is used to ground suggestions and citations in official ADGM sources when an API key is available.

What you can do with it
- Upload one or more .docx files (AoA, MoA, resolutions, forms, registers, etc.).
- Automatically detect the likely process (e.g., Company Incorporation) and the uploaded document types.
- Compare against a checklist to spot missing required documents.
- Detect typical red flags (jurisdiction, ambiguity, missing signatories, non-ADGM references).
- Get a reviewed .docx with highlighted text and an appended “Review Comments” section.
- View and download a structured JSON summary.
- Build a local reference index from official ADGM links for better, grounded suggestions.

Project layout
```
app.py                      # Streamlit UI (upload, analysis, downloads)
document_checker.py         # Core logic: RAG, doc type/process detection, checks, summary
doc_utils.py                # DOCX text extraction and review annotation
adgm_reference/
  official_links.json       # Curated list of official ADGM resources
  reference_docs/           # Cached PDFs / DOCX / text for RAG
outputs/                    # Reviewed .docx and JSON summaries
sample_docs/                # Put your example .docx inputs here
requirements.txt            # Python dependencies
.env                        # Environment variables (e.g., OPENAI_API_KEY)
```

Requirements
- Python 3.10+ (Windows, macOS, or Linux)
- Internet connection if you want to fetch official references or use the LLM
- Optional: OpenAI API key to enable RAG-backed suggestions and citations

Need (activate API key)
- If you are using DeepSeek (OpenAI-compatible API), create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
LLM_CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=local-sentence-transformers
```
- Then restart the app:
```
streamlit run app.py
```

Quick start
1) Create a virtual environment (recommended) and install dependencies
```
pip install -r requirements.txt
```

2) Add your OpenAI key (optional, but recommended for better output)
Create a file named .env in the repository root:
```
OPENAI_API_KEY=your_key_here
```
Without a key, the app still runs with basic heuristics and generic suggestions.

DeepSeek setup (alternative)
```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
LLM_CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=local-sentence-transformers
```
Restart Streamlit after updating `.env`.

3) Launch the app
```
streamlit run app.py
```
Streamlit will show a local link in the terminal. Open it in your browser.

Using the app
1) (Optional) Fetch official references
   - Click “Fetch official ADGM links” to download a curated set of official ADGM PDFs/DOCX into adgm_reference/reference_docs.
   - Then click “Rebuild RAG Index” to index those files.

2) Upload documents
   - Use the uploader to add one or more .docx files.
   - Click “Analyze Documents”.

3) Review results
   - Checklist & Summary: shows the detected process, required vs present documents, and counts.
   - Reviewed Files: for each file, a JSON-like list of issues and a button to download the reviewed .docx.
   - The reviewed files and the summary JSON are saved under outputs/.

How it works
1) Document intake
   - Each .docx is parsed for text (python-docx).
   - A simple keyword-based classifier maps content to a document type (e.g., “Articles of Association”).

2) Process detection & checklist
   - Based on the recognized document types, the app infers a process (defaulting to Company Incorporation if unclear).
   - It compares present document types to a required list and reports missing items.

3) Red flag checks
   - Jurisdiction: flags documents that discuss jurisdiction but don’t mention ADGM.
   - Ambiguity: flags soft language (“may”, “best efforts”, etc.).
   - Signatures: flags documents that lack a signatory section.
   - Incorrect references: flags mentions of non-ADGM courts.
   - A few doc-specific checks exist (e.g., AoA missing core provisions).

4) RAG and suggestions
   - If an OpenAI key is present, the app builds an embedding index over files in adgm_reference/reference_docs.
   - When a red flag is found, the app retrieves relevant snippets and requests a short, practical suggestion with a citation.
   - Without a key, it provides generic suggestions so you can still complete the workflow.

5) Inline comments in Word
   - python-docx does not provide Word “balloon” comments. Instead, the app highlights the first paragraph that matches the flagged pattern and appends a “Review Comments (Corporate Agent)” section with details and citations.

Output formats
- Reviewed Word files: Saved as <original>__reviewed.docx in outputs/.
- Summary JSON: Saved as outputs/analysis_summary.json and also available to download in the UI. A simple example:
```
{
  "process": "Company Incorporation",
  "documents_uploaded": 4,
  "required_documents": 8,
  "documents_present_types": ["Articles of Association", "Memorandum of Association"],
  "missing_documents": ["Register of Members and Directors", "UBO Declaration Form", "Board Resolution", "Shareholder Resolution", "Incorporation Application Form", "Change of Registered Address Notice"],
  "issues_found_total": 3
}
```

Extending the assistant
- Checklists: Update REQUIRED_DOCUMENTS_BY_PROCESS in document_checker.py to cover more processes or tailor for your use case.
- Document typing: Expand DOC_TYPE_KEYWORDS to support additional templates and naming schemes.
- Red flags: Add new regex checks in run_red_flag_checks for domain-specific issues.
- References: Add more official files (PDF/DOCX/TXT) into adgm_reference/reference_docs and rebuild the index.

Operational notes
- The “Fetch official ADGM links” feature writes downloaded files to adgm_reference/reference_docs. If a link changes or is restricted, a small “.url.txt” pointer file is saved so you can still see the source.
- If indexing runs without an API key, embeddings default to zeros and retrieval falls back to simple heuristics. The app remains functional, just less accurate.
- Some old or heavily customized .docx templates might not have standard styles. The app avoids depending on missing styles; the review header falls back to a bold paragraph if “Heading 1” is unavailable.

Troubleshooting
- Streamlit is running but no page opens: copy the Local URL from the terminal into your browser.
- Large or image-only PDFs in references: text extraction depends on embedded text; scanned PDFs without OCR will contribute little to RAG.
- Comments not visible where expected: only the first matching paragraph per issue is highlighted; details are always appended in the review section.
- Networking or blocked downloads: the fetch step may create a .url.txt pointer instead of the original file; you can download it manually and place it in adgm_reference/reference_docs.

Deliverables checklist (for internship submission)
- GitHub repository (or zip) including this codebase.
- One sample .docx before/after review in outputs/.
- The generated summary JSON in outputs/.
- A quick screenshot or short screen recording of the app running.

Samples and demo
- Generate a quick before/after sample automatically:
  ```
  python scripts/make_sample_docs.py
  ```
  This creates `sample_docs/example_before.docx`, runs analysis, and writes the reviewed file to `outputs/example_before__reviewed.docx`.

- Suggested demo outline:
  1) Show `.env` content (keys redacted) and click “Fetch official ADGM links” → “Rebuild RAG Index”.
  2) Upload an example `.docx`, click “Analyze Documents”.
  3) Show “Checklist & Summary”, per-file issues, and download the reviewed `.docx`.
  4) Open the reviewed file to highlight inline comments and the appended “Review Comments” section.
  5) Optionally click “Test LLM connectivity” to show routing to DeepSeek/OpenAI.

License and data
- This is a demonstration tool. It does not constitute legal advice.
- Verify all citations and outputs against the current, official ADGM sources.

Contact: bsvishwananth@gmail.com

