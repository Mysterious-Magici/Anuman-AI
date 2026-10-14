# DocuMind AI - Agentic Document Intelligence Platform

Upload PDF, DOCX, TXT, Markdown, CSV, Excel and PowerPoint files, chat about them (Groq), analyze them
(Gemini) and export branded reports (PDF with embedded logo, DOCX, TXT, Markdown, CSV/XLSX).
Built with Python + Streamlit. Free-tier friendly: parsing, retrieval, statistics and exports run locally.

## 1. Prerequisites
- Python 3.10 or newer (check with `python --version`)
- A Gemini API key and a Groq API key (both have free tiers; limits can change)

## 2. Project layout
Put the files exactly as listed in the answer. Your old `config.toml` must live at `.streamlit/config.toml`.
You can delete the old `agent.py` (its prompt now lives in `services/agent_workflow.py`).
Keep `profiler.py` in the project root.

## 3. Create a virtual environment and install
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

## 4. Configure your keys
```bash
cp .env.example .env        # Windows: copy .env.example .env
```
Open `.env` and set `GEMINI_API_KEY`, `GEMINI_MODEL`, `GROQ_API_KEY`, `GROQ_MODEL`.
Your old `GOOGLE_API_KEY` variable also works in place of `GEMINI_API_KEY`.
The app never picks a model for you: use model IDs your accounts support, then press the
**Test Gemini / Test Groq connection** buttons in *Settings & Help*.

## 5. Run
```bash
streamlit run app.py
```

## 6. Optional OCR (scanned PDFs)
```bash
pip install pytesseract pdf2image
```
Also install the Tesseract OCR program and Poppler on your computer and make sure they are on your PATH.

## 7. Run the tests
```bash
pytest
```
Tests use mocked AI responses. They do not call Gemini or Groq.

## 8. Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| "GEMINI_MODEL is not set" / "model not found" | Set a valid model ID in `.env`, restart the app |
| "rate limit or quota" | Wait a minute; run fewer analyses; lower `MAX_AGENT_STEPS` |
| PDF says "no extractable text" | The PDF is scanned. Install the optional OCR tools |
| "Partial coverage" warning | Document is larger than one run reads. Raise `MAX_AGENT_STEPS` or `MAX_CONTEXT_CHARS`, or select fewer documents |
| PDF shows `?` for some characters | PDFs use built-in Western fonts. Use the DOCX or Markdown export |
| Logo missing in PDF error | `pip install svglib reportlab` |

## 9. Free deployment (Streamlit Community Cloud)
1. Push the project to a GitHub repository (make sure `.env` is NOT committed; `.gitignore` already excludes it).
2. Sign in at share.streamlit.io, create an app from the repo, main file `app.py`.
3. In the app's **Settings > Secrets**, add your keys as TOML, e.g. `GEMINI_API_KEY = "..."`. Top-level secrets are exposed as environment variables.
Limitations: free apps have limited resources and may sleep when idle; uploaded data is not persisted;
everyone using your deployed app shares your API quota, so share the URL carefully.

## 10. Privacy and honest limitations
- Files are parsed locally; only relevant excerpts are sent to Gemini/Groq (processed under their policies).
- Documents, chat and history live in session memory only and vanish on refresh. Nothing is saved to disk.
- Retrieval is keyword-based (TF-IDF style), not semantic embeddings, so paraphrased questions may miss passages.
  The app warns when it falls back to sampled excerpts.
- Long documents: the app reads as much as the call budget allows and states the coverage percentage;
  it never claims a full analysis when coverage is partial.
- PPTX grouped shapes, images and charts are not read. Scanned PDFs need optional OCR.
- AI output can be wrong. Citations point to chunks, so verify important facts in the source files.
- The old voice-command and LLM-generated-code "Action Agent" was removed. Table operations are now a fixed,
  safe set (Document Workspace > Tables). Voice input can be added later as a separate, bounded feature.