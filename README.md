# Anuman AI: Agentic Document & Tabular Intelligence Platform

Anuman AI is an enterprise-grade agentic document intelligence and tabular data analysis platform built with Python, Streamlit, Google Gemini, and Groq. The system processes complex structured and unstructured documents (CSV, Excel, PDF, TXT, DOCX, XLSX, PPTX, MD) entirely in memory, orchestrating specialized AI agents for multi-task analytical workflows, grounded context retrieval, and professional report compilation.

Live Deployment: https://anuman-ai-analytics.streamlit.app/

---

## System Architecture & Key Capabilities

- **Hybrid LLM Orchestration:** Integrates **Google Gemini** for high-context analytical synthesis, summarization, extraction, and comparison, combined with ultra-low latency **Groq** inference for interactive retrieval-grounded chat.
- **Agentic Workflow Engine:** Autonomous routing of analytical requests into structured workflows including summarization, key metric extraction, cross-document comparison, and data profiling.
- **Deterministic Tabular Processing:** Ingests large datasets with in-memory segmentation, schema inspection, local data manipulation, and quality profiling without sending complete raw data off-premises.
- **Source Citation & Traceability:** Strictly grounds analytical deductions in parsed document segments with exact row and section references.
- **Enterprise Report Generation:** Compiles findings into downloadable reports (Markdown, DOCX, CSV, Excel) with executive summaries and methodology disclaimers.
- **Stateless Privacy Architecture:** No user data or documents are persisted to disk; operations remain strictly within browser session memory.

---

## User Interface & System Visuals

### 1. Central Analytics Dashboard
![Anuman AI - Central Analytics Dashboard](https://github.com/Mysterious-Magici/Anuman-AI/blob/main/Central%20Analytics%20Dashboard.png)
*Real-time session telemetry, document counters, provider status, and navigation routing.*

### 2. Document Workspace & Data Inspection
![Anuman AI - Document Ingestion and Tabular Preview](https://github.com/Mysterious-Magici/Anuman-AI/blob/main/Document%20Upload%20Queue.png)
*In-memory document segmentation, character metrics, metadata validation, and tabular data inspection.*

### 3. Grounded AI Assistant (RAG Chat)
![Anuman AI - Grounded Chat Assistant](https://github.com/Mysterious-Magici/Anuman-AI/blob/main/AI%20Assistant%20Module.png)
*Low-latency conversational agent with exact row-level citations from active documents.*

### 4. Multi-Task Document Analysis
![Anuman AI - Agentic Task Orchestration](https://github.com/Mysterious-Magici/Anuman-AI/blob/main/Agentic%20Task%20Routing.png)
*Configurable task execution: Auto-detect, structured data extraction, summarization, and data profiling.*

### 5. Session History & Audit Trail
![Anuman AI - Session History and Audit Logs](https://github.com/Mysterious-Magici/Anuman-AI/blob/main/Ephemeral%20Session%20History.png)
*Non-persistent, in-memory session logging tracking analytical traces and assistant interactions.*

---

## Tech Stack

- **Frontend & Interface:** Streamlit, Custom Responsive CSS
- **Large Language Models:** Google Gemini API, Groq Cloud API
- **Data Engineering & Manipulation:** Pandas, NumPy, OpenPyXL
- **Document Ingestion:** PyPDF, python-docx, OCR integration
- **Workflow Architecture:** Agentic multi-step task execution, Session-State Management
- **Deployment:** Streamlit Community Cloud

---

## Local Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Mysterious-Magici/Anuman-AI.git](https://github.com/Mysterious-Magici/Anuman-AI.git)
   cd Anuman-AI
