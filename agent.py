import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def analyze_data_profile(profile_data: dict, sample_data: list, api_key: str | None = None) -> str:
    """Passes the deterministic profile to the LLM for investigation."""
    resolved_key = (api_key if api_key is not None else (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "")).strip()
    if not resolved_key:
        return "Agent encountered an error: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Please check your .env file."

    model_name = (os.getenv("GEMINI_MODEL") or "gemini-3.7-flash").strip()

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2, 
            google_api_key=resolved_key
        )

        prompt = PromptTemplate(
            input_variables=["profile", "sample"],
            template="""
            You are DataSleuth AI, an expert Data Quality Engineer.
            Review the following automated data profile and sample records.

            DATA PROFILE:
            {profile}

            SAMPLE RECORDS:
            {sample}

            TASK:
            1. Summarize the overall health of this dataset.
            2. Identify the most critical data quality issues (e.g., missing values, duplicates).
            3. Formulate a hypothesis for WHY these errors might exist based on the column names
               and sample data.
            4. Suggest 3 concrete Python/Pandas data cleaning steps to fix these specific issues.

            Keep your response highly technical, structured, and concise.
            """
        )

        chain = prompt | llm
        response = chain.invoke({"profile": profile_data, "sample": sample_data})
        return str(response.content) if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Agent encountered an error calling the LLM API: {str(e)}\nPlease check your GEMINI_API_KEY."