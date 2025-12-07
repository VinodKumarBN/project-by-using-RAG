import os
import streamlit as st
from typing import Optional
from io import BytesIO
from google import genai
from google.genai.errors import APIError

# DOCUMENTATION / utilities

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def read_document_content(uploaded_file) -> str:
    """Return the text content of the uploaded file or an 'Error:...' string."""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    try:
        # TXT / MD files
        if file_extension in [".txt", ".md"]:
            return uploaded_file.getvalue().decode("utf-8")

        # PDF files
        elif file_extension == ".pdf":
            if not PdfReader:
                return "Error: Cannot read PDF. Please install pypdf."

            # pypdf expects a file-like object or bytes -> use BytesIO
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        # DOCX files
        elif file_extension == ".docx":
            if not Document:
                return "Error: Cannot read DOCX. Please install python-docx."

            doc = Document(BytesIO(uploaded_file.getvalue()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text

        else:
            return f"Error: Unsupported file type: {file_extension}"

    except Exception as e:
        return f"Error reading file content: {e}"


# -- CONFIGURATION --

from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite" 


class GeminiAPI:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate_content(
        self, model: str, contents: list, system_instruction: str
    ) -> str:
        """
        Call the genai SDK and return a textual answer.
        This implementation is defensive: if the response object isn't a plain string,
        we return a readable string representation or try to extract likely text fields.
        """
        try:
            client = genai.Client(api_key=self.api_key)

            config = genai.types.GenerateContentConfig(system_instruction=system_instruction)

            response = client.models.generate_content(
                model=model, contents=contents, config=config
            )

            # Try to extract text from common response shapes; fall back to str()
            try:
                # typical patterns: response.candidates[0].content or response.output[0].content
                if hasattr(response, "candidates") and response.candidates:
                    first = response.candidates[0]
                    if hasattr(first, "content"):
                        return first.content
                    # some SDKs nest output with parts
                    if hasattr(first, "output") and first.output:
                        return str(first.output)
                # fallback to attributes named 'text' or 'message'
                if hasattr(response, "text"):
                    return response.text
            except Exception:
                pass

            return str(response)

        except APIError as e:
            return f"Error during live API call: A Gemini API error occurred. Details: {e}"

        except Exception as e:
            return f"Error during live API call: An unexpected error occurred. Details: {e}"


# STREAMLIT UI AND LOGIC

st.set_page_config(page_title="Gemini RAG Workshop", layout="wide")
st.title("📘 First RAG System: Contextual Q&A with Gemini")
st.markdown(
    """
This application demonstrates the core concept of **Retrieval-Augmented Generation (RAG)**.
The LLM (Gemini) is forced to answer your questions *only* by referencing the document you provide.
Supported file types : '.txt', '.md', '.pdf', and '.docx'."""
)

# Initialize session state keys (safe defaults)
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

if "rag_response" not in st.session_state:
    st.session_state.rag_response = {}

if "user_prompt_input" not in st.session_state:
    st.session_state.user_prompt_input = ""


# 1) Browse Button to load data source
uploaded_file = st.file_uploader(
    "1. Upload Your Data Source (TXT, MD, PDF, or DOCX)",
    type=["txt", "md", "pdf", "docx"],
    help="Upload a document that Gemini must reference to answer your questions.",
)

if uploaded_file is not None:
    # Use the new generalized function to read the content
    file_contents = read_document_content(uploaded_file)

    if isinstance(file_contents, str) and file_contents.startswith("Error:"):
        st.error(file_contents)
        st.session_state.uploaded_text = ""
        st.stop()
    else:
        st.session_state.uploaded_text = file_contents
        st.success(
            f"File **{uploaded_file.name}** loaded successfully! ({len(file_contents)} characters)"
        )

        with st.expander("Review Extracted Document Text"):
            # Display only the first 2000 characters for performance/readability
            display_text = (
                file_contents[:2000] + "\n... Truncated for display ..."
                if len(file_contents) > 2000
                else file_contents
            )
            st.code(display_text, language="text")

if not st.session_state.uploaded_text:
    st.info("Please upload a supported file type to enable the Q&A section.")
    st.stop()


# 2) Text box for user prompt
st.subheader("2. Ask a Question in the Document")
st.text_area(
    "Enter your question here:",
    placeholder="e.g., what is the main purpose of this document?",
    height=100,
    key="user_prompt_input",
)


# 3) Initialize the API Handler
gemini_api = GeminiAPI(api_key=API_KEY)


def run_rag_query():
    current_prompt = st.session_state.get("user_prompt_input", "").strip()

    if not current_prompt:
        st.error("Please enter a question.")
        return

    if not st.session_state.uploaded_text:
        st.error("Please upload a document first.")
        return

    st.session_state.rag_response = {"prompt": current_prompt, "answer": None}

    with st.spinner(f"Augmenting Generation for: '{current_prompt[:50]}...'"):

        system_instruction = (
            "You are an expert Q&A system. Your sole task is to extract or summarize "
            "information to answer the user's question. DO NOT use external knowledge. "
            "Only use the text provided in the 'context' part of the prompt. "
            "If the answer is not present in the document, you MUST reply with: "
            "'I cannot find the answer in the provided document.'"
        )

        contents_payload = [
            {"parts": [{"text": st.session_state.uploaded_text}]},
            {"parts": [{"text": current_prompt}]},
        ]

        response_text = gemini_api.generate_content(
            model=MODEL_NAME, contents=contents_payload, system_instruction=system_instruction
        )

        st.session_state.rag_response["answer"] = response_text


# 4) Output box for populating the response
st.button("3. Get Grounded Answer", on_click=run_rag_query)

if st.session_state.get("rag_response") and st.session_state["rag_response"].get("answer"):
    st.markdown("---")
    st.subheader("RAG Response")
    st.markdown(f"**Question Asked:** **{st.session_state['rag_response']['prompt']}**")
    st.markdown(st.session_state["rag_response"]["answer"])
    st.markdown("---")
else:
    st.info("Your answer will appear here after you click 'Get Grounded Answer.'")

st.markdown("---")

st.caption(
    "Workshop Key Takeaway: The RAG system works by constructing a powerful prompt that includes both the external "
    "'context' (your document) and the user's query, forcing the LLM to act as a document reader."
)
