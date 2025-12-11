import streamlit as st
import tempfile
import os
import threading
import re
import requests
import hashlib
from pdf2image.exceptions import PDFInfoNotInstalledError
from langchain_community.document_loaders import UnstructuredPDFLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
# Voice input support
from streamlit_mic_recorder import speech_to_text
# Text-to-Speech (offline)
import pyttsx3
from typing import List
import json
from datetime import datetime
import uuid

# Set Poppler path for Windows PDF processing
POPPLER_PATH = r"C:\poppler-25.07.0\Library\bin"
os.environ["POPPLER_PATH"] = POPPLER_PATH
# Also ensure the Poppler bin path is on PATH so pdf2image/pdfinfo can be found
os.environ["PATH"] = POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# Global flag to control speech
stop_speaking = False


# ---------------- PDF & DOCX Extraction ---------------- #
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        loader = UnstructuredPDFLoader(
            tmp_file_path,
            unstructured_kwargs={"pdf_extractor": "pdf2image"}
        )
        docs = loader.load()
    except (PDFInfoNotInstalledError, FileNotFoundError, RuntimeError):
        # Fallback if Poppler or OCR path not available
        loader = UnstructuredPDFLoader(
            tmp_file_path,
            unstructured_kwargs={"pdf_extractor": "pdfminer"}
        )
        docs = loader.load()
    finally:
        os.unlink(tmp_file_path)

    return docs


def extract_text_from_docx(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = Docx2txtLoader(tmp_file_path)
    docs = loader.load()
    os.unlink(tmp_file_path)
    return docs


# ---------------- Text-to-Speech ---------------- #
def speak_text(text):
    global stop_speaking, tts_engine
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    if stop_speaking:
        return
    engine.say(text)
    try:
        engine.runAndWait()
    except:
        pass


def stop_speech():
    global stop_speaking
    stop_speaking = True


def reset_speech():
    global stop_speaking
    stop_speaking = False


# ---------------- Google Drive / Docs Fetch ---------------- #
def fetch_google_drive_file(url):
    file_id_match = re.search(r"/d/([A-Za-z0-9_-]+)", url)
    if not file_id_match:
        file_id_match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if not file_id_match:
        raise ValueError("Could not extract Google Drive file ID.")

    file_id = file_id_match.group(1)
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    resp = requests.get(direct_url, stream=True)
    if resp.status_code != 200:
        raise ValueError(f"Download failed (status {resp.status_code}). File may not be public.")

    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        ext = ".pdf"
    elif "word" in content_type or "docx" in content_type:
        ext = ".docx"
    else:
        ext = ".pdf" if "pdf" in url.lower() else ".docx"

    return resp.content, ext


def extract_from_raw_bytes(raw_bytes, ext):
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(raw_bytes)
        path = tmp.name
    try:
        if ext == ".pdf":
            loader = UnstructuredPDFLoader(path, unstructured_kwargs={"pdf_extractor": "pdf2image"})
        else:
            loader = Docx2txtLoader(path)
        docs = loader.load()
    finally:
        os.unlink(path)
    return docs


def fetch_google_docs(url):
    doc_id_match = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
    if not doc_id_match:
        raise ValueError("Could not extract Google Docs document ID from URL.")

    doc_id = doc_id_match.group(1)
    export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"

    headers = {
        "User-Agent": "ChatDocBot/1.0 (+https://github.com/YourUser/YourRepo)",
    }

    resp = requests.get(export_url, headers=headers)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch Google Docs (status {resp.status_code}). Document may not be public.")

    from langchain.schema import Document
    content = resp.text
    return [Document(page_content=content, metadata={"source": url, "type": "google_docs"})]


def fetch_url_documents(url):
    url = url.strip()
    if not url:
        return []

    if "docs.google.com/document" in url:
        return fetch_google_docs(url)

    if "drive.google.com" in url:
        raw, ext = fetch_google_drive_file(url)
        return extract_from_raw_bytes(raw, ext)

    lowered = url.lower()
    if lowered.endswith(".pdf") or lowered.endswith(".docx"):
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError(f"Failed to download file (status {resp.status_code}).")
        ext = ".pdf" if lowered.endswith(".pdf") else ".docx"
        return extract_from_raw_bytes(resp.content, ext)

    loader = WebBaseLoader(
        url,
        header_template={
            "User-Agent": "ChatDocBot/1.0 (+https://github.com/YourUser/YourRepo)",
            "Accept-Language": "en-US,en;q=0.9"
        }
    )
    return loader.load()


# ---------------- Vector Store & Embeddings ---------------- #
def build_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_documents(chunks, embeddings)


def cache_key(docs):
    h = hashlib.sha256()
    for d in docs:
        h.update(str(len(d.page_content)).encode())
        h.update(d.page_content[:200].encode(errors="ignore"))
    return h.hexdigest()


# ---------------- Search History ---------------- #
def save_search_history(query, answer, lang, sources_count=0):
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    history_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "language": lang,
        "sources_count": sources_count
    }

    st.session_state.search_history.append(history_entry)

    try:
        with open("search_history.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.search_history, f, ensure_ascii=False, indent=2)
    except:
        pass


def delete_history_entry(entry_id: str):
    if "search_history" in st.session_state:
        st.session_state.search_history = [
            h for h in st.session_state.search_history if h.get("id") != entry_id
        ]
        try:
            with open("search_history.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.search_history, f, ensure_ascii=False, indent=2)
        except:
            pass


def load_search_history():
    try:
        with open("search_history.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            changed = False
            for entry in data:
                if "id" not in entry:
                    entry["id"] = str(uuid.uuid4())
                    changed = True
            if changed:
                with open("search_history.json", "w", encoding="utf-8") as wf:
                    json.dump(data, wf, ensure_ascii=False, indent=2)
            return data
    except:
        return []


def ensure_history_ids():
    if "search_history" in st.session_state:
        changed = False
        for entry in st.session_state.search_history:
            if "id" not in entry:
                entry["id"] = str(uuid.uuid4())
                changed = True
        if changed:
            with open("search_history.json", "w", encoding="utf-8") as f:
                json.dump(st.session_state.search_history, f, ensure_ascii=False, indent=2)


def display_search_history():
    if "search_history" not in st.session_state or not st.session_state.search_history:
        st.info("No search history yet.")
        return

    ensure_history_ids()
    st.write(f"**Total searches:** {len(st.session_state.search_history)}")

    recent_history = st.session_state.search_history[-10:][::-1]
    for idx, entry in enumerate(recent_history):
        entry_id = entry.get("id", str(uuid.uuid4()))
        entry["id"] = entry_id
        timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        with st.expander(f"🔍 {entry['query'][:50]}... - {timestamp}"):
            st.write(f"**Language:** {entry['language']}")
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Answer:** {entry['answer']}")
            st.write(f"**Sources used:** {entry['sources_count']}")
            cols = st.columns(3)
            with cols[0]:
                if st.button("🔄 Reuse", key=f"reuse_{entry_id}"):
                    st.session_state.reused_query = entry['query']
                    st.rerun()
            with cols[1]:
                if st.button("🗑 Delete", key=f"del_{entry_id}"):
                    delete_history_entry(entry_id)
                    st.rerun()
            with cols[2]:
                st.caption(entry_id)


def clear_search_history():
    st.session_state.search_history = []
    try:
        if os.path.exists("search_history.json"):
            os.remove("search_history.json")
    except:
        pass


def export_search_history():
    if "search_history" not in st.session_state or not st.session_state.search_history:
        return None
    history_json = json.dumps(st.session_state.search_history, ensure_ascii=False, indent=2)
    return history_json.encode('utf-8')


# ---------------- Main Streamlit App ---------------- #
def main():
    st.title("OmniDoc AI: The Universal Document Intelligence Assistant")

    # Initialize session state
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "vectordb_key" not in st.session_state:
        st.session_state.vectordb_key = None
    if "search_history" not in st.session_state:
        st.session_state.search_history = load_search_history()
    ensure_history_ids()
    if "reused_query" not in st.session_state:
        st.session_state.reused_query = ""

    # Sidebar for search history
    with st.sidebar:
        st.header("🔍 Search History")

        if st.session_state.search_history:
            total_searches = len(st.session_state.search_history)
            st.metric("Total Searches", total_searches)

            if st.checkbox("Show Recent Queries"):
                recent = st.session_state.search_history[-5:][::-1]
                for entry in recent:
                    timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%m-%d %H:%M")
                    st.text(f"{timestamp}: {entry['query'][:30]}...")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 View All"):
                st.session_state.show_history = True
        with col2:
            if st.button("🗑 Clear"):
                clear_search_history()
                st.success("History cleared!")
                st.rerun()

        if st.session_state.search_history:
            history_data = export_search_history()
            if history_data:
                st.download_button(
                    label="💾 Export History",
                    data=history_data,
                    file_name=f"search_history_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )

    lang_choice = st.selectbox("Interaction language", ["English", "Kannada"], index=0)

    model_choice = st.selectbox(
        "Model",
        ["llama3.2:3b", "llama3.1:8b", "Claude Haiku 4.5 (Anthropic)"],
        index=0,
        help="Choose local Ollama model or Anthropic Claude Haiku 4.5 (requires ANTHROPIC_API_KEY)"
    )
    st.session_state["ollama_model"] = model_choice

    uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])
    url_input = st.text_input("Or enter a web / Google Drive link (public):")
    fetch_clicked = st.button("Fetch URL")

    docs_changed = False

    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == "pdf":
            with st.spinner("Extracting text from uploaded PDF..."):
                st.session_state.docs.extend(extract_text_from_pdf(uploaded_file))
        elif ext == "docx":
            with st.spinner("Extracting text from uploaded DOCX..."):
                st.session_state.docs.extend(extract_text_from_docx(uploaded_file))
        st.session_state.processed_files.append(uploaded_file.name)
        docs_changed = True

    if fetch_clicked and url_input:
        try:
            with st.spinner("Fetching URL content..."):
                fetched = fetch_url_documents(url_input)
                st.session_state.docs.extend(fetched)
                st.success(f"Fetched {len(fetched)} chunk(s) from URL.")
                docs_changed = True
        except Exception as e:
            st.error(f"URL fetch failed: {e}")

    docs = st.session_state.docs

    if hasattr(st.session_state, 'show_history') and st.session_state.show_history:
        st.header("📜 Complete Search History")
        display_search_history()
        if st.button("❌ Close History"):
            st.session_state.show_history = False
            st.rerun()
        st.divider()

    if docs:
        if docs_changed:
            with st.spinner("Building / updating vector index..."):
                key = cache_key(docs)
                st.session_state.vectordb = build_vectorstore(docs)
                st.session_state.vectordb_key = key
        elif st.session_state.vectordb is None:
            with st.spinner("Building vector index..."):
                key = cache_key(docs)
                st.session_state.vectordb = build_vectorstore(docs)
                st.session_state.vectordb_key = key

        vectordb = st.session_state.vectordb

        st.write("### Document preview:")
        preview_text = "\n\n".join(d.page_content for d in docs[:2])
        st.text_area("Preview (first documents, truncated):", preview_text[:1000] + "...", height=200)

        selected_model = st.session_state.get("ollama_model", "llama3.2:3b")

        # Anthropic / Claude Haiku 4.5 path
        if selected_model == "Claude Haiku 4.5 (Anthropic)":
            try:
                from langchain.chat_models import ChatAnthropic
            except Exception:
                st.error("Anthropic/chat model integration not available. Install 'anthropic' and a compatible 'langchain' version.")
                return

            # Initialize ChatAnthropic. Requires environment variable ANTHROPIC_API_KEY to be set.
            try:
                anth_llm = ChatAnthropic(model="claude-haiku-4.5", temperature=0)

                # Adapter to provide the small 'invoke' wrapper used elsewhere in the app
                class _AnthropicAdapter:
                    def __init__(self, llm):
                        self.llm = llm

                    def invoke(self, prompt: str):
                        try:
                            # LangChain chat models often expose 'predict' returning a string
                            out = self.llm.predict(prompt)
                        except Exception:
                            try:
                                out = self.llm(prompt)
                            except Exception:
                                # Fall back to generate and try to extract
                                gen = self.llm.generate([prompt])
                                try:
                                    out = gen.generations[0][0].text
                                except Exception:
                                    out = str(gen)
                        return type("R", (), {"content": out})

                    def __getattr__(self, name):
                        return getattr(self.llm, name)

                llm = _AnthropicAdapter(anth_llm)
                st.caption("Using model: Claude Haiku 4.5 (Anthropic)")
            except Exception as e:
                st.error(f"Failed initializing Anthropic client: {e}")
                return

        else:
            # Default: Ollama
            llm = ChatOllama(
                model=selected_model,
                temperature=0,
                num_ctx=2048,
                num_predict=512
            )

            st.caption(f"Using model: {selected_model}")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        col1, col2 = st.columns([1, 8])
        with col1:
            st.markdown('<span title="Record your voice question"><span style="font-size:35px; cursor:pointer;">🎤</span></span>', unsafe_allow_html=True)
        with col2:
            st.write("Ask by voice or type:")

        if lang_choice == "Kannada":
            voice_query = speech_to_text(language='kn', just_once=False, key='voice_input')
        else:
            voice_query = speech_to_text(language='en', just_once=False, key='voice_input')

        if st.session_state.reused_query:
            query = st.text_input("Type your question:", value=st.session_state.reused_query, key="type_box")
            st.session_state.reused_query = ""
        elif voice_query:
            query = st.text_input("Recognized voice input:", value=voice_query, key="voice_box")
        else:
            query = st.text_input("Type your question:", key="type_box")

        final_query = query.strip()

        if final_query:
            retrieval_query = final_query
            if lang_choice == "Kannada":
                trans_q = llm.invoke("Translate the following Kannada question to English. Reply ONLY with the English translation:\n\n" + final_query)
                retrieval_query = getattr(trans_q, "content", str(trans_q))

            with st.spinner("Generating answer..."):
                result = qa_chain.invoke({"query": retrieval_query})
                if isinstance(result, dict):
                    answer = result.get("result") or result.get("answer") or ""
                    sources = result.get("source_documents") or []
                else:
                    answer = str(result)
                    sources = []

            if lang_choice == "Kannada":
                trans_a = llm.invoke("Translate the following answer to Kannada. Reply ONLY with the Kannada translation:\n\n" + answer)
                answer = getattr(trans_a, "content", str(trans_a))

            st.write("*Answer:*", answer)

            save_search_history(final_query, answer, lang_choice, len(sources))

            if sources:
                with st.expander("Sources"):
                    for i, src in enumerate(sources, 1):
                        snippet = src.page_content[:350].replace("\n", " ")
                        st.write(f"**Source {i}:** {snippet}...")
                        meta = src.metadata
                        if meta:
                            st.caption(str(meta))

            st.markdown("### 🔊 Audio Output")
            cols = st.columns(2)
            with cols[0]:
                if st.button("▶ Speak", key="speak_btn"):
                    reset_speech()
                    tts_thread = threading.Thread(target=speak_text, args=(answer,))
                    tts_thread.start()
            with cols[1]:
                if st.button("⏹ Stop", key="stop_btn"):
                    stop_speech()
                    st.warning("Speech stopped.")

        else:
            st.info("Type or speak your question to get started.")
    else:
        st.info("Please upload or fetch a document first.")


if __name__ == "__main__":
    main()
