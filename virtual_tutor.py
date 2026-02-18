import streamlit as st
import os
import tempfile
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader, 
    UnstructuredExcelLoader
)
# Note the updated import path below
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Secure API Key Access ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("Missing `.streamlit/secrets.toml`. Create it in your Codespace to proceed.")
    st.stop()

# --- 2. Setup & Folders ---
st.set_page_config(page_title="Big Corpus Search", layout="wide")
DB_PATH = "faiss_index_store" # Folder where the index will be saved

def get_loader(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    
    # LangChain's UnstructuredPowerPointLoader technically supports both, 
    # but needs the 'unstructured' library installed to work with .ppt
    if ext in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path)
    
    if ext == ".pdf": 
        return PyPDFLoader(file_path)
    if ext == ".docx": 
        return Docx2txtLoader(file_path)
    if ext == ".xlsx": 
        return UnstructuredExcelLoader(file_path)
    
    return None

# --- 3. Sidebar UI ---
with st.sidebar:
    st.header("Storage & Ingestion")
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    process_btn = st.button("Build/Update Index")
    
    if st.button("Clear Saved Index"):
        if os.path.exists(DB_PATH):
            import shutil
            shutil.rmtree(DB_PATH)
            st.success("Deleted local index.")

# --- 4. Logic: Ingest and Vectorize ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if process_btn and uploaded_files:
    with st.spinner("Processing large corpus..."):
        all_docs = []
        for f in uploaded_files:
            # 1. Create a safe temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                tmp.write(f.getvalue())
                tmp_path = tmp.name
            
            try:
                # 2. Get the appropriate loader
                loader = get_loader(tmp_path)
                
                # 3. If a valid loader exists, attempt to load text
                if loader is not None:
                    data = loader.load()
                    if data:
                        for d in data: 
                            d.metadata["source"] = f.name
                        all_docs.extend(data)
                    else:
                        st.warning(f"File {f.name} appears to be empty.")
                else:
                    st.error(f"Unsupported file format: {f.name}")
                    
            except Exception as e:
                st.error(f"Error processing {f.name}: {str(e)}")
            finally:
                # 4. Always clean up the temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # --- Proceed to Chunking ---
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(all_docs)
            
            if chunks:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vector_db = FAISS.from_documents(chunks, embeddings)
                vector_db.save_local(DB_PATH)
                st.session_state["db"] = vector_db
                st.success(f"Successfully indexed {len(all_docs)} pages!")
        else:
            st.error("No text was extracted. Ensure your files are not password protected.")
            
# --- 5. Logic: Querying ---
# Load from disk if it exists and isn't in session yet
if "db" not in st.session_state and os.path.exists(DB_PATH):
    st.session_state["db"] = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

if "db" in st.session_state:
    query = st.text_input("Enter your question about the corpus:")
    if query:
        # Retrieve top 5 matches
        docs = st.session_state["db"].similarity_search(query, k=5)
        
        # Combine content for LLM
        context = "\n\n".join([f"SOURCE [{d.metadata['source']}]: {d.page_content}" for d in docs])
        
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0)
        ans = llm.invoke(f"Answer using this context. Cite all sources.\n\nContext:\n{context}\n\nQuestion: {query}")

        st.markdown("### 🤖 Answer")
        st.write(ans.content)
        
        st.markdown("---")
        st.markdown("### 📚 Reference Locations")
        for d in docs:
            with st.expander(f"Snippet from: {d.metadata['source']}"):
                st.write(d.page_content)