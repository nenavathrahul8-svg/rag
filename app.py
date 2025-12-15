import streamlit as st
import tempfile
import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page Config
st.set_page_config(page_title="Secure RAG System", layout="wide")

st.title("🔒 Secure RAG Data Access")
st.markdown("""
This system uses **Prompt Engineering** to explicitly restrict access to sensitive fields 
(e.g., Account Numbers, Phone Numbers, Exact Balances, Salaries).
It allows for high-level summaries and permitted analytical insights.
""")

# Sidebar for Setup
with st.sidebar:
    st.header("Configuration")
    provider = st.radio("Select LLM Provider", ["Google Gemini", "OpenAI"])
    api_key = st.text_input("Enter API Key", type="password")
    
    st.divider()
    st.info("System restricts: Account #, Phone #, Credit Score, Salary")

# Initialize Session State
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        if uploaded_file.name.endswith('.csv'):
            loader = CSVLoader(file_path=tmp_path)
        elif uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path=tmp_path)
        else:
            loader = TextLoader(file_path=tmp_path)
            
        docs = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Create Vector Store (using local embeddings to avoid cost/limit issues for embeddings)
        # Using a lightweight model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_db = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings,
            collection_name="secure_rag_collection"
        )
        
        return vector_db
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None
    finally:
        os.remove(tmp_path) # Cleanup

# File Uploader
uploaded_file = st.file_uploader("Upload Document (CSV, PDF, TXT)", type=["csv", "pdf", "txt"])

if uploaded_file and st.button("Process Document"):
    with st.spinner("Processing document... (This may take a moment for embeddings)"):
        st.session_state.vector_db = process_file(uploaded_file)
        if st.session_state.vector_db:
            st.success("Document processed and embedded successfully!")

# Chat Interface
if st.session_state.vector_db:
    st.divider()
    query = st.text_input("Ask a question about the document:")
    
    if query:
        if not api_key:
            st.warning("Please enter an API Key in the sidebar to proceed.")
        else:
            try:
                # Retriever
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                
                # LLM Setup
                if provider == "Google Gemini":
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0)
                else:
                    llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)

                # STRICT SECURITY PROMPT
                system_template = """
                You are a highly secure data analyst assistant. 
                You are provided with context from internal documents that may contain SENSITIVE PRIVATE INFORMATION (PII).
                
                YOUR PRIMARY DIRECTIVE IS TO PROTECT SENSITIVE DATA.
                
                STRICT RULES:
                1. NEVER reveal individual "Account Numbers" (full or partial, mask as XXXX-XXXX).
                2. NEVER reveal individual "Phone Numbers".
                3. NEVER reveal exact "Salaries" or "Credit Scores" for specific individuals.
                4. NEVER reveal exact "Balances" for specific individuals (providing ranges or "high/low" is okay).
                5. If a user asks for any of the above specific restricted information, you must REFUSE nicely.
                   Example violation request: "What is John Doe's account number?"
                   Example Compliant Response: "I cannot provide specific account numbers for security reasons. However, I can confirm that John Doe has a transaction in the dataset."
                
                ALLOWED ACTIONS:
                1. You MAY provide aggregated statistics (e.g., "Total balance across all accounts", "Average salary", "Count of transactions").
                2. You MAY provide high-level summaries (e.g., "John spending seems focused on Groceries").
                3. You MAY provide specific transaction details IF they are not one of the restricted fields (e.g. Date, Merchant, Category are OK).
                
                CONTEXT:
                {context}
                
                USER QUESTION: 
                {question}
                
                Generate a response that helps the user but strictly adheres to the security rules. 
                If you have to block information, explicitly state that it is restricted for privacy.
                """
                
                prompt = ChatPromptTemplate.from_template(system_template)
                
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                with st.spinner("Analyzing securely..."):
                    response = chain.invoke(query)
                    
                st.markdown("### Response:")
                st.write(response)
                
                with st.expander("Show Retrieved Context (Debug/Demo)"):
                    docs = retriever.invoke(query)
                    for i, doc in enumerate(docs):
                        st.text(f"Chunk {i+1}:")
                        st.caption(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error during generation: {e}")
                
else:
    st.info("Upload a document to start.")
