import os
import streamlit as st
# from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load Environment Variables
# load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]     // os.getenv("OPENAI_API_KEY")

# Initialize Streamlit Page
st.set_page_config(
    page_title="NextierGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stSidebar {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Nextier GPT")
st.markdown("Hi, I’m your helpful assistant here to tell you all about Nextier—how it works, what it offers, and how it can support your needs.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_name = st.selectbox(
        "LLM Model",
        ["gpt-4o", "gpt-4o-mini", "o1-mini"],
        index=1,
        help="Select the OpenAI model to analyze the company data."
    )
    
    top_k = st.slider(
        "Context Precision (Top-K)",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of document chunks to retrieve for each query."
    )
    
    st.divider()
    
    if not OPENAI_API_KEY:
        st.info("API Key not found in environment. Please enter it below.")
        OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    else:
        st.success("OpenAI API Key loaded.")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Resource Loading (Cached)
@st.cache_resource
def init_resources():
    # Initialize Embedding Model
    with st.spinner("Loading embedding model..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-V2"
        )
    
    # Load FAISS Index
    if os.path.exists("faiss_index"):
        with st.spinner("Loading FAISS index from disk..."):
            vectorstore = FAISS.load_local(
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vectorstore
    else:
        st.error("❌ FAISS index not found! Please run the indexing script first.")
        return None

# Load Vector Store
vectorstore = init_resources()
if vectorstore is None:
    st.stop()

# Initialize LLM
if not OPENAI_API_KEY:
    st.warning("⚠️ Please provide an OpenAI API key in the sidebar to start chatting.")
    st.stop()

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=model_name,
    streaming=True
)

# Define Prompt Templates
system_prompt = (
    "You are an expert analyst. Answer the question based ONLY on the provided context. "
    "If the context doesn't contain the answer, politely state that the information is not available in the company profile.\n\n"
    "Note that Nextier's business units are Nextier Advisory, Nextier Power, and Nextier SPD (Security, Peace & Development) and Nextier Liberia..\n\n"
    "CONTEXT:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Setup RAG Chain (Modern Pattern)
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(
    vectorstore.as_retriever(search_kwargs={"k": top_k}), 
    question_answer_chain
)

# Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about Nextier..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Searching and thinking..."):
            try:
                # Perform RAG (Modern Pattern)
                result = rag_chain.invoke({"input": prompt})
                answer = result["answer"]
                sources = result["context"]
                
                # Update UI
                response_placeholder.markdown(answer)
                
                # Display Sources in an expander
                if sources:
                    with st.expander("📚 View Source Context"):
                        for i, doc in enumerate(sources):
                            source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                            page_num = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Source {i+1} — {source_name} (p. {page_num})**")
                            st.caption(doc.page_content)
                            st.divider()
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error communicating with OpenAI: {str(e)}")
