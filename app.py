import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
import os
import time
import logging
import sys
import shutil
import contextlib
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import yaml
import litellm

from langchain_litellm import ChatLiteLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from loaders.load_documents import load_docs
from utils.chunker import chunk_documents_hierarchical
from utils.embedder import embed_and_store
from chains.rag_chains import get_rag_chain
from utils.query_classifier import classify_query

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('chromadb').setLevel(logging.CRITICAL)
os.environ["LITELLM_CONFIG_PATH"] = "config/litellm_config.yaml"

PRIMARY_MODEL = "groq/meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "ollama/gemma2:2b-gpu-only"
CLASSIFIER_MODEL = "groq/llama3-8b-8192"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
EXACT_DISCLAIMER = os.getenv("EXACT_DISCLAIMER")
DB_PATH = os.getenv("DB_PATH")

st.set_page_config(page_title="Quantum Tutor", layout="centered")
st.title("Quantum Tutor ⚛️")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

@st.cache_resource
def get_llm_with_fallback():
    try:
        llm = ChatLiteLLM(model=PRIMARY_MODEL, temperature=0.2, max_tokens=2048)
        llm.invoke("test connection")
        logging.info(f"Using primary LLM: {PRIMARY_MODEL}")
        st.session_state.active_model_name = PRIMARY_MODEL
        return llm
    except Exception as e:
        logging.warning(f"Primary LLM ({PRIMARY_MODEL}) failed: {e}. Falling back to {FALLBACK_MODEL}.")
        st.session_state.active_model_name = FALLBACK_MODEL
        return ChatLiteLLM(model=FALLBACK_MODEL, temperature=0.4)

@st.cache_resource
def get_classifier_llm():
    try:
        llm = ChatLiteLLM(model=CLASSIFIER_MODEL, temperature=0.0)
        llm.invoke("test connection")
        logging.info(f"Using classifier LLM: {CLASSIFIER_MODEL}")
        return llm
    except Exception as e:
        logging.warning(f"Classifier LLM ({CLASSIFIER_MODEL}) failed: {e}. Falling back to {FALLBACK_MODEL}.")
        return ChatLiteLLM(model=FALLBACK_MODEL, temperature=0.0)

llm_for_rag = get_llm_with_fallback()
llm_classifier = get_classifier_llm()

def get_active_chat():
    if st.session_state.active_chat_id is None and not st.session_state.chat_sessions:
        new_chat(activate=False)
    for session in st.session_state.chat_sessions:
        if session["id"] == st.session_state.active_chat_id:
            return session
    if st.session_state.chat_sessions:
        st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]
        return st.session_state.chat_sessions[0]
    return None

def new_chat(activate=True):
    chat_id = str(int(time.time()))
    st.session_state.chat_sessions.append({
        "id": chat_id, "title": "New Chat", "messages": [], "history": ChatMessageHistory(),
    })
    st.session_state.active_chat_id = chat_id
    if activate:
        st.rerun()

def generate_conversation_title(llm, history):
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    prompt = (
        "Based on the following conversation, create a concise title. "
        "The title must be 5 words or less. "
        "Do not use any punctuation or quotation marks. "
        "Provide only the raw text for the title."
        f"\n\nConversation:\n{history_str}\n\nTitle:"
    )
    try:
        title_response = llm.invoke(prompt)
        clean_title = (title_response.content if hasattr(title_response, 'content') else str(title_response))
        clean_title = clean_title.replace('"', '').replace("'", "").strip()
        return clean_title if clean_title else "Quantum Chat"
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return "Quantum Chat"

def log_feedback(feedback_type, response, comment="N/A"):
    timestamp = datetime.now().isoformat()
    username = st.session_state.get('display_name', 'N/A')
    feedback_log_entry = (f"[{timestamp}] User: {username}\n" f"Vote: {feedback_type}\n" f"Response: {response[:200]}...\n" f"Comment: {comment.strip() or 'N/A'}\n" "-----\n")
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(feedback_log_entry)

def display_feedback_form(message, key_suffix):
    st.write("")
    feedback_cols = st.columns([1, 1, 8])
    with feedback_cols[0]:
        if st.button("👍", key=f"good_{key_suffix}"):
            log_feedback("GOOD", message['content'])
            st.toast("✅ Thanks for your feedback!")
    with feedback_cols[1]:
        if st.button("👎", key=f"bad_{key_suffix}"):
            st.session_state[f"show_comment_for_{key_suffix}"] = True
            st.rerun()
    if st.session_state.get(f"show_comment_for_{key_suffix}"):
        with st.form(key=f"comment_form_{key_suffix}"):
            comment = st.text_area("Please provide additional comments:", key=f"comment_{key_suffix}")
            if st.form_submit_button("Submit Comment"):
                log_feedback("BAD", message['content'], comment)
                st.toast("✅ Thanks for your detailed feedback!")
                st.session_state[f"show_comment_for_{key_suffix}"] = False
                st.rerun()

@st.cache_resource(show_spinner="Loading and processing documents...")
def warmup_db():
    start_time = time.time()
    child_db_path = os.path.join(DB_PATH, "child_chunks")
    parent_db_path = os.path.join(DB_PATH, "parent_chunks")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={'normalize_embeddings': True})
    child_chunks = []
    
    if os.path.exists(child_db_path) and os.path.exists(parent_db_path):
        try:
            child_vectordb = Chroma(persist_directory=child_db_path, embedding_function=embedding_model)
            parent_vectordb = Chroma(persist_directory=parent_db_path, embedding_function=embedding_model)
            retrieved_child_data = child_vectordb.get(include=['documents', 'metadatas'])
            child_chunks = [Document(page_content=d, metadata=m) for d, m in zip(retrieved_child_data['documents'], retrieved_child_data['metadatas'])]
            parent_docstore = InMemoryStore()
            retrieved_data = parent_vectordb.get(ids=parent_vectordb.get()['ids'], include=["documents", "metadatas"])
            parent_docs = [Document(page_content=d, metadata=m) for d, m in zip(retrieved_data["documents"], retrieved_data["metadatas"]) if "parent_id" in m]
            parent_docstore.mset([(doc.metadata["parent_id"], doc) for doc in parent_docs])
            logger.info("Knowledge base loaded from disk.")
            end_time = time.time()
            return child_vectordb, parent_docstore, parent_vectordb, child_chunks, end_time - start_time
        except Exception as e:
            st.warning(f"Failed to load DBs: {e}. Rebuilding...")
    
    if os.path.exists(DB_PATH): shutil.rmtree(DB_PATH)
    docs_loaded = load_docs()
    if not docs_loaded:
        st.error("No documents loaded. Please check the `data` folder.")
        st.stop()
    child_chunks, parent_chunks = chunk_documents_hierarchical(docs_loaded)
    child_vectordb, parent_docstore, parent_vectordb = embed_and_store((child_chunks, parent_chunks))
    end_time = time.time()
    logger.info(f"Knowledge base built and loaded in {end_time - start_time:.2f}s")
    return child_vectordb, parent_docstore, parent_vectordb, child_chunks, end_time - start_time

child_vectordb, parent_docstore, parent_vectordb, child_chunks, db_load_time = warmup_db()

_, _, _, compression_retriever = get_rag_chain(
    child_vectordb=child_vectordb,
    parent_docstore=parent_docstore,
    parent_vectordb=parent_vectordb,
    model_name=st.session_state.get('active_model_name', PRIMARY_MODEL),
    child_chunks=child_chunks
)

active_chat = get_active_chat()
if not active_chat:
    st.stop()

history_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up relevant information.")
])
history_aware_retriever = create_history_aware_retriever(llm_classifier, compression_retriever, history_prompt)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for quantum computing. Answer the user's question based on the context provided. Be concise and clear.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
answer_chain = create_stuff_documents_chain(llm_for_rag, answer_prompt)
conversational_chain = create_retrieval_chain(history_aware_retriever, answer_chain)

conversational_rag_chain_with_history = RunnableWithMessageHistory(
    conversational_chain,
    lambda session_id: active_chat["history"],
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

user = st.text_input("Enter your name to begin", key="username")
if not user:
    st.stop()
display_name = user.strip().title()
st.session_state.display_name = display_name
active_model_name = st.session_state.get('active_model_name', 'Unknown')

with st.sidebar:
    st.markdown(f"👤 **User**: {display_name}")
    st.markdown(f"⚡ **Active Model**: {active_model_name}")
    st.markdown(f"💾 **DB Load Time**: {db_load_time:.2f}s")
    st.markdown("---")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
    st.markdown("### Chat History")
    for session in st.session_state.chat_sessions:
        if st.button(session["title"], key=session["id"], use_container_width=True):
            st.session_state.active_chat_id = session["id"]
            st.rerun()

if not active_chat["messages"]:
    with st.spinner("Assistant is waking up..."):
        greeting_prompt = (
            f"Craft a short, friendly, and enthusiastic welcome message for a user named {display_name}. "
            "You are their personal Quantum Tutor. Include a fun emoji like 🚀 or ⚛️. Do not ask them a question back."
        )
        try:
            greeting_response = llm_for_rag.invoke(greeting_prompt)
            generated_greeting = greeting_response.content if hasattr(greeting_response, 'content') else str(greeting_response)
        except Exception as e:
            logger.error(f"Failed to generate dynamic greeting: {e}")
            generated_greeting = f"Hello {display_name}! 👋 I'm ready to help you explore the world of quantum computing! ⚛️"
        active_chat["messages"].append({"role": "assistant", "content": generated_greeting})
        active_chat["history"].add_ai_message(generated_greeting)

for i, msg in enumerate(active_chat["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📌 Source Documents", expanded=False):
                for doc in msg["sources"]:
                    source_name = doc.metadata.get("source", "N/A")
                    display_source = source_name.split('/')[-1] if source_name != 'N/A' else 'N/A'
                    st.markdown(f"**Source** - *{display_source}*, page {doc.metadata.get('page', '?')}\n\n> {doc.page_content[:300].strip()}...")
        if msg["role"] == "assistant" and i > 0:
            key_suffix = hash(msg['content'])
            display_feedback_form(msg, key_suffix=key_suffix)

if prompt := st.chat_input(f"{display_name}, ask something..."):
    active_chat["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    start_time = time.time()
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            query_category = classify_query(prompt)

            if query_category == "OUT_OF_SCOPE":
                full_response = EXACT_DISCLAIMER
                st.markdown(full_response)
                active_chat["messages"].append({"role": "assistant", "content": full_response, "sources": []})
                active_chat["history"].add_user_message(prompt)
                active_chat["history"].add_ai_message(full_response)
            else:
                response_placeholder = st.empty()
                full_response = ""
                unique_docs = []
                
                config = {"configurable": {"session_id": active_chat["id"]}}
                for chunk in conversational_rag_chain_with_history.stream({"input": prompt}, config=config):
                    if "answer" in chunk and chunk["answer"]:
                        full_response += chunk["answer"]
                        response_placeholder.markdown(full_response + "▌")
                    if "context" in chunk and chunk["context"]:
                        unique_docs = chunk["context"]

                response_placeholder.markdown(full_response)
                active_chat["messages"].append({"role": "assistant", "content": full_response, "sources": unique_docs})

                if unique_docs:
                    with st.expander("📌 Source Documents", expanded=False):
                        for doc in unique_docs:
                            source_name = doc.metadata.get("source", "N/A")
                            display_source = source_name.split('/')[-1] if source_name != 'N/A' else 'N/A'
                            st.markdown(f"**Source** - *{display_source}*, page {doc.metadata.get('page', '?')}\n\n> {doc.page_content[:300].strip()}...")
                
                key_suffix = hash(full_response)
                display_feedback_form(active_chat["messages"][-1], key_suffix=key_suffix)

    st.markdown(f"_⏱️ Total Response Time: {time.time() - start_time:.2f}s_")

    if (active_chat["title"] == "New Chat" and 'unique_docs' in locals() and unique_docs):
        title_messages = active_chat["history"].messages
        history_for_title = [{"role": "user" if "Human" in str(type(m)) else "assistant", "content": m.content} for m in title_messages]
        title = generate_conversation_title(llm_classifier, history_for_title)
        active_chat["title"] = title
        st.rerun()