import os
import base64
import tiktoken
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ------------------------------
# Global CPU & FAISS Settings
# ------------------------------
faiss.omp_set_num_threads(os.cpu_count())

# ------------------------------
# Streamlit Page and Session Setup
# ------------------------------
st.set_page_config(page_title="چت بات FAQ General", page_icon=":speech_balloon:", layout="centered")

if 'messages' not in st.session_state:
    st.session_state.messages = []  # conversation history
if 'last_retrieved_answer' not in st.session_state:
    st.session_state.last_retrieved_answer = ""
if 'rerank_enabled' not in st.session_state:
    st.session_state.rerank_enabled = True  # Rerank is enabled by default

# ------------------------------
# Utility Functions
# ------------------------------
@st.cache_data(show_spinner=False)
def load_base64_image(image_path: str) -> str:
    """
    Load an image from disk and return its base64 encoding.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"خطا در بارگذاری تصویر {image_path}: {e}")
        return ""

def display_header(image_path: str, title: str) -> None:
    """
    Display a header with a logo and title.
    """
    logo_base64 = load_base64_image(image_path)
    st.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Logo" width="150">
            <h1 style="margin-top: 10px;">{title}</h1>
        </div>
    """, unsafe_allow_html=True)
    st.write()

# Show header
display_header("payeshgaran_logo.jfif", "چت بات FAQ General")

# ------------------------------
# Initialize Tokenizer and System Prompt
# ------------------------------
tokenizer = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = (
    "پاسخ‌های خود را در درجه اول بر اساس اطلاعات بازیابی‌شده از اسناد ارائه دهید. "
    "اگر اطلاعات کافی برای پاسخ به سوال نیست، این موضوع را صریحا بیان کنید و با استفاده از دانش خود، پاسخ را تکمیل کنید."
)

# ------------------------------
# Load Resources with Caching
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_index_and_docs() -> (faiss.Index, pd.DataFrame):
    """
    Load the FAISS index and corresponding documents.
    """
    index = faiss.read_index("faiss_questions (4).index")
    documents = pd.read_csv("questions (4).csv")
    return index, documents

index, documents = load_index_and_docs()

@st.cache_resource(show_spinner=False)
def load_model() -> BGEM3FlagModel:
    """Load the embedding model for dense retrieval."""
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

model = load_model()

# (Comment out the FlagReranker if you no longer need it)
# from FlagEmbedding import FlagReranker
# @st.cache_resource(show_spinner=False)
# def load_reranker() -> FlagReranker:
#     """Load the old reranker model."""
#     return FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
# reranker = load_reranker()

@st.cache_resource(show_spinner=False)
def load_llm() -> ChatGroq:
    """Load the ChatGroq LLM (Gemma)."""
    return ChatGroq(
        groq_api_key="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",
        model_name="Gemma2-9b-It"
    )

llm = load_llm()

@st.cache_resource(show_spinner=False)
def load_llm_llama() -> ChatGroq:
    """Load the ChatGroq LLM (Gemma)."""
    return ChatGroq(
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )

llm_llama = load_llm_llama()

# ------------------------------
# Functions for Retrieval and Chat Processing
# ------------------------------
def get_question_embeddings(question: str) -> np.ndarray:
    """Generate embeddings for the input question."""
    result = model.encode([question], batch_size=12, max_length=512)
    return result['dense_vecs'][0]

def search_questions(query: str, top_k: int = 7) -> pd.DataFrame:
    """
    Search the FAISS index for documents relevant to the query.
    """
    query_embedding = get_question_embeddings(query).astype(np.float16, copy=False)[np.newaxis, :]
    distances, indices = index.search(query_embedding, top_k)
    if indices[0, 0] == -1:
        return pd.DataFrame()
    return documents.iloc[indices[0]]

def gemma_rerank_documents(query: str, docs_df: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
    """
    Use the ChatGroq (Gemma) LLM to re-rank the given docs based on the query.
    Expects an integer index list in JSON (e.g. [1, 0, 2]) as the LLM's output.
    """
    if docs_df.empty:
        return docs_df

    # Build a special re-ranking prompt
    system_message = SystemMessage(
        content=(
            "You are an assistant that re-ranks documents by relevance to a user query. "
            "Return only a JSON array of indices in sorted order (most relevant first) "
            "with no additional explanation."
        )
    )

    titles_list = docs_df["title"].tolist()
    # Make a plain list of doc indices + titles
    titles_text = "\n".join([f"{idx}. {title}" for idx, title in enumerate(titles_list)])

    user_content = (
        f"User query: {query}\n"
        f"Document titles:\n{titles_text}\n\n"
        "Return the re-ranked indices in a JSON array, e.g. [2, 0, 1]."
    )

    # Call Gemma to reorder
    response = llm_llama(messages=[system_message, HumanMessage(content=user_content)])
    raw_output = response.content.strip()

    # Safely parse the JSON list of indices
    # We expect something like: [2, 0, 1]
    try:
        import json
        ranked_indices = json.loads(raw_output)
        # ranked_indices should be a list of integers in descending relevance
        # Reorder the docs_df accordingly
        # Filter out out-of-range or duplicate indices
        valid_indices = [
            idx for idx in ranked_indices
            if isinstance(idx, int) and 0 <= idx < len(docs_df)
        ]
        # Reindex the docs DataFrame based on new order
        reranked_df = docs_df.iloc[valid_indices].reset_index(drop=True)
    except Exception:
        # If parsing fails, just return the original docs
        st.error("خطا در تجزیه خروجی ریرنک Gemma. از ترتیـب اصلی استفاده می‌شود.")
        return docs_df

    # Return only top_n
    return reranked_df.head(top_n)

def count_tokens(messages: list) -> int:
    """Count the tokens across all messages."""
    return sum(len(tokenizer.encode(msg.content)) for msg in messages)

def build_chat_messages(user_question: str, conversation: list, retrieved_info: str) -> list:
    """
    Build messages to send to the LLM.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if retrieved_info and "متأسفم" not in retrieved_info:
        messages.append(HumanMessage(content=f"اطلاعات بازیابی‌شده:\n{retrieved_info}"))
    
    # Append conversation history
    for msg in conversation:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))
    
    # Finally, add the new user query
    messages.append(HumanMessage(content=user_question))
    return messages

MAX_HISTORY = 4  # Limit conversation history

def chatbot(user_question: str, conversation: list) -> (str, str):
    """
    Main chatbot function that:
      1. Retrieves relevant documents.
      2. Optionally re-ranks them with Gemma.
      3. Builds messages and counts tokens.
      4. Queries the LLM for the final answer.
      
    Returns the LLM’s answer and an associated URL (if any).
    """
    relevant_docs = search_questions(user_question)
    
    if relevant_docs.empty:
        retrieved_answer = "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد."
        url = None
    else:
        original_docs = relevant_docs.copy()
        if st.session_state.rerank_enabled:
            # =====================
            # Gemma-based Re-rank
            # =====================
            reranked_docs = gemma_rerank_documents(user_question, relevant_docs)
            # Show a quick info if the top doc changed
            if not reranked_docs.head(1).equals(original_docs.head(1)):
                st.info("ترتیب اسناد توسط ریرنک Gemma تغییر کرد.")
            retrieved_answers = [
                f"سند: {row['title']}\nلینک: {row['url']}" 
                for _, row in reranked_docs.iterrows()
            ]
            retrieved_answer = "\n---\n".join(retrieved_answers)
            url = reranked_docs['url'].iloc[0]
        else:
            retrieved_answers = [
                f"سند: {row['title']}\nلینک: {row['url']}" 
                for _, row in relevant_docs.iterrows()
            ]
            retrieved_answer = "\n---\n".join(retrieved_answers)
            url = relevant_docs['url'].iloc[0]
    
    st.session_state.last_retrieved_answer = retrieved_answer
    messages = build_chat_messages(user_question, conversation, retrieved_answer)
    
    num_tokens = count_tokens(messages)
    st.info(f"تعداد توکن‌ها: {num_tokens}")
    if num_tokens > 8000:
        st.warning("تعداد توکن‌ها از 8K بیشتر شده است!")
    
    # Final LLM call for generating the answer to the user
    response = llm(messages=messages)
    return response.content, url

# ------------------------------
# User Interface Components
# ------------------------------
if st.button("فعال/غیرفعال کردن ریرنک", key="rerank_toggle"):
    st.session_state.rerank_enabled = not st.session_state.rerank_enabled
    st.success(f"ریرنک {'فعال' if st.session_state.rerank_enabled else 'غیرفعال'} شد.")

# Display conversation history using Streamlit’s chat_message interface
for msg in st.session_state.messages:
    role = msg['role']
    with st.chat_message(role):
        st.write(msg['content'])

# Get the user’s new question
user_question = st.chat_input("سوال خود را وارد کنید:")

if user_question and user_question.strip():
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Limit conversation history to the last MAX_HISTORY messages.
    if len(st.session_state.messages) > MAX_HISTORY:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
    
    with st.chat_message("user"):
        st.write(user_question)
    
    with st.chat_message("assistant"):
        with st.spinner("در حال پردازش..."):
            # Pass conversation history excluding the current user message
            answer, url = chatbot(user_question, st.session_state.messages[:-1])
            full_response = f"{url}\n\n{answer}" if url else answer
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            if len(st.session_state.messages) > MAX_HISTORY:
                st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]
            
            if url:
                st.markdown(f"[لینک مرتبط به پاسخ]({url})")
            st.write("\n\n")
            st.write(answer)
