import tiktoken
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="چت بات FAQ General", page_icon=":speech_balloon:", layout="centered")

# Initialize the tokenizer for token counting (using a model similar to GPT-3 or GPT-4)
tokenizer = tiktoken.get_encoding("cl100k_base")

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        import base64
        return base64.b64encode(image_file.read()).decode()

# Encode the image
logo_base64 = get_base64_encoded_image("payeshgaran_logo.jfif")

# Use HTML to center-align the logo and title
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/jpeg;base64,{logo_base64}" alt="Logo" width="150">
        <h1 style="margin-top: 10px;">چت بات FAQ General</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.write()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'last_retrieved_answer' not in st.session_state:
    st.session_state['last_retrieved_answer'] = ""

if 'rerank_enabled' not in st.session_state:
    st.session_state['rerank_enabled'] = True  # Default is that rerank is enabled

SYSTEM_PROMPT = (
    "پاسخ‌های خود را در درجه اول بر اساس اطلاعات بازیابی‌شده از اسناد ارائه دهید. "
    "اگر اطلاعات کافی برای پاسخ به سوال نیست، این موضوع را صریحا بیان کنید و با استفاده از دانش خود، پاسخ را تکمیل کنید."
)

@st.cache_resource
def load_index_and_docs():
    index = faiss.read_index("faiss_questions (4).index")
    documents = pd.read_csv("questions (4).csv")
    return index, documents

index, documents = load_index_and_docs()

@st.cache_resource
def load_model():
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    return model

model = load_model()

@st.cache_resource
def load_reranker():
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)  # Load reranker model
    return reranker

reranker = load_reranker()

@st.cache_resource
def load_llm():
    llm = ChatGroq(groq_api_key="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", model_name="Gemma2-9b-It")
    return llm

llm = load_llm()

def get_question_embeddings(question):
    sentences = [question]
    embeddings = model.encode(sentences, batch_size=12, max_length=512)['dense_vecs']
    return embeddings[0]

def search_questions(query, top_k=10):
    query_embedding = get_question_embeddings(query).astype('float16').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    if indices[0][0] == -1:
        return pd.DataFrame()
    results = documents.iloc[indices[0]]
    return results

def rerank_documents(query, documents_df):
    # Compute relevance scores for each document
    doc_pairs = [[query, doc] for doc in documents_df['title'].tolist()]
    scores = reranker.compute_score(doc_pairs, normalize=True)  # normalize to get score between 0-1
    documents_df['score'] = scores
    documents_df = documents_df.sort_values(by='score', ascending=False)  # Sort by score
    return documents_df.head(2)  # Return only the top 3 ranked documents

def count_tokens(messages):
    """Count the number of tokens in a list of messages."""
    tokens = 0
    for msg in messages:
        tokens += len(tokenizer.encode(msg.content))
    return tokens

def chatbot(user_question, conversation):
    relevant_questions = search_questions(user_question)
    if relevant_questions.empty:
        retrieved_answer = "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد."
        url = None
    else:
        original_documents = relevant_questions.copy()
        
        # Rerank the documents if rerank is enabled
        if st.session_state['rerank_enabled']:
            reranked_documents = rerank_documents(user_question, relevant_questions)
            if not reranked_documents.equals(original_documents):
                st.write("ترتیب اسناد به دلیل ریرنک تغییر کرده است.")
            retrieved_answers = [
                f"سند: {row['title']}\nلینک: {row['url']}" 
                for _, row in reranked_documents.iterrows()
            ]
            retrieved_answer = "\n---\n".join(retrieved_answers)
            url = reranked_documents['url'].iloc[0]
        else:
            retrieved_answers = [
                f"سند: {row['title']}\nلینک: {row['url']}" 
                for _, row in relevant_questions.iterrows()
            ]
            retrieved_answer = "\n---\n".join(retrieved_answers)
            url = relevant_questions['url'].iloc[0]

    # Store only the last retrieved_answer
    st.session_state['last_retrieved_answer'] = retrieved_answer

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if retrieved_answer != "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد.":
        messages.append(HumanMessage(content=f"اطلاعات بازیابی‌شده:\n{retrieved_answer}"))
    
    # Adding the conversation history to the messages
    for msg in conversation:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))
    
    messages.append(HumanMessage(content=user_question))

    # Count the tokens before sending the request
    num_tokens = count_tokens(messages)
    st.write(f"تعداد توکن‌ها: {num_tokens}")

    # Check if the token count exceeds 8000
    if num_tokens > 8000:
        st.warning("تعداد توکن‌ها از 8K بیشتر شده است!")

    # Send the messages to the model
    response = llm(messages=messages)
    return response.content, url

# Add a button to enable/disable rerank
rerank_button = st.button("فعال/غیرفعال کردن ریرنک", key="rerank_button")
if rerank_button:
    st.session_state['rerank_enabled'] = not st.session_state['rerank_enabled']
    st.write(f"ریرنک {'فعال' if st.session_state['rerank_enabled'] else 'غیرفعال'} شد.")

# Display previous messages (only the last two user inputs and responses)
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.write(msg['content'])
    else:
        with st.chat_message("assistant"):
            st.write(msg['content'])

user_question = st.chat_input("سوال خود را وارد کنید:")

if user_question and user_question.strip():
    # Add new user message
    st.session_state['messages'].append({"role": "user", "content": user_question})

    # Limit stored messages to only the last two user inputs and responses (total 4)
    if len(st.session_state['messages']) > 4:
        st.session_state['messages'] = st.session_state['messages'][-4:]

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("در حال پردازش..."):
            answer, url = chatbot(user_question, st.session_state['messages'][:-1])

            # Add new assistant message
            st.session_state['messages'].append({"role": "assistant", "content": url + "\n\n" + answer})

            # Limit stored messages again after adding assistant response
            if len(st.session_state['messages']) > 4:
                st.session_state['messages'] = st.session_state['messages'][-4:]

            if url:
                st.write(f"[لینک مرتبط به پاسخ]({url})")
            st.write("\n\n")
            st.write(answer)
