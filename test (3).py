import streamlit as st
import faiss
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage

st.set_page_config(page_title="چت بات FAQ General", page_icon=":speech_balloon:", layout="centered")
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

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

SYSTEM_PROMPT = (
    "پاسخ‌های خود را در درجه اول بر اساس اطلاعات بازیابی‌شده از اسناد ارائه شده بنویسید. اما سعی کن دقیق سوال را پاسخ بدی "
    "اگر اطلاعات اسناد برای پاسخ کامل به سوال کافی نیست، این موضوع را صریحا بیان کنید و سپس با استفاده از دانش خود، پاسخ را تکمیل کنید."
    
    
)

@st.cache_resource
def load_index_and_docs():
    index = faiss.read_index("faiss_questions (3).index")
    documents = pd.read_csv("questions (3).csv")
    return index, documents

index, documents = load_index_and_docs()

@st.cache_resource
def load_model():
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    return model

model = load_model()

@st.cache_resource
def load_llm():
    llm = ChatGroq(groq_api_key="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", model_name="Gemma2-9b-It")
    return llm

llm = load_llm()

def get_question_embeddings(question):
    sentences = [question]
    embeddings = model.encode(sentences, batch_size=12, max_length=512)['dense_vecs']
    return embeddings[0]

def search_questions(query, top_k=9):
    query_embedding = get_question_embeddings(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    if indices[0][0] == -1:
        return pd.DataFrame()
    results = documents.iloc[indices[0]]
    return results

def chatbot(user_question, conversation):
    relevant_questions = search_questions(user_question)
    if relevant_questions.empty:
        retrieved_answer = "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد."
        url = None
    else:
        retrieved_answers = [
            f"سند: {row['title']}\nلینک: {row['url']}" 
            for _, row in relevant_questions.reset_index().iterrows()
        ]
        retrieved_answer = "\n---\n".join(retrieved_answers)
        url = relevant_questions.reset_index()['url'][0]  

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if retrieved_answer != "متأسفم، پاسخ مناسبی در دیتابیس پیدا نشد.":
        messages.append(HumanMessage(content=f"اطلاعات بازیابی‌شده:\n{retrieved_answer}"))
    
    for msg in conversation:
        if msg['role'] == 'user':
            messages.append(HumanMessage(content=msg['content']))
        else:
            messages.append(AIMessage(content=msg['content']))
    messages.append(HumanMessage(content=user_question))
    
    response = llm(messages=messages)
    return response.content, url

for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.write(msg['content'])
    else:
        with st.chat_message("assistant"):
            st.write(msg['content'])

user_question = st.chat_input("سوال خود را وارد کنید:")

if user_question and user_question.strip():
    st.session_state['messages'].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question) 


    with st.chat_message("assistant"):
        with st.spinner("در حال پردازش..."):
            answer, url = chatbot(user_question, st.session_state['messages'][:-1])
            st.session_state['messages'].append({"role": "assistant", "content":url+"\n\n"+ answer})
            if url:
                st.write(f"[لینک مرتبط به پاسخ]({url})")
            st.write("\n\n")
            st.write(answer)
