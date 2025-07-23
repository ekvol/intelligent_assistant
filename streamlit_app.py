
import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Настройки страницы
st.set_page_config(page_title="Интеллектуальный помощник", layout="centered")

# Загрузка модели
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.load_state_dict(torch.load("model/model.pt", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Генерация ответа
def generate_answer(question):
    input_text = f"translate Russian to English: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=32, truncation=True)
    output_ids = model.generate(input_ids, max_length=32)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Интерфейс
st.title("🎓 Интеллектуальный помощник для абитуриентов")
question = st.text_input("Введите ваш вопрос на русском языке:")

if question:
    with st.spinner("Генерация ответа..."):
        answer = generate_answer(question)
    st.success(f"Ответ: {answer}")
