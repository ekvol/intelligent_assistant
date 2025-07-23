from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch

app = Flask(__name__)

# Загрузка моделей и токенизаторов
t5_model_name = "t5-base"  # или путь к дообученной модели
roberta_model_name = "roberta-base"

t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model = t5_model.to(device)
roberta_model = roberta_model.to(device)

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Empty question"}), 400
    
    # Классификация интента через RoBERTa
    inputs = roberta_tokenizer(question, return_tensors="pt").to(device)
    outputs = roberta_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    # Здесь можешь расширить маппинг классов и логику
    intent_map = {
        0: "Условия поступления",
        1: "Стоимость обучения",
        2: "Документы",
        3: "Сроки подачи",
        4: "Льготы",
        5: "Общая информация"
    }
    intent = intent_map.get(predicted_class_id, "Неизвестный интент")

    # Генерация ответа через T5
    input_text = f"question: {question} intent: {intent}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = t5_model.generate(input_ids, max_length=100)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"intent": intent, "answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)