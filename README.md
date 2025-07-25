# Дообучение модели T5 для интеллектуального помощника абитуриентов

## Описание
Этот проект содержит пример кода для дообучения модели T5 (русская версия) на корпусе вопросов-ответов, связанных с поступлением в вуз.

## Файлы
- `train_t5.py` — скрипт дообучения модели
- `data.json` — пример обучающих данных в формате JSON
- `requirements.txt` — зависимости Python

## Инструкция по запуску
1. Установите зависимости:
```
pip install -r requirements.txt
```
2. Запустите скрипт обучения:
```
python train_t5.py
```
3. Модель и токенизатор сохранятся в папке `./t5_finetuned_model`.

## Настройка
- Для работы с другими корпусами замените файл `data.json`.
- Для использования другой модели измените переменную `model_name` в `train_t5.py`.
