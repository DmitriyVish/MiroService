import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import os

# --- Настройки ---

INPUT_FILES = [
    "prompt_templates/results_question_1.xlsx",
    "prompt_templates/results_question_2.xlsx",
    "prompt_templates/results_question_3.xlsx",
    "prompt_templates/results_question_4.xlsx"
]
OUTPUT_FILE = "prompt_templates/evaluation_results.xlsx"

# --- Препроцессинг текста ---
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text.lower())  # Удаление пунктуации
    text = re.sub(r'\s+', ' ', text).strip()  # Нормализация пробелов
    return text

# --- Векторизация предложений ---
def get_embeddings(sentences: list, model: SentenceTransformer) -> np.ndarray:
    return model.encode(sentences)

# --- Сравнение по косинусному сходству ---
def compare_with_cosine(reference_emb: np.ndarray, candidate_emb: np.ndarray) -> float:
    similarity = cosine_similarity([reference_emb], [candidate_emb])[0][0]
    return round(similarity * 10, 1)  # Оценка от 0 до 10

# --- Генерация комментария ---
def generate_comment(score: float) -> str:
    if score >= 8.5:
        return "Ответ полностью соответствует эталону: точная аргументация, корректные термины, логичная структура."
    elif score >= 6.0:
        return "Ответ в целом соответствует эталону, но есть небольшие упущения или неточности в деталях."
    elif score >= 4.0:
        return "Частичное соответствие: основные идеи верны, но много неточностей или пропущены ключевые моменты."
    else:
        return "Низкое соответствие: ответ содержит существенные ошибки, пропуски или нелогичные рассуждения."

# --- Основная логика ---
def evaluate_all_questions(input_files: list, output_file: str):
    # Инициализация модели
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = []  # Для сбора всех результатов

    for file_idx, file_path in enumerate(input_files, start=1):
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}. Пропускаем.")
            continue

        print(f"\n--- Обработка вопроса {file_idx} ({file_path}) ---")

        # 1. Загрузка данных
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            print(f'!Ошибка при чтении файла {file_path}: {e}')
            continue

        # 2. Поиск эталонного ответа
        reference_row = df[df["Роль агента"] == "generative_ai_qwen"]
        if reference_row.empty:
            print(f'!В файле {file_path} не найден эталон (generative_ai_qwen). Пропускаем.')
            continue

        reference_answer = reference_row["Ответ"].values[0]
        reference_emb = get_embeddings([preprocess_text(reference_answer)], model)[0]

        # 3. Обработка ответов кандидатов
        for _, row in df.iterrows():
            role = row["Роль агента"]
            candidate_answer = row["Ответ"]

            if role == "generative_ai_qwen":
                continue  # Пропускаем эталон

            # Векторизация ответа кандидата
            candidate_emb = get_embeddings([preprocess_text(candidate_answer)], model)[0]

            # Сравнение с эталоном
            score = compare_with_cosine(reference_emb, candidate_emb)

            # Генерация комментария
            comment = generate_comment(score)

            # Сохранение результата
            results.append({
                "Номер вопроса": file_idx,
                "Роль": role,
                "Ответ": candidate_answer,
                "Комментарий": comment,
                "Оценка": score
            })

    # 4. Сохранение в Excel
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ Результаты сохранены в {output_file}")
    else:
        print("\n❌ Не удалось обработать ни одного вопроса. Проверьте входные файлы.")

# --- Запуск ---
if __name__ == "__main__":
    evaluate_all_questions(INPUT_FILES, OUTPUT_FILE)
