import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# --- Загрузка данных ---
def load_data(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    # Оставляем только нужные колонки
    df = df[["Роль агента", "Ответ"]].copy()
    return df

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
def generate_comment(score: float, reference: str, candidate: str) -> str:
    if score >= 8.5:
        return "Ответ полностью соответствует эталону: точная аргументация, корректные термины, логичная структура."
    elif score >= 6.0:
        return "Ответ в целом соответствует эталону, но есть небольшие упущения или неточности в деталях."
    elif score >= 4.0:
        return "Частичное соответствие: основные идеи верны, но много неточностей или пропущены ключевые моменты."
    else:
        return "Низкое соответствие: ответ содержит существенные ошибки, пропуски или нелогичные рассуждения."

# --- Основная логика ---
def evaluate_responses(excel_path: str, output_path: str):
    # 1. Загрузка данных
    df = load_data(excel_path)
    
    # 2. Поиск эталонного ответа (generative_ai_qwen)
    reference_row = df[df["Роль агента"] == "generative_ai_qwen"]
    if reference_row.empty:
        raise ValueError("Не найден эталонный ответ (generative_ai_qwen)")
    
    reference_answer = reference_row["Ответ"].values[0]
    
    # 3. Подготовка модели для векторизации
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Лёгкая модель для семантического сходства
    
    # 4. Векторизация эталонного ответа
    reference_emb = get_embeddings([preprocess_text(reference_answer)], model)[0]
    
    # 5. Подготовка результата
    results = []
    
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
        comment = generate_comment(score, reference_answer, candidate_answer)

        results.append({
            "Номер вопроса": 4,  # Если вопросов несколько — можно добавить цикл
            "Роль": role,
            "Ответ": candidate_answer,
            "Комментарий": comment,
            "Оценка": score
        })
    
    # 6. Сохранение в Excel
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Результаты сохранены в {output_path}")

# --- Запуск ---
if __name__ == "__main__":
    evaluate_responses(
        excel_path="prompt_templates/results_question_4.xlsx",
        output_path="prompt_templates/evaluation_results.xlsx"
    )
