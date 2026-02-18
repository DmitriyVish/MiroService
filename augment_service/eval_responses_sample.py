import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --- Настройки ---
INPUT_FILE = "prompt_templates/generat_responses/full.json"
OUTPUT_FILE = "prompt_templates/eval_result_sample/evaluation_results_v3.xlsx"

REFERENCE_ROLE = "generative_ai_qwen"  # эталон
QUESTION_KEY = "Вопрос номер"
ANSWER_KEY = "ответ"
ROLE_KEY = "роль_агента"

# --- Калибровка cosine -> score (настрой под свои данные) ---
COSINE_LO = 0.72   # ниже этого считаем "похоже слабо" -> ~1
COSINE_HI = 0.92   # выше этого считаем "очень похоже" -> ~10

# --- Сигналы качества (простые эвристики) ---
UNCERTAINTY_PATTERNS = [
    r"\bне уверен\b", r"\bне уверена\b", r"\bне уверен(а)?\b",
    r"\bне знаю\b", r"\bне помню\b", r"\bзатрудняюсь\b",
    r"\bне эксперт\b", r"\bне специалист\b", r"\bмогу ошибаться\b",
    r"\bвозможно\b", r"\bскорее всего\b", r"\bпредположим\b",
]
CONCRETENESS_PATTERNS = [
    r"\bforeign key\b", r"\bfk\b", r"\bуникал", r"\bunique\b",
    r"\bprimary key\b", r"\bpk\b", r"\bindex\b", r"\bиндекс\b",
    r"\bconstraint\b", r"\bnot null\b", r"\bcheck\b",
    r"\bтип\b", r"\bint\b", r"\bdate\b", r"\btimestamp\b",
]


# --- Препроцессинг текста ---
def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)     # пунктуация -> пробел
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Векторизация ---
def get_embeddings(sentences: List[str], model: SentenceTransformer) -> np.ndarray:
    # normalize_embeddings=True => cosine = dot product; стабильнее
    return model.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # при normalize_embeddings=True это просто скалярное произведение
    return float(np.dot(a, b))


def cosine_to_score(sim: float, lo: float = COSINE_LO, hi: float = COSINE_HI) -> float:
    """
    Калибруем cosine в оценку 1..10.
    sim<=lo -> 1, sim>=hi -> 10.
    """
    if hi <= lo:
        # защита от неверной настройки
        lo, hi = min(lo, hi), max(lo, hi) + 1e-6

    x = (sim - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    score = 1.0 + 9.0 * x
    return round(score, 1)


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    hits = 0
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            hits += 1
    return hits


def apply_quality_adjustments(base_score: float, raw_text: str) -> float:
    """
    Лёгкая корректировка оценки:
    - штраф за неуверенность
    - небольшой бонус за конкретику
    """
    t = raw_text or ""
    unc = count_pattern_hits(t, UNCERTAINTY_PATTERNS)
    conc = count_pattern_hits(t, CONCRETENESS_PATTERNS)

    score = base_score
    score -= 0.6 * min(unc, 3)     # до -1.8
    score += 0.2 * min(conc, 5)    # до +1.0

    score = max(1.0, min(10.0, score))
    return round(score, 1)


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


def normalize_question_id(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON должен быть списком объектов (list[dict]).")
    return [x for x in data if isinstance(x, dict)]


def find_reference_answer(records: List[Dict[str, Any]]) -> Optional[str]:
    for r in records:
        if r.get(ROLE_KEY) == REFERENCE_ROLE and r.get(ANSWER_KEY):
            return r.get(ANSWER_KEY)
    return None


def evaluate_json_file(input_file: str, output_file: str) -> None:
    # 1) Загрузка JSON
    try:
        records = load_json_records(input_file)
    except Exception as e:
        print(f"!Ошибка при чтении JSON {input_file}: {e}")
        return

    if not records:
        print("В JSON нет записей для обработки.")
        return

    # 2) Группировка по вопросам
    by_question: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        qid = normalize_question_id(r.get(QUESTION_KEY))
        if not qid:
            continue
        by_question.setdefault(qid, []).append(r)

    if not by_question:
        print(f"Не найдено ни одной записи с ключом '{QUESTION_KEY}'.")
        return

    # 3) Инициализация модели
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results: List[Dict[str, Any]] = []

    # 4) Обработка каждого вопроса
    def q_sort_key(x: str):
        return int(x) if str(x).isdigit() else str(x)

    for qid in sorted(by_question.keys(), key=q_sort_key):
        q_records = by_question[qid]

        reference_answer = find_reference_answer(q_records)
        if not reference_answer:
            print(f"!Для вопроса {qid} не найден эталон (роль '{REFERENCE_ROLE}'). Пропускаем вопрос.")
            continue

        ref_txt = preprocess_text(reference_answer)
        reference_emb = get_embeddings([ref_txt], model)[0]

        for r in q_records:
            role = r.get(ROLE_KEY, "")
            candidate_answer = r.get(ANSWER_KEY, "")

            if role == REFERENCE_ROLE:
                continue
            if not candidate_answer:
                continue

            cand_txt = preprocess_text(candidate_answer)
            candidate_emb = get_embeddings([cand_txt], model)[0]

            raw_cos = cosine_sim(reference_emb, candidate_emb)
            base_score = cosine_to_score(raw_cos)
            final_score = apply_quality_adjustments(base_score, candidate_answer)
            comment = generate_comment(final_score)

            results.append(
                {
                    "Номер вопроса": qid,
                    "Роль": role,
                    "Ответ": candidate_answer,
                    "Cosine (raw)": round(raw_cos, 4),
                    "Оценка (base)": base_score,
                    "Оценка (final)": final_score,
                    "Комментарий": comment,
                }
            )

    # 5) Сохранение
    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"Результаты сохранены в {output_file}")
        print(f"Пороговая калибровка: LO={COSINE_LO}, HI={COSINE_HI}")
    else:
        print("Нет результатов: проверь роли, наличие эталона и ключи JSON.")


if __name__ == "__main__":
    evaluate_json_file(INPUT_FILE, OUTPUT_FILE)