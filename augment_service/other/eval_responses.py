# import json
# import re
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from sentence_transformers import SentenceTransformer


# # ---------------------------
# # Настройки
# # ---------------------------
# CANDIDATE_FILE = "prompt_templates/eval_result_parser/candidat_1.json"
# ETALON_HR_FILE = "prompt_templates/interview_questions/etalon_hr.json"
# ETALON_AI_FILE = "prompt_templates/interview_questions/etalon_ai.json"
# OUTPUT_FILE = "prompt_templates/evaluation_results_v3.json"

# # Калибровка cosine -> score (настройте под данные)
# COSINE_LO = 0.72
# COSINE_HI = 0.92

# # Эвристики качества
# UNCERTAINTY_PATTERNS = [
#     r"\bне уверен\b", r"\bне уверена\b", r"\bне уверен(а)?\b",
#     r"\bне знаю\b", r"\bне помню\b", r"\bзатрудняюсь\b",
#     r"\bне эксперт\b", r"\bне специалист\b", r"\bмогу ошибаться\b",
#     r"\bвозможно\b", r"\bскорее всего\b", r"\bпредположим\b",
# ]

# CONCRETENESS_PATTERNS = [
#     r"\bforeign key\b", r"\bfk\b", r"\bуникал", r"\bunique\b",
#     r"\bprimary key\b", r"\bpk\b", r"\bindex\b", r"\bиндекс\b",
#     r"\bconstraint\b", r"\bnot null\b", r"\bcheck\b",
#     r"\bтип\b", r"\bint\b", r"\bdate\b", r"\btimestamp\b",
# ]


# # ---------------------------
# # Утилиты
# # ---------------------------
# def load_json(path: str) -> Any:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def save_json(path: str, data: Any) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)


# def preprocess_text(text: Any) -> str:
#     if text is None:
#         return ""
#     text = str(text).lower()
#     text = re.sub(r"[^\w\s]", " ", text)  # пунктуация -> пробел
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def normalize_question_id(raw_key: Any) -> Optional[str]:
#     """
#     Принимает ключи вида: "Вопрос 1", "Вопрос №2", "1", и т.п.
#     Возвращает "1", "2", ...
#     """
#     if raw_key is None:
#         return None
#     s = str(raw_key).strip()
#     m = re.search(r"(\d+)", s)
#     return m.group(1) if m else None


# def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#     # при normalize_embeddings=True это скалярное произведение
#     return float(np.dot(a, b))


# def cosine_to_score(sim: float, lo: float = COSINE_LO, hi: float = COSINE_HI) -> float:
#     """
#     Калибруем cosine в оценку 1..10.
#     sim<=lo -> 1, sim>=hi -> 10.
#     """
#     if hi <= lo:
#         lo, hi = min(lo, hi), max(lo, hi) + 1e-6

#     x = (sim - lo) / (hi - lo)
#     x = max(0.0, min(1.0, x))
#     score = 1.0 + 9.0 * x
#     return round(score, 1)


# def count_pattern_hits(text: str, patterns: List[str]) -> int:
#     hits = 0
#     for p in patterns:
#         if re.search(p, text, flags=re.IGNORECASE):
#             hits += 1
#     return hits


# def apply_quality_adjustments(base_score: float, raw_text: str) -> float:
#     """
#     Лёгкая корректировка оценки:
#     - штраф за неуверенность
#     - небольшой бонус за конкретику
#     """
#     t = raw_text or ""
#     unc = count_pattern_hits(t, UNCERTAINTY_PATTERNS)
#     conc = count_pattern_hits(t, CONCRETENESS_PATTERNS)

#     score = base_score
#     score -= 0.6 * min(unc, 3)   # до -1.8
#     score += 0.2 * min(conc, 5)  # до +1.0

#     score = max(1.0, min(10.0, score))
#     return round(score, 1)


# def generate_comment(score: float) -> str:
#     if score >= 8.5:
#         return "Ответ полностью соответствует эталону: точная аргументация, корректные термины, логичная структура."
#     elif score >= 6.0:
#         return "Ответ в целом соответствует эталону, но есть небольшие упущения или неточности в деталях."
#     elif score >= 4.0:
#         return "Частичное соответствие: основные идеи верны, но много неточностей или пропущены ключевые моменты."
#     else:
#         return "Низкое соответствие: ответ содержит существенные ошибки, пропуски или нелогичные рассуждения."


# def ai_etalon_to_text(ai_obj: Any) -> str:
#     """
#     Превращает структуру etalon_ai.json по одному вопросу в текст для семантического сравнения.
#     Поддерживает dict/list/str.
#     """
#     if ai_obj is None:
#         return ""
#     if isinstance(ai_obj, str):
#         return ai_obj

#     lines: List[str] = []

#     def emit(prefix: str, v: Any) -> None:
#         if v is None:
#             return
#         if isinstance(v, str):
#             if v.strip():
#                 lines.append(f"{prefix}{v.strip()}")
#         elif isinstance(v, (int, float, bool)):
#             lines.append(f"{prefix}{v}")
#         elif isinstance(v, list):
#             for item in v:
#                 emit(prefix + "- ", item)
#         elif isinstance(v, dict):
#             for k, vv in v.items():
#                 key = str(k)
#                 if isinstance(vv, (str, int, float, bool)) and str(vv).strip():
#                     lines.append(f"{key}: {vv}")
#                 else:
#                     lines.append(f"{key}:")
#                     emit("  ", vv)
#         else:
#             lines.append(f"{prefix}{str(v)}")

#     emit("", ai_obj)
#     return "\n".join(lines).strip()


# def get_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
#     return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# def pick_final(base_hr: float, base_ai: float) -> Tuple[float, str]:
#     """
#     Как собрать финальную оценку из 2 эталонов:
#     - берём максимум (кандидат близок хотя бы к одному эталону)
#     Можно заменить на среднее, если нужно:
#       final = 0.5*(base_hr+base_ai)
#     """
#     if base_hr >= base_ai:
#         return base_hr, "hr"
#     return base_ai, "ai"


# # ---------------------------
# # Основная логика
# # ---------------------------
# def evaluate(candidate_path: str, hr_path: str, ai_path: str, output_path: str) -> None:
#     cand = load_json(candidate_path)
#     et_hr = load_json(hr_path)
#     et_ai = load_json(ai_path)

#     if not isinstance(cand, dict):
#         raise ValueError("Файл кандидата должен быть JSON-объектом (dict) вида {'Вопрос 1': '...', ...}.")
#     if not isinstance(et_hr, dict):
#         raise ValueError("etalon_hr.json должен быть dict.")
#     if not isinstance(et_ai, dict):
#         raise ValueError("etalon_ai.json должен быть dict.")

#     # Индексация по номеру вопроса
#     cand_by_q: Dict[str, str] = {}
#     for k, v in cand.items():
#         qid = normalize_question_id(k)
#         if qid:
#             cand_by_q[qid] = "" if v is None else str(v)

#     hr_by_q: Dict[str, str] = {}
#     for k, v in et_hr.items():
#         qid = normalize_question_id(k)
#         if qid:
#             hr_by_q[qid] = "" if v is None else str(v)

#     ai_by_q: Dict[str, str] = {}
#     for k, v in et_ai.items():
#         qid = normalize_question_id(k)
#         if qid:
#             ai_by_q[qid] = ai_etalon_to_text(v)

#     all_qids = sorted(
#         set(cand_by_q.keys()) | set(hr_by_q.keys()) | set(ai_by_q.keys()),
#         key=lambda x: int(x) if str(x).isdigit() else str(x),
#     )

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Предварительно посчитаем эмбеддинги для эталонов (чтобы не дёргать модель лишний раз)
#     hr_texts = {qid: preprocess_text(txt) for qid, txt in hr_by_q.items()}
#     ai_texts = {qid: preprocess_text(txt) for qid, txt in ai_by_q.items()}

#     # батчинг
#     hr_qids = [qid for qid in all_qids if qid in hr_texts and hr_texts[qid]]
#     ai_qids = [qid for qid in all_qids if qid in ai_texts and ai_texts[qid]]

#     hr_embs = {}
#     if hr_qids:
#         embs = get_embeddings([hr_texts[qid] for qid in hr_qids], model)
#         hr_embs = {qid: embs[i] for i, qid in enumerate(hr_qids)}

#     ai_embs = {}
#     if ai_qids:
#         embs = get_embeddings([ai_texts[qid] for qid in ai_qids], model)
#         ai_embs = {qid: embs[i] for i, qid in enumerate(ai_qids)}

#     results: List[Dict[str, Any]] = []
#     meta_warnings: List[str] = []

#     for qid in all_qids:
#         cand_raw = cand_by_q.get(qid, "")
#         if not cand_raw:
#             meta_warnings.append(f"Вопрос {qid}: у кандидата нет ответа или пусто.")

#         cand_txt = preprocess_text(cand_raw)
#         cand_emb = get_embeddings([cand_txt], model)[0] if cand_txt else None

#         hr_sim = None
#         hr_score_base = None
#         if cand_emb is not None and qid in hr_embs:
#             hr_sim = round(cosine_sim(hr_embs[qid], cand_emb), 4)
#             hr_score_base = cosine_to_score(hr_sim)

#         ai_sim = None
#         ai_score_base = None
#         if cand_emb is not None and qid in ai_embs:
#             ai_sim = round(cosine_sim(ai_embs[qid], cand_emb), 4)
#             ai_score_base = cosine_to_score(ai_sim)

#         if hr_score_base is None and ai_score_base is None:
#             # Нет эталонов/кандидата
#             final_base = None
#             final_score = None
#             comment = "Нет данных для оценки (нет эталона и/или ответа кандидата)."
#             chosen = None
#         else:
#             # Если один из эталонов отсутствует — используем другой
#             _hr = hr_score_base if hr_score_base is not None else -1e9
#             _ai = ai_score_base if ai_score_base is not None else -1e9
#             final_base, chosen = pick_final(_hr, _ai)
#             final_score = apply_quality_adjustments(final_base, cand_raw)
#             comment = generate_comment(final_score)

#         results.append(
#             {
#                 "Номер вопроса": qid,
#                 "Ответ кандидата": cand_raw,
#                 "Cosine HR": hr_sim,
#                 "Оценка HR (base)": hr_score_base,
#                 "Cosine AI": ai_sim,
#                 "Оценка AI (base)": ai_score_base,
#                 "Эталон выбран": chosen,          # "hr" / "ai" / null
#                 "Оценка (base)": final_base,
#                 "Оценка (final)": final_score,
#                 "Комментарий": comment,
#             }
#         )

#     out = {
#         "input": {
#             "candidate_file": candidate_path,
#             "etalon_hr_file": hr_path,
#             "etalon_ai_file": ai_path,
#         },
#         "calibration": {
#             "cosine_lo": COSINE_LO,
#             "cosine_hi": COSINE_HI,
#             "final_policy": "max_of_two_etalon_scores_then_quality_adjustments",
#         },
#         "warnings": meta_warnings,
#         "results": results,
#     }
#     save_json(output_path, out)
#     print(f"OK: сохранено в {output_path}")


# if __name__ == "__main__":
#     evaluate(CANDIDATE_FILE, ETALON_HR_FILE, ETALON_AI_FILE, OUTPUT_FILE)

#  #Вариант2
# import json
# import re
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# from sentence_transformers import SentenceTransformer


# # ---------------------------
# # Настройки
# # ---------------------------
# CANDIDATE_FILE = "prompt_templates/parser_responses/candidat_1.json"
# ETALON_HR_FILE = "prompt_templates/interview_questions/etalon_hr.json"
# ETALON_AI_FILE = "prompt_templates/interview_questions/etalon_ai.json"
# OUTPUT_FILE = "prompt_templates/eval_result_parser/evaluation_results_v3.json"

# # Если True — подберёт lo/hi по распределению cosine на ваших данных (рекомендуется)
# AUTO_CALIBRATE = True
# CALIBRATE_LO_Q = 0.20   # 20-й перцентиль
# CALIBRATE_HI_Q = 0.90   # 90-й перцентиль
# MIN_CALIBRATION_SAMPLES = 8  # минимум пар cosine, чтобы автокалибровка включилась

# # Ручные lo/hi используются, если AUTO_CALIBRATE=False или мало данных
# COSINE_LO = 0.72
# COSINE_HI = 0.92

# # Мягкая шкала (логистическая): чем меньше k, тем более “плавная” середина
# SIGMOID_K = 7.5

# # Эвристики качества (неуверенность)
# UNCERTAINTY_PATTERNS = [
#     r"\bне уверен\b", r"\bне уверена\b", r"\bне уверен(а)?\b",
#     r"\bне знаю\b", r"\bне помню\b", r"\bзатрудняюсь\b",
#     r"\bне эксперт\b", r"\bне специалист\b", r"\bмогу ошибаться\b",
#     r"\bвозможно\b", r"\bскорее всего\b", r"\bпредположим\b",
# ]


# # Конкретика — теперь по категориям (чтобы не “набивать” однотипными токенами)
# CONCRETENESS_CATEGORIES: Dict[str, List[str]] = {
#     "keys": [
#         r"\bprimary key\b", r"\bpk\b", r"\bforeign key\b", r"\bfk\b", r"\breference(s)?\b",
#     ],
#     "constraints": [
#         r"\bconstraint\b", r"\bnot null\b", r"\bcheck\b", r"\bunique\b", r"\bуникал\w*\b",
#     ],
#     "indexes": [
#         r"\bindex\b", r"\bиндекс\w*\b", r"\bbtree\b", r"\bhash\b",
#     ],
#     "types": [
#         r"\bint\b", r"\binteger\b", r"\bbigint\b", r"\bsmallint\b",
#         r"\bdate\b", r"\btimestamp\b", r"\bdatetime\b", r"\bboolean\b",
#         r"\btext\b", r"\bvarchar\b", r"\bchar\b", r"\bjsonb?\b",
#         r"\bnumeric\b", r"\bdecimal\b", r"\bfloat\b", r"\bdouble\b",
#     ],
#     "normalization": [
#         r"\bнормализац\w*\b", r"\b1нф\b", r"\b2нф\b", r"\b3нф\b",
#         r"\bденормализац\w*\b",
#     ],
# }


# # ---------------------------
# # Утилиты
# # ---------------------------
# def load_json(path: str) -> Any:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def save_json(path: str, data: Any) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)


# def preprocess_text(text: Any) -> str:
#     if text is None:
#         return ""
#     text = str(text).lower()
#     text = re.sub(r"[^\w\s]", " ", text)  # пунктуация -> пробел
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def normalize_question_id(raw_key: Any) -> Optional[str]:
#     if raw_key is None:
#         return None
#     s = str(raw_key).strip()
#     m = re.search(r"(\d+)", s)
#     return m.group(1) if m else None


# def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
#     # при normalize_embeddings=True это скалярное произведение
#     return float(np.dot(a, b))


# def cosine_to_score_soft(sim: float, lo: float, hi: float, k: float = SIGMOID_K) -> float:
#     """
#     Мягкая калибровка cosine -> 1..10:
#     - lo/hi задают “плохую” и “отличную” зоны
#     - вместо жёсткого клипа — логистическая S-кривая
#     """
#     if hi <= lo:
#         lo, hi = min(lo, hi), max(lo, hi) + 1e-6

#     # нормируем в 0..1
#     z = (sim - lo) / (hi - lo)
#     # логистическая: z=0 -> ~0.5? нет, поэтому сдвигаем центр на 0.5
#     # чтобы z=0.5 давал около середины шкалы
#     s = 1.0 / (1.0 + np.exp(-k * (z - 0.5)))
#     score = 1.0 + 9.0 * float(s)

#     # Гарантируем диапазон
#     score = max(1.0, min(10.0, score))
#     return round(score, 1)


# def count_pattern_hits(text: str, patterns: List[str]) -> int:
#     hits = 0
#     for p in patterns:
#         if re.search(p, text, flags=re.IGNORECASE):
#             hits += 1
#     return hits


# def concreteness_coverage(text: str) -> int:
#     """
#     Считает, сколько разных категорий конкретики покрыто (0..len(categories)).
#     Это менее “играемо”, чем просто количество совпадений.
#     """
#     t = preprocess_text(text)
#     covered = 0
#     for _, pats in CONCRETENESS_CATEGORIES.items():
#         if any(re.search(p, t, flags=re.IGNORECASE) for p in pats):
#             covered += 1
#     return covered


# def apply_quality_adjustments(base_score: float, raw_text: str) -> float:
#     """
#     Корректировка:
#     - штраф за неуверенность (сильнее, но ограничен)
#     - бонус за покрытие различных аспектов (категорий), не за “спам” терминов
#     """
#     t = preprocess_text(raw_text or "")

#     unc = count_pattern_hits(t, UNCERTAINTY_PATTERNS)          # 0..n
#     cov = concreteness_coverage(t)                              # 0..5

#     score = float(base_score)

#     # Неуверенность: до -2.0
#     score -= 0.7 * min(unc, 3)

#     # Конкретика-категории: до +1.0 (5 категорий -> +1.0)
#     score += 0.2 * min(cov, 5)

#     score = max(1.0, min(10.0, score))
#     return round(score, 1)


# def generate_comment(score: float) -> str:
#     if score >= 8.5:
#         return "Ответ полностью соответствует эталону: точная аргументация, корректные термины, логичная структура."
#     elif score >= 6.0:
#         return "Ответ в целом соответствует эталону, но есть небольшие упущения или неточности в деталях."
#     elif score >= 4.0:
#         return "Частичное соответствие: основные идеи верны, но много неточностей или пропущены ключевые моменты."
#     else:
#         return "Низкое соответствие: ответ содержит существенные ошибки, пропуски или нелогичные рассуждения."


# def ai_etalon_to_text(ai_obj: Any) -> str:
#     if ai_obj is None:
#         return ""
#     if isinstance(ai_obj, str):
#         return ai_obj

#     lines: List[str] = []

#     def emit(prefix: str, v: Any) -> None:
#         if v is None:
#             return
#         if isinstance(v, str):
#             if v.strip():
#                 lines.append(f"{prefix}{v.strip()}")
#         elif isinstance(v, (int, float, bool)):
#             lines.append(f"{prefix}{v}")
#         elif isinstance(v, list):
#             for item in v:
#                 emit(prefix + "- ", item)
#         elif isinstance(v, dict):
#             for k, vv in v.items():
#                 key = str(k)
#                 if isinstance(vv, (str, int, float, bool)) and str(vv).strip():
#                     lines.append(f"{key}: {vv}")
#                 else:
#                     lines.append(f"{key}:")
#                     emit("  ", vv)
#         else:
#             lines.append(f"{prefix}{str(v)}")

#     emit("", ai_obj)
#     return "\n".join(lines).strip()


# def get_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
#     return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# def pick_final(base_hr: float, base_ai: float) -> Tuple[float, str]:
#     if base_hr >= base_ai:
#         return base_hr, "hr"
#     return base_ai, "ai"


# def auto_calibrate_thresholds(
#     cand_by_q: Dict[str, str],
#     hr_by_q: Dict[str, str],
#     ai_by_q: Dict[str, str],
#     model: SentenceTransformer,
#     lo_fallback: float,
#     hi_fallback: float,
# ) -> Tuple[float, float, List[float]]:
#     """
#     Быстрая автокалибровка lo/hi по распределению cosine на имеющихся данных.
#     Берём cosine(candidate, hr/ai) там, где есть и ответ, и эталон.
#     """
#     sims: List[float] = []

#     for qid, cand_raw in cand_by_q.items():
#         cand_txt = preprocess_text(cand_raw)
#         if not cand_txt:
#             continue
#         cand_emb = get_embeddings([cand_txt], model)[0]

#         if qid in hr_by_q and preprocess_text(hr_by_q[qid]):
#             hr_emb = get_embeddings([preprocess_text(hr_by_q[qid])], model)[0]
#             sims.append(cosine_sim(hr_emb, cand_emb))

#         if qid in ai_by_q and preprocess_text(ai_by_q[qid]):
#             ai_emb = get_embeddings([preprocess_text(ai_by_q[qid])], model)[0]
#             sims.append(cosine_sim(ai_emb, cand_emb))

#     if len(sims) < MIN_CALIBRATION_SAMPLES:
#         return lo_fallback, hi_fallback, sims

#     lo = float(np.quantile(sims, CALIBRATE_LO_Q))
#     hi = float(np.quantile(sims, CALIBRATE_HI_Q))

#     # страховка от “слишком узко”
#     if hi - lo < 0.06:
#         mid = 0.5 * (hi + lo)
#         lo = mid - 0.03
#         hi = mid + 0.03

#     return lo, hi, sims


# # ---------------------------
# # Основная логика
# # ---------------------------
# def evaluate(candidate_path: str, hr_path: str, ai_path: str, output_path: str) -> None:
#     cand = load_json(candidate_path)
#     et_hr = load_json(hr_path)
#     et_ai = load_json(ai_path)

#     if not isinstance(cand, dict):
#         raise ValueError("Файл кандидата должен быть JSON-объектом (dict) вида {'Вопрос 1': '...', ...}.")
#     if not isinstance(et_hr, dict):
#         raise ValueError("etalon_hr.json должен быть dict.")
#     if not isinstance(et_ai, dict):
#         raise ValueError("etalon_ai.json должен быть dict.")

#     # Индексация по номеру вопроса
#     cand_by_q: Dict[str, str] = {}
#     for k, v in cand.items():
#         qid = normalize_question_id(k)
#         if qid:
#             cand_by_q[qid] = "" if v is None else str(v)

#     hr_by_q: Dict[str, str] = {}
#     for k, v in et_hr.items():
#         qid = normalize_question_id(k)
#         if qid:
#             hr_by_q[qid] = "" if v is None else str(v)

#     ai_by_q: Dict[str, str] = {}
#     for k, v in et_ai.items():
#         qid = normalize_question_id(k)
#         if qid:
#             ai_by_q[qid] = ai_etalon_to_text(v)

#     all_qids = sorted(
#         set(cand_by_q.keys()) | set(hr_by_q.keys()) | set(ai_by_q.keys()),
#         key=lambda x: int(x) if str(x).isdigit() else str(x),
#     )

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # (опционально) автокалибровка lo/hi
#     used_lo, used_hi = COSINE_LO, COSINE_HI
#     calib_sims: List[float] = []
#     if AUTO_CALIBRATE:
#         used_lo, used_hi, calib_sims = auto_calibrate_thresholds(
#             cand_by_q=cand_by_q,
#             hr_by_q=hr_by_q,
#             ai_by_q=ai_by_q,
#             model=model,
#             lo_fallback=COSINE_LO,
#             hi_fallback=COSINE_HI,
#         )

#     # Предварительно эмбеддинги эталонов
#     hr_texts = {qid: preprocess_text(txt) for qid, txt in hr_by_q.items()}
#     ai_texts = {qid: preprocess_text(txt) for qid, txt in ai_by_q.items()}

#     hr_qids = [qid for qid in all_qids if qid in hr_texts and hr_texts[qid]]
#     ai_qids = [qid for qid in all_qids if qid in ai_texts and ai_texts[qid]]

#     hr_embs = {}
#     if hr_qids:
#         embs = get_embeddings([hr_texts[qid] for qid in hr_qids], model)
#         hr_embs = {qid: embs[i] for i, qid in enumerate(hr_qids)}

#     ai_embs = {}
#     if ai_qids:
#         embs = get_embeddings([ai_texts[qid] for qid in ai_qids], model)
#         ai_embs = {qid: embs[i] for i, qid in enumerate(ai_qids)}

#     results: List[Dict[str, Any]] = []
#     meta_warnings: List[str] = []

#     for qid in all_qids:
#         cand_raw = cand_by_q.get(qid, "")
#         if not cand_raw:
#             meta_warnings.append(f"Вопрос {qid}: у кандидата нет ответа или пусто.")

#         cand_txt = preprocess_text(cand_raw)
#         cand_emb = get_embeddings([cand_txt], model)[0] if cand_txt else None

#         hr_sim = None
#         hr_score_base = None
#         if cand_emb is not None and qid in hr_embs:
#             hr_sim = round(cosine_sim(hr_embs[qid], cand_emb), 4)
#             hr_score_base = cosine_to_score_soft(hr_sim, lo=used_lo, hi=used_hi)

#         ai_sim = None
#         ai_score_base = None
#         if cand_emb is not None and qid in ai_embs:
#             ai_sim = round(cosine_sim(ai_embs[qid], cand_emb), 4)
#             ai_score_base = cosine_to_score_soft(ai_sim, lo=used_lo, hi=used_hi)

#         if hr_score_base is None and ai_score_base is None:
#             final_base = None
#             final_score = None
#             comment = "Нет данных для оценки (нет эталона и/или ответа кандидата)."
#             chosen = None
#         else:
#             _hr = hr_score_base if hr_score_base is not None else -1e9
#             _ai = ai_score_base if ai_score_base is not None else -1e9
#             final_base, chosen = pick_final(_hr, _ai)
#             final_score = apply_quality_adjustments(final_base, cand_raw)
#             comment = generate_comment(final_score)

#         results.append(
#             {
#                 "Номер вопроса": qid,
#                 "Ответ кандидата": cand_raw,
#                 "Cosine HR": hr_sim,
#                 "Оценка HR (base)": hr_score_base,
#                 "Cosine AI": ai_sim,
#                 "Оценка AI (base)": ai_score_base,
#                 "Эталон выбран": chosen,
#                 "Оценка (base)": final_base,
#                 "Оценка (final)": final_score,
#                 "Комментарий": comment,
#             }
#         )

#     out = {
#         "input": {
#             "candidate_file": candidate_path,
#             "etalon_hr_file": hr_path,
#             "etalon_ai_file": ai_path,
#         },
#         "calibration": {
#             "auto_calibrate": AUTO_CALIBRATE,
#             "cosine_lo": used_lo,
#             "cosine_hi": used_hi,
#             "sigmoid_k": SIGMOID_K,
#             "final_policy": "max_of_two_etalon_scores_then_quality_adjustments",
#             "calibration_samples": len(calib_sims),
#         },
#         "warnings": meta_warnings,
#         "results": results,
#     }
#     save_json(output_path, out)
#     print(f"OK: сохранено в {output_path}")


# if __name__ == "__main__":
#     evaluate(CANDIDATE_FILE, ETALON_HR_FILE, ETALON_AI_FILE, OUTPUT_FILE)

# Вариант 3
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------
# Настройки
# ---------------------------
CANDIDATE_FILE = "prompt_templates/parser_responses/candidat_1.json"
ETALON_HR_FILE = "prompt_templates/interview_questions/etalon_hr.json"
ETALON_AI_FILE = "prompt_templates/interview_questions/etalon_ai.json"
OUTPUT_FILE = "prompt_templates/eval_result_parser/evaluation_results_v4.json"

# Рекомендация: автокалибровку выключаем полностью, чтобы оценка не “плавала”
AUTO_CALIBRATE = False

# Фиксированная линейная шкала cosine -> 0..10 (потом клип в 1..10)
# Ниже lo — оценка тяготеет к 0 (после клипа станет 1), выше hi — к 10.
COSINE_LO = 0.45
COSINE_HI = 0.90

# Эвристики качества (неуверенность)
UNCERTAINTY_PATTERNS = [
    r"\bне уверен\b", r"\bне уверена\b", r"\bне уверен(а)?\b",
    r"\bне знаю\b", r"\bне помню\b", r"\bзатрудняюсь\b",
    r"\bне эксперт\b", r"\bне специалист\b", r"\bмогу ошибаться\b",
    r"\bвозможно\b", r"\bскорее всего\b", r"\bпредположим\b",
]

# Конкретика — по категориям
CONCRETENESS_CATEGORIES: Dict[str, List[str]] = {
    "keys": [
        r"\bprimary key\b", r"\bpk\b", r"\bforeign key\b", r"\bfk\b", r"\breference(s)?\b",
    ],
    "constraints": [
        r"\bconstraint\b", r"\bnot null\b", r"\bcheck\b", r"\bunique\b", r"\bуникал\w*\b",
    ],
    "indexes": [
        r"\bindex\b", r"\bиндекс\w*\b", r"\bbtree\b", r"\bhash\b",
    ],
    "types": [
        r"\bint\b", r"\binteger\b", r"\bbigint\b", r"\bsmallint\b",
        r"\bdate\b", r"\btimestamp\b", r"\bdatetime\b", r"\bboolean\b",
        r"\btext\b", r"\bvarchar\b", r"\bchar\b", r"\bjsonb?\b",
        r"\bnumeric\b", r"\bdecimal\b", r"\bfloat\b", r"\bdouble\b",
    ],
    "normalization": [
        r"\bнормализац\w*\b", r"\b1нф\b", r"\b2нф\b", r"\b3нф\b",
        r"\bденормализац\w*\b",
    ],
}


# ---------------------------
# Утилиты
# ---------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def preprocess_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)  # пунктуация -> пробел
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_question_id(raw_key: Any) -> Optional[str]:
    if raw_key is None:
        return None
    s = str(raw_key).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # при normalize_embeddings=True это скалярное произведение
    return float(np.dot(a, b))


def cosine_to_score_linear(sim: float, lo: float = COSINE_LO, hi: float = COSINE_HI) -> float:
    """
    Фиксированная (стабильная) шкала:
    score0_10 = clip( (sim-lo)/(hi-lo) * 10, 0, 10 ), затем клип в 1..10.
    """
    if hi <= lo:
        lo, hi = min(lo, hi), max(lo, hi) + 1e-6
    x = (sim - lo) / (hi - lo)
    score_0_10 = 10.0 * float(np.clip(x, 0.0, 1.0))
    score_1_10 = float(np.clip(score_0_10, 1.0, 10.0))
    return round(score_1_10, 1)


def count_pattern_hits(text: str, patterns: List[str]) -> int:
    hits = 0
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            hits += 1
    return hits


def concreteness_coverage(text: str) -> int:
    """
    Считает, сколько разных категорий конкретики покрыто (0..len(categories)).
    """
    t = preprocess_text(text)
    covered = 0
    for _, pats in CONCRETENESS_CATEGORIES.items():
        if any(re.search(p, t, flags=re.IGNORECASE) for p in pats):
            covered += 1
    return covered


def apply_quality_adjustments(base_score: float, raw_text: str) -> float:
    """
    Рекомендация: корректировки — только штрафы (и/или мягкие), без “раздувания” балла.
    Здесь:
      - штраф за неуверенность до -2.0
      - штраф за отсутствие конкретики (если нет ни одной категории) до -1.0
    """
    t = preprocess_text(raw_text or "")

    unc = count_pattern_hits(t, UNCERTAINTY_PATTERNS)  # 0..n
    cov = concreteness_coverage(t)                     # 0..5

    score = float(base_score)

    # Неуверенность: до -2.0
    score -= 0.7 * min(unc, 3)

    # Нет конкретики вообще: -1.0 (иначе 0)
    if cov == 0 and t:
        score -= 1.0

    score = float(np.clip(score, 1.0, 10.0))
    return round(score, 1)


def generate_comment(score: float) -> str:
    if score >= 8.5:
        return "Ответ полностью соответствует эталону: точная аргументация, корректные термины, логичная структура."
    elif score >= 6.0:
        return "Ответ в целом соответствует эталону, но есть небольшие упущения или неточности в деталях."
    elif score >= 4.0:
        return "Частичное соответствие: основные идеи верны, но много неточностей или пропущены ключевые моменты."
    else:
        return "Низкое соответствие: ответ содержит существенные ошибки, пропуски или нелогичные рассуждения."


def ai_etalon_to_text(ai_obj: Any) -> str:
    if ai_obj is None:
        return ""
    if isinstance(ai_obj, str):
        return ai_obj

    lines: List[str] = []

    def emit(prefix: str, v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            if v.strip():
                lines.append(f"{prefix}{v.strip()}")
        elif isinstance(v, (int, float, bool)):
            lines.append(f"{prefix}{v}")
        elif isinstance(v, list):
            for item in v:
                emit(prefix + "- ", item)
        elif isinstance(v, dict):
            for k, vv in v.items():
                key = str(k)
                if isinstance(vv, (str, int, float, bool)) and str(vv).strip():
                    lines.append(f"{key}: {vv}")
                else:
                    lines.append(f"{key}:")
                    emit("  ", vv)
        else:
            lines.append(f"{prefix}{str(v)}")

    emit("", ai_obj)
    return "\n".join(lines).strip()


def get_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def pick_final_by_max_sim(hr_sim: Optional[float], ai_sim: Optional[float]) -> Tuple[Optional[float], Optional[str]]:
    if hr_sim is None and ai_sim is None:
        return None, None
    if hr_sim is None:
        return ai_sim, "ai"
    if ai_sim is None:
        return hr_sim, "hr"
    return (hr_sim, "hr") if hr_sim >= ai_sim else (ai_sim, "ai")


# ---------------------------
# Основная логика
# ---------------------------
def evaluate(candidate_path: str, hr_path: str, ai_path: str, output_path: str) -> None:
    cand = load_json(candidate_path)
    et_hr = load_json(hr_path)
    et_ai = load_json(ai_path)

    if not isinstance(cand, dict):
        raise ValueError("Файл кандидата должен быть JSON-объектом (dict) вида {'Вопрос 1': '...', ...}.")
    if not isinstance(et_hr, dict):
        raise ValueError("etalon_hr.json должен быть dict.")
    if not isinstance(et_ai, dict):
        raise ValueError("etalon_ai.json должен быть dict.")

    # Индексация по номеру вопроса
    cand_by_q: Dict[str, str] = {}
    for k, v in cand.items():
        qid = normalize_question_id(k)
        if qid:
            cand_by_q[qid] = "" if v is None else str(v)

    hr_by_q: Dict[str, str] = {}
    for k, v in et_hr.items():
        qid = normalize_question_id(k)
        if qid:
            hr_by_q[qid] = "" if v is None else str(v)

    ai_by_q: Dict[str, str] = {}
    for k, v in et_ai.items():
        qid = normalize_question_id(k)
        if qid:
            ai_by_q[qid] = ai_etalon_to_text(v)

    all_qids = sorted(
        set(cand_by_q.keys()) | set(hr_by_q.keys()) | set(ai_by_q.keys()),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Предварительно эмбеддинги эталонов
    hr_texts = {qid: preprocess_text(txt) for qid, txt in hr_by_q.items()}
    ai_texts = {qid: preprocess_text(txt) for qid, txt in ai_by_q.items()}

    hr_qids = [qid for qid in all_qids if qid in hr_texts and hr_texts[qid]]
    ai_qids = [qid for qid in all_qids if qid in ai_texts and ai_texts[qid]]

    hr_embs: Dict[str, np.ndarray] = {}
    if hr_qids:
        embs = get_embeddings([hr_texts[qid] for qid in hr_qids], model)
        hr_embs = {qid: embs[i] for i, qid in enumerate(hr_qids)}

    ai_embs: Dict[str, np.ndarray] = {}
    if ai_qids:
        embs = get_embeddings([ai_texts[qid] for qid in ai_qids], model)
        ai_embs = {qid: embs[i] for i, qid in enumerate(ai_qids)}

    results: List[Dict[str, Any]] = []
    meta_warnings: List[str] = []

    for qid in all_qids:
        cand_raw = cand_by_q.get(qid, "")
        if not cand_raw:
            meta_warnings.append(f"Вопрос {qid}: у кандидата нет ответа или пусто.")

        cand_txt = preprocess_text(cand_raw)
        cand_emb = get_embeddings([cand_txt], model)[0] if cand_txt else None

        hr_sim: Optional[float] = None
        if cand_emb is not None and qid in hr_embs:
            hr_sim = round(cosine_sim(hr_embs[qid], cand_emb), 4)

        ai_sim: Optional[float] = None
        if cand_emb is not None and qid in ai_embs:
            ai_sim = round(cosine_sim(ai_embs[qid], cand_emb), 4)

        chosen_sim, chosen = pick_final_by_max_sim(hr_sim, ai_sim)

        if chosen_sim is None:
            final_base = None
            final_score = None
            comment = "Нет данных для оценки (нет эталона и/или ответа кандидата)."
        else:
            final_base = cosine_to_score_linear(chosen_sim, lo=COSINE_LO, hi=COSINE_HI)
            final_score = apply_quality_adjustments(final_base, cand_raw)
            comment = generate_comment(final_score)

        results.append(
            {
                "Номер вопроса": qid,
                "Ответ кандидата": cand_raw,
                "Cosine HR": hr_sim,
                "Cosine AI": ai_sim,
                "Эталон выбран": chosen,
                "Оценка (base)": final_base,
                "Оценка (final)": final_score,
                "Комментарий": comment,
            }
        )

    out = {
        "input": {
            "candidate_file": candidate_path,
            "etalon_hr_file": hr_path,
            "etalon_ai_file": ai_path,
        },
        "calibration": {
            "auto_calibrate": AUTO_CALIBRATE,
            "cosine_lo": COSINE_LO,
            "cosine_hi": COSINE_HI,
            "base_mapping": "linear_clip_cosine_to_1_10",
            "final_policy": "max_cosine_to_etalon_then_penalties_only",
        },
        "warnings": meta_warnings,
        "results": results,
    }
    save_json(output_path, out)
    print(f"OK: сохранено в {output_path}")


if __name__ == "__main__":
    evaluate(CANDIDATE_FILE, ETALON_HR_FILE, ETALON_AI_FILE, OUTPUT_FILE)