# evaluate_v4.py
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from eval_criterias import (
    evaluate_criteria,
    get_criteria_for_question,
    criteria_score_1_10,
)

# ---------------------------
# Настройки
# ---------------------------
CANDIDATE_FILE = "prompt_templates/parser_responses/candidat_1.json"
ETALON_HR_FILE = "prompt_templates/interview_questions/etalon_hr.json"
ETALON_AI_FILE = "prompt_templates/interview_questions/etalon_ai.json"
OUTPUT_FILE = "prompt_templates/eval_result_parser/evaluation_results_v5.json"

AUTO_CALIBRATE = False

COSINE_LO = 0.45
COSINE_HI = 0.90

# Весовая формула объединения
W_COSINE = 0.7
W_CRITERIA = 0.3

# Длина ответа: сравнение с AI-эталоном (по кол-ву символов в "raw" тексте).
# Если ответ короче эталона — штраф до MAX_LEN_PENALTY.
# Если ответ длиннее эталона — бонус до MAX_LEN_BONUS.
LEN_TARGET_MIN_RATIO = 1.0   # хотим "не менее чем в эталоне"
MAX_LEN_PENALTY = 1.5        # максимум штрафа в баллах
MAX_LEN_BONUS = 0.5          # небольшой бонус
LEN_SMOOTHING = 30           # чтобы короткие эталоны не ломали ratio

UNCERTAINTY_PATTERNS = [
    r"\bне уверен\b", r"\bне уверена\b", r"\bне уверен(а)?\b",
    r"\bне знаю\b", r"\bне помню\b", r"\bзатрудняюсь\b",
    r"\bне эксперт\b", r"\bне специалист\b", r"\bмогу ошибаться\b",
    r"\bвозможно\b", r"\bскорее всего\b", r"\bпредположим\b",
]

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
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_question_id(raw_key: Any) -> Optional[str]:
    if raw_key is None:
        return None
    s = str(raw_key).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def cosine_to_score_linear(sim: float, lo: float = COSINE_LO, hi: float = COSINE_HI) -> float:
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
    t = preprocess_text(text)
    covered = 0
    for _, pats in CONCRETENESS_CATEGORIES.items():
        if any(re.search(p, t, flags=re.IGNORECASE) for p in pats):
            covered += 1
    return covered


def apply_quality_adjustments_penalties_only(base_score: float, raw_text: str) -> float:
    t = preprocess_text(raw_text or "")

    unc = count_pattern_hits(t, UNCERTAINTY_PATTERNS)
    cov = concreteness_coverage(t)

    score = float(base_score)

    score -= 0.7 * min(unc, 3)

    if cov == 0 and t:
        score -= 1.0

    score = float(np.clip(score, 1.0, 10.0))
    return round(score, 1)


def length_adjustment_by_ai_etalon(cand_raw: str, ai_etalon_raw: str) -> Dict[str, Any]:
    """
    Штраф/бонус по длине (кол-во символов).
    - Если кандидат короче AI эталона: штраф до MAX_LEN_PENALTY.
    - Если кандидат длиннее: бонус до MAX_LEN_BONUS.
    """
    cand_len = len((cand_raw or "").strip())
    et_len = len((ai_etalon_raw or "").strip())

    # если эталона нет — не трогаем
    if et_len <= 0 or cand_len <= 0:
        return {
            "cand_len": cand_len,
            "ai_etalon_len": et_len,
            "ratio": None,
            "delta_score": 0.0,
        }

    denom = float(et_len + LEN_SMOOTHING)
    ratio = float((cand_len + LEN_SMOOTHING) / denom)

    delta = 0.0
    if ratio < LEN_TARGET_MIN_RATIO:
        # линейный штраф: при ratio=0 -> MAX_LEN_PENALTY, при ratio=1 -> 0
        delta = -MAX_LEN_PENALTY * (LEN_TARGET_MIN_RATIO - ratio) / LEN_TARGET_MIN_RATIO
    else:
        # небольшой бонус, если существенно длиннее (насыщаемся)
        # при ratio=1 -> 0, при ratio>=1.8 -> MAX_LEN_BONUS
        bonus_ratio = min(max((ratio - 1.0) / 0.8, 0.0), 1.0)
        delta = MAX_LEN_BONUS * bonus_ratio

    return {
        "cand_len": cand_len,
        "ai_etalon_len": et_len,
        "ratio": round(ratio, 3),
        "delta_score": round(float(delta), 2),
    }


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


def criteria_key_from_qid(qid: str) -> str:
    return f"Вопрос {qid}"


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

    ai_by_q_raw: Dict[str, str] = {}
    for k, v in et_ai.items():
        qid = normalize_question_id(k)
        if qid:
            ai_by_q_raw[qid] = ai_etalon_to_text(v)

    all_qids = sorted(
        set(cand_by_q.keys()) | set(hr_by_q.keys()) | set(ai_by_q_raw.keys()),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    model = SentenceTransformer("all-MiniLM-L6-v2")

    hr_texts = {qid: preprocess_text(txt) for qid, txt in hr_by_q.items()}
    ai_texts = {qid: preprocess_text(txt) for qid, txt in ai_by_q_raw.items()}

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

        # --- criteria pack ---
        qkey = criteria_key_from_qid(qid)
        criteria = get_criteria_for_question(qkey)
        criteria_pack = evaluate_criteria(cand_raw, criteria) if criteria else None
        criteria_1_10 = round(criteria_score_1_10(criteria_pack), 2) if criteria_pack else None

        # --- scoring ---
        if chosen_sim is None:
            final_score = None
            comment = "Нет данных для оценки (нет эталона и/или ответа кандидата)."
            len_pack = {"cand_len": len((cand_raw or "").strip()), "ai_etalon_len": 0, "ratio": None, "delta_score": 0.0}
            combined_before_penalties = None
        else:
            cosine_1_10 = cosine_to_score_linear(chosen_sim, lo=COSINE_LO, hi=COSINE_HI)

            # Если критериев нет для вопроса — используем только cosine
            if criteria_1_10 is None:
                combined = float(cosine_1_10)
            else:
                combined = float(W_COSINE * cosine_1_10 + W_CRITERIA * criteria_1_10)

            combined = float(np.clip(combined, 1.0, 10.0))
            combined_before_penalties = round(combined, 2)

            # длина vs AI эталон
            ai_etalon_raw = ai_by_q_raw.get(qid, "")
            len_pack = length_adjustment_by_ai_etalon(cand_raw, ai_etalon_raw)
            combined += float(len_pack["delta_score"])

            # ваши старые эвристики (penalties-only)
            final_score = apply_quality_adjustments_penalties_only(combined, cand_raw)

            comment = generate_comment(final_score)

        results.append(
            {
                "Номер вопроса": qid,
                "Ответ кандидата": cand_raw,
                "Cosine HR": hr_sim,
                "Cosine AI": ai_sim,
                "Эталон выбран": chosen,

                "Criteria used": bool(criteria_pack is not None),
                "Criteria pack": criteria_pack,
                "Criteria (1..10)": criteria_1_10,

                "Cosine (1..10 base)": None if chosen_sim is None else cosine_to_score_linear(chosen_sim),
                "Combined (0.7*cosine + 0.3*criteria) before penalties": combined_before_penalties,

                "Length vs AI etalon": len_pack,

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
        "scoring": {
            "cosine_mapping": {"type": "linear_clip_cosine_to_1_10", "cosine_lo": COSINE_LO, "cosine_hi": COSINE_HI},
            "blend": {"formula": "final_blend = 0.7*cosine_1_10 + 0.3*criteria_1_10 (if criteria exists else cosine_1_10)", "w_cosine": W_COSINE, "w_criteria": W_CRITERIA},
            "length_adjustment": {"target_min_ratio": LEN_TARGET_MIN_RATIO, "max_penalty": MAX_LEN_PENALTY, "max_bonus": MAX_LEN_BONUS, "smoothing": LEN_SMOOTHING},
            "final_policy": "blend -> length_adjust -> penalties_only(uncertainty, no_concreteness)",
        },
        "warnings": meta_warnings,
        "results": results,
    }
    save_json(output_path, out)
    print(f"OK: сохранено в {output_path}")


if __name__ == "__main__":
    evaluate(CANDIDATE_FILE, ETALON_HR_FILE, ETALON_AI_FILE, OUTPUT_FILE)