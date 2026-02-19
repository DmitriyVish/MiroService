import re
import json 
import logging 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(console_handler)
    
class ParsingValidator:
    """
    Класс для валидации качества парсинга и оценки извлеченных ответов.
    """
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Инициализирует валидатор.
        Args:
            embedding_model_name (str): Имя модели для вычисления эмбеддингов (для семантической оценки).
                                        sentence-transformers/all-MiniLM-L6-v2 
        """
        logger.info(f"Загрузка модели эмбеддингов: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Модель эмбеддингов загружена")
        
    def calculate_semantic_similarity(self, extracted_answer: str, reference_answer: str) -> float:
        """
        Вычисляет косинусное сходство между эмбеддингами извлеченного и эталонного ответа.
        Args:
            extracted_answer (str): Ответ, извлеченный из документа.
            reference_answer (str): Эталонный (правильный) ответ.
        Returns:
            float: Косинусное сходство (от 0 до 1). Диапазон [-1, 0) обрезается до 0.
        """
        
        if not extracted_answer or not reference_answer:
            logger.debug("Один из ответов пустой")
            return 0.0
        
        embeddings = self.embedding_model.encode([extracted_answer, reference_answer])
        cos_sim = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
        return max(0.0, min(1.0, float(cos_sim)))
    
    def calculate_extract_match(self, extracted_answer: str, reference_answer: str) -> int: 
        """
        Проверяет, совпадает ли извлеченный ответ с эталонным (без учета регистра и пробелов).
        если тексты идентичны после нормализации, иначе 0.        
        Используется для оценки точности воспроизведения конкретных фраз или ответов.

        Args:
            extracted_answer (str): Ответ, извлеченный из документа.
            reference_answer (str): Эталонный (правильный) ответ.
        Returns:
            int: 1, если совпадает, иначе 0.
        """
        if not extracted_answer and not reference_answer:
            logger.debug("Оба ответа пустые")
            return 0
        if not extracted_answer or not reference_answer:
            logger.debug("Один из ответов пустой")
            return 0
        # re.sub(r'\s+', ' ', ...) заменяет любую последовательность пробельных символов на один пробел
        norm_extracted = re.sub(r'\s+', ' ', extracted_answer.lower()).strip()
        norm_reference = re.sub(r'\s+', ' ', reference_answer.lower()).strip()
        # Сравниваем строки
        match = int(norm_extracted == norm_reference)
        logger.debug(f"Точное совпадение: {match} для '{norm_extracted}' и '{norm_reference}'")
        return match
    
    def calculate_f1_score_tokens(self, extracted_answer: str, reference_answer: str) -> float:
        """
        Вычисляет F1-меру для наборов токенов между извлеченным и эталонным ответом.       
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        где TP (True Positives) - токены, правильно извлеченные моделью (есть и в extracted, и в reference),
             FP (False Positives) - токены, извлеченные моделью, но отсутствующие в эталоне,
             FN (False Negatives) - токены, присутствующие в эталоне, но не извлеченные моделью.
        Используется для оценки пересечения слов между двумя ответами.

        Args:
            extracted_answer (str): Ответ, извлеченный из документа.
            reference_answer (str): Эталонный (правильный) ответ.
        Returns:
            float: F1-мера (от 0 до 1).
        """
        # Обработка случая, когда оба ответа пустые
        if not extracted_answer and not reference_answer:
            logger.debug("Оба ответа пустые")
            return 0.0
        # Обработка случая, когда один из ответов пустой
        if not extracted_answer or not reference_answer:
            logger.debug("Один из ответов пустой")
            return 0.
        # re.findall(r'\b\w+\b', ...) находит все последовательности "словных" символов (буквы, цифры, подчеркивание), ограниченные границами слов (\b)
        tokens_extracted = set(re.findall(r'\b\w+\b', extracted_answer.lower()))
        tokens_reference = set(re.findall(r'\b\w+\b', reference_answer.lower()))
        
        # Пересечение токенов (TP)
        intersection = tokens_extracted.intersection(tokens_reference)
        if not intersection:
            logger.debug("Нет пересечения токенов")
            return 0.0

        # Precision и Recall
        precision = len(intersection) / len(tokens_extracted) if tokens_extracted else 0
        recall = len(intersection) / len(tokens_reference) if tokens_reference else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        logger.debug(f"F1 (по токенам): {round(f1, 3)} для '{tokens_extracted}' и '{tokens_reference}'")
        return f1
    
    def evaluate_single_answer(self, extracted_answer: str, reference_answer: str, threshold_semantic: float = 0.7) -> dict[str, float]:
        """
        Оценивает один извлеченный ответ по отношению к эталонному.
        Args:
            extracted_answer (str): Ответ, извлеченный из документа.
            reference_answer (str): Эталонный (правильный) ответ.
            threshold_semantic (float): Порог для семантического сходства (для бинарной оценки).
        Returns:
            dict: Словарь с вычисленными метриками.
        """
        semantic_sim = self.calculate_semantic_similarity(extracted_answer, reference_answer)
        extract_match = self.calculate_extract_match(extracted_answer, reference_answer)
        f1 = self.calculate_f1_score_tokens(extracted_answer, reference_answer)
        # Бинарное совпадение по семантике
        semantic_match = 1 if semantic_sim >= threshold_semantic else 0
        return {
            "exact_match": extract_match, 
            "f1_score": f1, 
            "semantic_similarity": semantic_sim, 
            "semantic_match": semantic_match 
        }
        
    def validate_parsing_quality_with_reference(self, parsed_results: list[dict], reference_results: list[dict]) -> dict[str, float]:
        """
        Валидирует качество парсинга по сравнению с эталонными результатами.        
        Среднее значение вычисляется по всем парам (извлеченный ответ, эталонный ответ).

        Args:
            parsed_results (list[dict]): Список результатов парсинга 
            reference_results (list[dict]): Список эталонных результатов 
        Returns:
            dict[str, float]: Словарь с усредненными метриками.
        """
        # Словарь для хранения всех значений метрик для усреднения
        all_metrics = {
            "exact_match": [],
            "f1_score": [],
            "semantic_similarity": [],
            "semantic_match": []
            }
        
        # Проходим по парам результатов
        for i, (parsed_res, ref_res) in enumerate(zip(parsed_results, reference_results)):
            logger.debug(f"Оценка результата {i+1}")
            for q_num in range(1, 5):
                q_key = f"вопрос {q_num}"
                extracted_ans = parsed_res.get(q_key, "")
                reference_ans = ref_res.get(q_key, "")
                
                # Вычисляем метрики для этой пары ответов
                eval_res = self.evaluate_single_answer(extracted_ans, reference_ans)
                for metric, value in eval_res.items():
                    all_metrics[metric].append(value)
                logger.debug(
                            f"""{q_key}: EM={eval_res['exact_match']:.3f},
                            F1={eval_res['f1_score']:.3f},
                            SemSim={eval_res['semantic_similarity']:.3f},
                            SemMatch={eval_res['semantic_match']}"""
                            ) 
        # Среднее значение по всем метрикам   
        average_metrics = {}
        for metric, values in all_metrics.items():
            if values:
                average_metrics[metric] = np.mean(values) 
            else:
                average_metrics[metric] = 0.0
            logger.info(f"Средняя {metric}: {average_metrics[metric]:.3f}")
        return average_metrics
    
    def validate_parsing_quality_without_reference(self, parsed_results: list[dict]) -> dict[str, float]:
        """
        Оценивает качество парсинга без эталона (например, наличие всех 4 ответов, их длина).
        Эти метрики не оценивают смысл ответа, а показывают полноту заполнения и характеристики данных.

        Args:
            parsed_results (list[dict]): Список результатов парсинга 
        Returns:
            dict[str, float]: Словарь с метриками качества парсинга
        """ 
        # Общее количество полей для заполнения
        total_questions = len(parsed_results) * 4
        filled_questions = 0 # Счетчик заполненных полей
        total_length = 0 # Общая длина всех ответов
        lengths_per_question = []
        
        # Проходим по всем результатам
        for res in parsed_results:            
            for q_num in range(1, 5):
                q_key = f"вопрос {q_num}"               
                ans = res.get(q_key, "").strip()
                # Если ответ не пустой, увеличиваем счетчик заполненных
                if ans:
                    filled_questions += 1
                    # Добавляем длину ответа к общей длине
                    total_length += len(ans)
                    # Сохраняем длину для вычисления стандартного отклонения
                    lengths_per_question.append(len(ans))
                else:
                    # Если ответ пустой, длина = 0. 
                    lengths_per_question.append(0)   
         
        # completeness_ratio: Доля заполненных полей от общего числа (Filled Questions) / (Total Questions)        
        # Показывает, насколько полно модель заполнила ожидаемые поля.
        completeness_ratio = filled_questions / total_questions if total_questions > 0 else 0.0

        # average_answer_length: Средняя длина ответа (по всем вопросам, включая пустые). (Total Length of All Answers) / (Total Questions)
        # Показывает типичный размер сгенерированного ответа. 
        average_length = total_length / total_questions if total_questions > 0 else 0.0

        # std_dev_answer_length: Стандартное отклонение длины ответа (по всем вопросам, включая пустые).
        # Standard Deviation = sqrt(sum((length_i - mean)^2) / N )      
        # Показывает, насколько длина ответов варьируется. Высокое std_dev может указывать на непоследовательность генерации.       
        length_std_dev = np.std(lengths_per_question, ddof=0) if lengths_per_question else 0.0

        metrics = {
            "completeness_ratio": completeness_ratio, 
            "average_answer_length": average_length,  
            "std_dev_answer_length": length_std_dev   
        }

        logger.info(f"Метрики парсинга (без эталона): {metrics}")
        return metrics
    
if __name__ == "__main__":
    validator = ParsingValidator()

    # Пример результатов парсинга
    parsed_results_example = [
        {
            "вопрос 1": "Да, возраст лучше заменить на дату рождения.",
            "вопрос 2": "Создать отдельную таблицу для отслеживания изменений цен.",
            "вопрос 3": "SELECT ... FROM ... WHERE ...",
            "вопрос 4": "Потому что он упал у стены."
        },
        {
            "вопрос 1": "Нет, возраст - плохая практика.",
            "вопрос 2": "", # Пустой ответ
            "вопрос 3": "SELECT d.department_id, d.department_name, SUM(e.salary) FROM employees e JOIN departments d ON e.department_id = d.department_id WHERE e.manager_id IS NULL GROUP BY d.department_id HAVING SUM(e.salary) > 100000;",
            "вопрос 4": "Карандаш лежит у плинтуса."
        }
    ]

    # Пример эталонных результатов (для демонстрации validate_parsing_quality_with_reference)
    reference_results_example = [
        {
            "вопрос 1": "Считаю, что поле 'age' не корректно. Лучше хранить 'birth_date'.",
            "вопрос 2": "Для историчности цен нужно создать таблицу 'item_price_history'.",
            "вопрос 3": "SELECT D.DEPARTMENT_ID, D.DEPARTMENT_NAME, SUM(E.SALARY) AS TOTAL_SALARY FROM EMPLOYEES E JOIN DEPARTMENTS D ON E.DEPARTMENT_ID = D.DEPARTMENT_ID JOIN LOCATIONS L ON D.LOCATION_ID = L.LOCATION_ID WHERE E.MANAGER_ID IS NULL AND L.CITY = 'Seoul' GROUP BY D.DEPARTMENT_ID, D.DEPARTMENT_NAME HAVING SUM(E.SALARY) > 100000;",
            "вопрос 4": "Карандаш находится в углу комнаты."
        },
        {
            "вопрос 1": "Age number не подходит, используйте date birth.",
            "вопрос 2": "Создайте таблицу item_prices_history.",
            "вопрос 3": "SELECT d.department_id, d.department_name, SUM(e.salary) AS total_salary FROM employees e JOIN departments d ON e.department_id = d.department_id WHERE e.manager_id IS NULL AND d.location_id IN ( SELECT location_id FROM locations WHERE city = 'SEOUL' ) GROUP BY d.department_id, d.department_name HAVING SUM(e.salary) > 100000;",
            "вопрос 4": "Карандаш лежит у стены."
        }
    ]

    print("\n--- Валидация с эталоном ---")
    metrics_with_ref = validator.validate_parsing_quality_with_reference(parsed_results_example, reference_results_example)
    print("Средние метрики (с эталоном):")
    for k, v in metrics_with_ref.items():
        print(f"  {k}: {v:.3f}")

    print("\n--- Валидация без эталона ---")
    metrics_without_ref = validator.validate_parsing_quality_without_reference(parsed_results_example)
    print("Метрики парсинга (без эталона):")
    for k, v in metrics_without_ref.items():
        print(f"  {k}: {v:.3f}")