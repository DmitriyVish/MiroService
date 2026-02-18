import sys
import os
import logging
import json
import pandas as pd
from datetime import datetime
from augment_service.other.augment_module import TaskGenerator



class QuestionEvaluator:
    """
    Класс для генерации ответов от разных "агентов" на вопросы из JSON,  сохранения в Excel (по вопросу)
    и в единый JSON‑файл со всеми результатами.
    """

    ANSWER_PROMPT_TEMPLATE = """
    Вопрос: "{QUESTION_TEXT}"

    Представь, что ты {AGENT_ROLE_DESCRIPTION}.
    Проходишь тестирование на должность DATA ENGINEER в крупную IT-компанию. Ответь на вопрос в соответствующей твоей роли форме.
    Ответ:
    """

    AGENT_ROLES = {
        "well_prepared_candidate": {
            "description": "хорошо подготовленный кандидат, обладающий отличными аналитическими способностями, глубоко анализирует вопрос и дает точный, логичный и обоснованный ответ"
        },
        "moderately_prepared_candidate": {
            "description": "средне подготовленный кандидат, понимает основы, но может упустить некоторые тонкие детали или дать частично правильный, частично неуверенный ответ"
        },
        "poorly_prepared_candidate": {
            "description": "плохо подготовленный кандидат, слабо понимает суть вопроса, может дать поверхностный, запутанный или неправильный ответ"
        },
        "generative_ai_qwen": {
            "description": "генеративная нейронная сеть Qwen, обученная на большом объёме текстов, дающая логичный, структурированный и вежливый ответ, основанный на вероятностях"
        }
    }

    def __init__(
        self,
        model_name: str = "qwen3-4b-instruct-2507",
        log_level: int = logging.INFO,
        log_file: str | None = None
    ):
        self.generator = TaskGenerator(model_name)
        self.setup_sys_path()
        self.setup_logging(log_level, log_file)
        self.all_results = []  # Для сбора всех результатов (для итогового JSON)

    def setup_sys_path(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)

    def setup_logging(self, log_level: int, log_file: str | None):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8") if log_file
                else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_prompt(self, question_text: str, agent_type: str) -> str:
        role_desc = self.AGENT_ROLES.get(agent_type, {}).get("description", "агента")
        prompt = self.ANSWER_PROMPT_TEMPLATE.replace("{QUESTION_TEXT}", question_text)
        prompt = prompt.replace("{AGENT_ROLE_DESCRIPTION}", role_desc)
        return prompt

    def load_questions_from_json(self, json_path: str) -> dict[str, str]:
        """Читает вопросы из JSON‑файла. Ожидаемый формат: {"Вопрос 1": "текст", "Вопрос 2": "текст", ...}"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON должен содержать объект (словарь) с вопросами.")
            self.logger.info(f"Загружено {len(data)} вопросов из {json_path}")
            return data
        except Exception as e:
            self.logger.error(f"Ошибка при чтении JSON: {e}")
            return {}

    def evaluate_question(
        self,
        question_num: str,
        question_text: str,
        output_excel: str,
        temperature: float = 0.7,
        max_length: int = 1000,
        num_return_sequences: int = 1,
        **kwargs # Для возможных дополнительных параметров generate_tasks
    ):
        """Генерирует ответы для одного вопроса и сохраняет в Excel + добавляет в общий набор результатов.
        
        Args:
            question: текст вопроса.
            output_filename: имя выходного Excel‑файла.
            temperature: температура генерации (от 0.0 до 1.0+).
            max_length: максимальная длина ответа в токенах.
            num_return_sequences: количество генерируемых ответов на промпт (обычно 1).
            **kwargs: дополнительные аргументы для generator.generate_tasks().
        """
        excel_data = []
        generation_time = datetime.now().isoformat()

         # Проверка типа
        if not isinstance(question_text, str):
            self.logger.error(f"question_text не является строкой для вопроса {question_num}: {type(question_text)}")
            question_text = str(question_text)  # Приводим к строке

        preview = question_text[:100]
        self.logger.info(f'Обработка вопроса {question_num}: {preview}...')
        
        # self.logger.info(f'!Обработка вопроса {question_num}: {question_text[:100]}...')
        self.logger.info(f'!Выходной Excel: {output_excel}')

        for agent in self.AGENT_ROLES:
            agent_prompt = self.generate_prompt(question_text, agent)
            self.logger.debug(f"\n--- ПРОМПТ ДЛЯ {agent} ---\n{agent_prompt}")


            try:
                # Генерация ответа с переданными параметрами
                response = self.generator.generate_tasks(
                    agent_prompt,
                    num_return_sequences=num_return_sequences,
                    max_length=max_length,
                    temperature=temperature,
                    **kwargs # Передаём дополнительные параметры
                )[0]
            except Exception as e:
                self.logger.error(f'!Ошибка при генерации ответа для {agent}: {e}')
                response = "[Ошибка генерации]"

            item = {
                "Дата и время генерации": generation_time,
                "Вопрос номер": question_num,
                "Промпт": agent_prompt,
                "ответ": response.strip(),
                "роль_агента": agent
            }
            excel_data.append(item)
            self.all_results.append(item)  # Добавляем в общий список для JSON

        # Сохранение в Excel (один файл на вопрос)
        try:
            df = pd.DataFrame(excel_data)
            df.to_excel(output_excel, index=False, engine='openpyxl')
            self.logger.info(f'!Результаты сохранены в Excel: {output_excel}')
        except Exception as e:
            self.logger.error(f'!Ошибка при сохранении в Excel: {e}')

        # Сводка в лог
        self.logger.info("--- Сводка ответов ---")
        for item in excel_data:
            self.logger.info(f"[{item['роль_агента']}]: {item['ответ']}")
        self.logger.info('-!' * 40)

    def save_all_results_to_json(self, json_output: str):
        """Сохраняет все собранные результаты в единый JSON‑файл."""
        try:
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Все результаты сохранены в JSON: {json_output}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении в JSON: {e}")

    def run_evaluation(
        self,
        questions_json: str,
        excel_prefix: str = "results_question_",
        json_output: str = "all_results.json",
        **generation_params
    ):
        """
        Основной метод: читает вопросы из JSON, обрабатывает каждый, сохраняет Excel и итоговый JSON.
        Args:
            questions_json: путь к JSON с вопросами.
            excel_prefix: префикс для Excel‑файлов (будет: excel_prefix + "1.xlsx", ...).
            json_output: имя итогового JSON‑файла.
            **generation_params: параметры для evaluate_question (temperature, max_length и т. д.).
        """
        questions = self.load_questions_from_json(questions_json)
        if not questions:
            self.logger.error("Нет вопросов для обработки.")
            return

        for q_num, q_text in questions.items():
            # Формируем имя Excel‑файла: заменяем пробелы на _ в номере вопроса
            excel_file = f"{excel_prefix}{q_num.replace(' ', '_')}.xlsx"
            self.evaluate_question(
                question_num=q_num,
                question_text=q_text,
                output_excel=excel_file,
                **generation_params
            )

        # Сохраняем все результаты в один JSON
        self.save_all_results_to_json(json_output)


# --- Примеры использования ---
if __name__ == "__main__":
    # # # Пример 1: вывод в консоль (уровень INFO), без файла логов
    # evaluator = QuestionEvaluator(
    #     model_name="qwen3-4b-instruct-2507",
    #     log_level=logging.INFO,
    #     log_file="logs/evaluation.log"
    # )

    # # Запускаем обработку
    # evaluator.run_evaluation(
    #     questions_json="questions.json",  # Путь к файлу с вопросами
    #     excel_prefix="results_",        # Префикс для Excel: "results_Вопрос_1.xlsx" и т. д.
    #     json_output="all_results.json",   # Итоговый JSON со всеми ответами
    #     temperature=0.7,
    #     max_length=1000,
    #     num_return_sequences=1
    # )

    # Пример 2: запись логов в файл (уровень DEBUG)
    evaluator_with_log = QuestionEvaluator(
        model_name="qwen3-4b-instruct-2507",
        log_level=logging.DEBUG,
        log_file="logs/evaluation.log"
    )

    evaluator_with_log.run_evaluation(
        questions_json="prompt_templates/questions_with_params.json",
        excel_prefix="prompt_templates/debug_results_",
        json_output="prompt_templates/debug_all_results.json",
        temperature=0.7,
        max_length=1000,
        num_return_sequences=1
        # do_sample=True,
        # top_p=0.9
    )