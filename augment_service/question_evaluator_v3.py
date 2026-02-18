import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging
from augment_module_v2 import TaskGenerator



class QuestionEvaluator:
    """
    Класс для генерации ответов от разных "агентов" на вопросы из JSON, с возможностью
    добавления дополнительных инструкций к отдельным вопросам и полным логированием.
    """

    ANSWER_PROMPT_TEMPLATE = """
    Вопрос: "{QUESTION_TEXT}"

    {ADDITIONAL_INSTRUCTIONS}

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

    def __init__(self, model_name: str = "qwen3-4b-instruct-2507", log_file: str = "logs/evaluator.log"):
        self.all_results = []

        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Удаляем существующие обработчики, чтобы не дублировать вывод
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Форматировщик
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Handler для файла
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Handler для консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Добавляем обработчики
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.setup_sys_path()
        self.generator = TaskGenerator(model_name)

    def setup_sys_path(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        self.logger.info(f"Добавлен в sys.path: {parent_dir}")

    def generate_prompt(
        self,
        question_text: str,
        agent_type: str,
        additional_instructions: str = ""
    ) -> str:
        role_desc = self.AGENT_ROLES.get(agent_type, {}).get("description", "агента")
        prompt = self.ANSWER_PROMPT_TEMPLATE.replace("{QUESTION_TEXT}", question_text)
        prompt = prompt.replace("{AGENT_ROLE_DESCRIPTION}", role_desc)
        prompt = prompt.replace("{ADDITIONAL_INSTRUCTIONS}", additional_instructions.strip())
        return prompt

    def load_questions_from_json(self, json_path: str) -> dict[str, str]:
        self.logger.info(f"Чтение вопросов из файла: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSON должен содержать объект (словарь) с вопросами.")
            self.logger.info(f"Загружено {len(data)} вопросов.")
            return data
        except FileNotFoundError:
            self.logger.error(f"Файл не найден: {json_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка парсинга JSON: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка при чтении файла: {e}")
            return {}

    def evaluate_question(
        self,
        question_num: str,
        question_text: str,
        output_excel: str,
        additional_instructions_map: dict = None,
        temperature: float = 0.7,
        max_length: int = 1000,
        num_return_sequences: int = 1,
        **kwargs
    ):
        self.logger.info(f'!Начало обработки вопроса: {question_num}')
        excel_data = []
        generation_time = datetime.now().isoformat()

        if not isinstance(question_text, str):
            question_text = str(question_text)
            self.logger.warning(f'!Текст вопроса {question_num} приведён к строке.')

        # Получаем дополнительные инструкции
        additional_instructions = ""
        if additional_instructions_map and question_num in additional_instructions_map:
            additional_instructions = additional_instructions_map[question_num]
            self.logger.debug(f"Для вопроса {question_num} добавлены инструкции: {additional_instructions}")

        for agent in self.AGENT_ROLES:
            self.logger.debug(f'!Генерация ответа для агента: {agent}')
            agent_prompt = self.generate_prompt(question_text, agent, additional_instructions)

            try:
                response = self.generator.generate_tasks(
                    agent_prompt,
                    num_return_sequences=num_return_sequences,
                    max_length=max_length,
                    temperature=temperature,
                    **kwargs
                )[0]
                self.logger.debug(f'!Ответ от {agent} получен.')
            except Exception as e:
                self.logger.error(f'!Ошибка при генерации ответа для агента {agent}: {e}')
                response = "Ошибка генерации ответа."

            item = {
                "Дата и время генерации": generation_time,
                "Вопрос номер": question_num,
                "Промпт": agent_prompt,
                "ответ": response.strip(),
                "роль_агента": agent
            }
            excel_data.append(item)
            self.all_results.append(item)

        # Сохранение в Excel
        try:
            df = pd.DataFrame(excel_data)
            df.to_excel(output_excel, index=False, engine='openpyxl')
            self.logger.info(f'!Результаты для вопроса {question_num} сохранены в {output_excel}')
        except Exception as e:
            self.logger.error(f'!Ошибка при сохранении в Excel ({output_excel}): {e}')

    def save_all_results_to_json(self, json_output: str):
        self.logger.info(f'!Сохранение всех результатов в JSON: {json_output}')
        try:
            with open(json_output, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Все результаты успешно сохранены в {json_output}")
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении JSON ({json_output}): {e}")


    def run_evaluation(
        self,
        questions_json: str,
        excel_prefix: str = "results_question_",
        json_output: str = "all_results.json",
        additional_instructions: dict = None,
        **generation_params
    ):
        """
        Основной метод: читает вопросы из JSON, обрабатывает каждый, сохраняет Excel и итоговый JSON.
        Логирует все этапы выполнения.

        Args:
            questions_json (str): путь к JSON с вопросами.
            excel_prefix (str): префикс для Excel‑файлов (будет: excel_prefix + "Вопрос_1.xlsx" и т. д.).
            json_output (str): имя итогового JSON‑файла.
            additional_instructions (dict): карта {номер_вопроса: текст_инструкции} для дополнения промпта.
            **generation_params: параметры для evaluate_question (temperature, max_length и др.).
        """
        self.logger.info("Запуск оценки вопросов...")
        self.logger.debug(f"Параметры генерации: {generation_params}")
        self.logger.debug(f"Дополнительные инструкции: {additional_instructions}")

        # Загрузка вопросов
        questions = self.load_questions_from_json(questions_json)
        if not questions:
            self.logger.error("Не удалось загрузить вопросы. Проверка прервана.")
            return

        self.logger.info(f"Начата обработка {len(questions)} вопросов.")

        # Обработка каждого вопроса
        for q_num, q_text in questions.items():
            self.logger.info(f'!Обработка вопроса: {q_num}')

            # Формируем безопасное имя файла (заменяем пробелы и спецсимволы)
            safe_q_num = q_num.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
            excel_file = f"{excel_prefix}{safe_q_num}.xlsx"

            try:
                self.evaluate_question(
                    question_num=q_num,
                    question_text=q_text,
                    output_excel=excel_file,
                    additional_instructions_map=additional_instructions,
                    **generation_params
                )
            except Exception as e:
                self.logger.error(f"Критическая ошибка при обработке вопроса {q_num}: {e}")
                # Продолжаем с следующим вопросом

        # Сохранение общего результата
        try:
            self.save_all_results_to_json(json_output)
        except Exception as e:
            self.logger.error(f"Не удалось сохранить итоговый JSON: {e}")

        self.logger.info("Оценка завершена.")


evaluator = QuestionEvaluator()



# С инструкцией только для вопроса 4
additional_instructions = {
    "Вопрос 4": "Внимание! Вопрос, который ты сейчас видишь, - это нестандартная логическая задачка, призванная проверить твоё креативное мышление, способность анализировать необычные ситуации и аргументировать свою точку зрения. Подумай нестандартно, рассмотри вопрос с разных сторон и ответь, опираясь на логику и здравый смысл. Обоснуй свой ответ."
}
evaluator.run_evaluation(
    questions_json="prompt_templates/questions.json",
    excel_prefix="prompt_templates/eval_",
    json_output="prompt_templates/full.json",
    additional_instructions=additional_instructions,
    temperature=0.7,
    max_length=1000
)

#  С инструкциями для нескольких вопросов
# additional_instructions = {
#     "Вопрос 1": "Опиши архитектуру решения в виде схемы (текст + ASCII-арт). Укажи основные компоненты и потоки данных.",
#     "Вопрос 3":
    

# #Простой запуск (базовые параметры)
# evaluator.run_evaluation(
#     questions_json="augment_service/prompt_templates/questions.json",
#     excel_prefix="results_",
#     json_output="all_results.json"
# )

# # С изменёнными параметрами генерации

# evaluator.run_evaluation(
#     questions_json="augment_service/prompt_templates/questions.json",
#     excel_prefix="eval_",
#     json_output="full_evaluation.json",
#     temperature=0.5,           # более «консервативные» ответы
#     max_length=1500
# )