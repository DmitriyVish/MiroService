import sys
import os
import pandas as pd
from datetime import datetime
from augment_service.other.augment_module import TaskGenerator



class QuestionEvaluator:
    """
    Класс для генерации ответов на вопросы от разных "агентов" с сохранением в Excel.
    Позволяет задавать параметры генерации (temperature, max_length и др.) для каждого вопроса.
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

    def __init__(self, model_name: str = "qwen3-4b-instruct-2507"):
        self.generator = TaskGenerator(model_name)
        self.setup_sys_path()

    def setup_sys_path(self):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)

    def generate_prompt(self, question_text: str, agent_type: str) -> str:
        role_desc = self.AGENT_ROLES.get(agent_type, {}).get("description", "агента")
        prompt = self.ANSWER_PROMPT_TEMPLATE.replace("{QUESTION_TEXT}", question_text)
        prompt = prompt.replace("{AGENT_ROLE_DESCRIPTION}", role_desc)
        return prompt

    def evaluate_question(
        self,
        question: str,
        output_filename: str,
        temperature: float = 0.7,
        max_length: int = 1000,
        num_return_sequences: int = 1,
        **kwargs  # Для возможных дополнительных параметров generate_tasks
    ):
        """
        Генерирует ответы для всех агентов по заданному вопросу и сохраняет в Excel.
        
        Args:
            question: текст вопроса.
            output_filename: имя выходного Excel‑файла.
            temperature: температура генерации (от 0.0 до 1.0+).
            max_length: максимальная длина ответа в токенах.
            num_return_sequences: количество генерируемых ответов на промпт (обычно 1).
            **kwargs: дополнительные аргументы для generator.generate_tasks().
        """
        excel_data = []
        generation_time = datetime.now()

         # Генерация промпта
        for agent in self.AGENT_ROLES:
            agent_prompt = self.generate_prompt(question, agent)
            print(f"\n--- ПРОМПТ ДЛЯ {agent} ---")
            print(agent_prompt)

            # Генерация ответа с переданными параметрами
            response = self.generator.generate_tasks(
                agent_prompt,
                num_return_sequences=num_return_sequences,
                max_length=max_length,
                temperature=temperature,
                **kwargs  # Передаём дополнительные параметры
            )[0]

            # --- Сохраняем данные для Excel ---
            excel_data.append({
                "Дата и время генерации": generation_time,
                "Промпт": agent_prompt,
                "Ответ": response.strip(),
                "Роль агента": agent
            })

        # Сохранение в Excel
        df = pd.DataFrame(excel_data)
        df.to_excel(output_filename, index=False, engine='openpyxl')
        print(f"\n--- Результаты сохранены в файл: {output_filename} ---")

        # Сводка в консоль
        print("\n--- Сводка ответов ---")
        for item in excel_data:
            agent_key = item["Роль агента"]
            answer = item["Ответ"]
            print(f"[{agent_key}]: {answer}")
        print('-!'* 40)



# --- Примеры использования ---
if __name__ == "__main__":
    evaluator = QuestionEvaluator(model_name="qwen3-4b-instruct-2507")

    # Вопрос 1: анализ схемы БД (стандартные параметры)
    QUESTION_1 = """Представьте, что вы устроились работать Дата-инженером..."""
    evaluator.evaluate_question(
        question=QUESTION_1,
        output_filename="results_question_1.xlsx",
        temperature=0.7,
        max_length=1000
    )

    # Вопрос 4: логическая задачка (повышенная креативность)
    QUESTION_4 = "В некоторой комнате на пол уронили карандаш. Объясните, почему вы не можете через него перепрыгнуть?"
    evaluator.evaluate_question(
        question=QUESTION_4,
        output_filename="results_question_4.xlsx",
        temperature=0.9,      # Более креативные ответы
        max_length=200,     # Короткий ответ (1–2 предложения)
        num_return_sequences=1
    )

    # Вопрос 5: можно задать свои параметры и kwargs
    QUESTION_5 = "Как бы вы оптимизировали ETL-процесс для потоковой обработки данных?"
    evaluator.evaluate_question(
        question=QUESTION_5,
        output_filename="results_question_3.xlsx",
        temperature=0.6,
        max_length=1500,
        do_sample=True,        # Передаётся через **kwargs в generate_tasks
        top_k=50
    )
