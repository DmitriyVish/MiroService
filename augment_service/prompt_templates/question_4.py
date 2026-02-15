# --- Шаблон промпта для генерации одного ответа от конкретного "агента" ---
ANSWER_PROMPT_TEMPLATE = """
Вопрос: "{QUESTION_TEXT}"

Представь, что ты {AGENT_ROLE_DESCRIPTION}.
Тебя пригласили на интервью на должность DATA ENGINEER в крупную IT-компанию.
Вопрос, который ты сейчас видишь, - это нестандартная логическая задачка, призванная проверить твоё креативное мышление, способность анализировать необычные ситуации и аргументировать свою точку зрения.
Подумай нестандартно, рассмотри вопрос с разных сторон и ответь, опираясь на логику и здравый смысл. Обоснуй свой ответ. Ответь кратко (1 - 2 предложения)

Ответ:
"""

# --- Описания ролей ---
AGENT_ROLES = {
    "well_prepared_candidate": {
        "description": """хорошо подготовленный кандидат, обладающий отличными аналитическими способностями, глубоко анализирует вопрос и дает точный, логичный и обоснованный ответ""",
    },
    "moderately_prepared_candidate": {
        "description": """средне подготовленный кандидат, понимает основы, но может упустить некоторые тонкие детали или дать частично правильный, частично неуверенный ответ""",
    },
    "poorly_prepared_candidate": {
        "description": """плохо подготовленный кандидат, слабо понимает суть вопроса, может дать поверхностный, запутанный или неправильный ответ""",
    },
    "generative_ai_qwen": {
        "description": "генеративная нейронная сеть Qwen, обученная на большом объёме текстов, дающая логичный, структурированный и вежливый ответ, основанный на вероятностях"
    }
}

def generate_prompt(queston_text: str, agent_type: str) -> str:
    """
    Подбирает промпт для генерации ответа от конкретного агента.
    Args:
        question_text (str): Текст вопроса.
        agent_type (str): Ключ роли (например, 'well_prepared_candidate').
    Returns:
        str: Промпт для генерации ответа.
    """
    role_description = AGENT_ROLES.get(agent_type, {}).get("description", "агента")
    prompt = ANSWER_PROMPT_TEMPLATE.replace("{QUESTION_TEXT}", queston_text)
    prompt = prompt.replace("{AGENT_ROLE_DESCRIPTION}", role_description)
    return prompt

if __name__ == "__main__":
    import sys
    import os   
    import pandas as pd
    from datetime import datetime 
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.insert(0, parent_dir)
    
    from augment_module import TaskGenerator
    generator = TaskGenerator("qwen3-4b-instruct-2507")
    
    # Фиксированный вопрос
    QUESTION = "В некоторой комнате на пол уронили карандаш. Объясните почему вы не можете через него перепрыгнуть?"
    
    excel_data = []
    generation_time = datetime.now() # Запоминаем время генерации
    
    # Генерация промпта
    for agent in AGENT_ROLES:
        agent_prompt = generate_prompt(QUESTION, agent)
        
        print(f"---ПРОМПТ ДЛЯ {agent}---")
        print(agent_prompt)
        
        # Генерация ответа
        response = generator.generate_tasks(
            agent_prompt,
            num_return_sequences=1,
            max_length=2000,
            temperature=0.9
        )[0]

        # responses[agent] = response.strip()        
    
        # --- Сохраняем данные для Excel ---
        excel_data.append({
            "Дата и время генерации": generation_time,
            "Промпт": agent_prompt,
            "Ответ": response.strip(), # Убираем лишние пробелы
            "Роль агента": agent
        })
        
        # --- Создание DataFrame и сохранение в Excel ---
        df = pd.DataFrame(excel_data)
        # Укажите имя файла. Можно добавить путь, если нужно сохранить в другую папку.
        excel_filename = f"results_question_4.xlsx"

        df.to_excel(excel_filename, index=False, engine='openpyxl') # index=False чтобы не сохранять индекс pandas
        print(f"\n--- Результаты сохранены в файл: {excel_filename} ---")

        # --- Вывод сводки (опционально) ---
        print("\n--- Сводка ответов ---")
        for item in excel_data: # Используем данные из списка, чтобы не держать в памяти отдельный словарь responses
            agent_key = item["Роль агента"]
            answer = item["Ответ"]
            print(f"[{agent_key}]: {answer}")
        print("------------------------")