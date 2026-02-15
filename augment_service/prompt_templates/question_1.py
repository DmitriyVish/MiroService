# --- Шаблон промпта для генерации одного ответа от конкретного "агента" ---
ANSWER_PROMPT_TEMPLATE = """
Вопрос: "{QUESTION_TEXT}"

Представь, что ты {AGENT_ROLE_DESCRIPTION}.
Проходишь тестирование на должность DATA ENGINEER в крупную IT-компанию. Ответь на вопрос в соответствующей твоей роли форме.
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
    QUESTION = """Представьте, что вы устроились работать Дата-инженером в некоторый интернет магазин. 
    До вас в этой компании уже работал один разработчик, который придумал небольшую 
    базу данных для этого магазина и потом неожиданно уволился. Ваша задача провести 
    проверку существующей архитектуры и решить корректна ли она или можно внести 
    некоторые доработки. 

    Вы обратили внимание на таблицу с информацией о зарегистрировавшихся 
    покупателях (клиентах). 

    Таблица имеет следующую схему: 
    clients (  
    	client_id number, --уникальный id клиента 
    	client_name varchar(255), --имя клиента   
    	client_surname varchar(255), --фамилия клиента  
    	login varchar(30), --логин, который придумал клиент 
    	city_id number, --id города, который указал клиент (в интерфейсе выбирается название города, а в таблицу сохраняется id)	
    	age -- number, --возраст клиента 
    	reg_date -- date –дата регистрации на сайте 
    ); 

    Считаете ли вы данный набор и смысл полей корректным? Если нет, то напишите что по вашему мнению некорректно и какие изменения внесли бы."""
    
    excel_data = []
    generation_time = datetime.now() # Запоминаем время генерации

    for agent in AGENT_ROLES:
        agent_prompt = generate_prompt(QUESTION, agent)
        
        print(f"---ПРОМПТ ДЛЯ {agent}---")
        print(agent_prompt)
        
        # --- Генерация ответа ---
        response = generator.generate_tasks(
            agent_prompt,
            num_return_sequences=1,
            max_length=1000, # Уменьшено для примера, может быть больше
            temperature=0.7
        )[0]

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
    excel_filename = f"generation_results_{generation_time.strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    df.to_excel(excel_filename, index=False, engine='openpyxl') # index=False чтобы не сохранять индекс pandas
    print(f"\n--- Результаты сохранены в файл: {excel_filename} ---")

    # --- Вывод сводки (опционально) ---
    print("\n--- Сводка ответов ---")
    for item in excel_data: # Используем данные из списка, чтобы не держать в памяти отдельный словарь responses
        agent_key = item["Роль агента"]
        answer = item["Ответ"]
        print(f"[{agent_key}]: {answer}")
    print("------------------------")
