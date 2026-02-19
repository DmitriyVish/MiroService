import json 
import os 
import re 
import regex
import chardet
import PyPDF2
import logging
from pathlib import Path 
from augment_module import TaskGenerator 
from docx import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Определение вопросов 
QUESTIONS = {
    1: """Представьте, что вы устроились работать Дата-инженером в некоторый интернет магазин.
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

Считаете ли вы данный набор и смысл полей корректным? Если нет, то напишите что по вашему мнению некорректно и какие изменения внесли бы.""",
    2: """Представьте,что вы устроились работать Дата-инженером в некоторый интернет магазин.
До вас в этой компании уже работал один разработчик, который придумал небольшую базу данных для этого магазина и потом неожиданно уволился.
Ваша задача провести проверку существующей архитектуры и решить корректна ли она или можно внести некоторые доработки.
Вы обратили внимание на таблицу с информацией о товарах магазина.
Таблица имеет следующую схему:

items(
	item_id		number, --уникальный id товара
	item_name	varchar(255),--наименование товара
	item_cost	number –стоимость товара

);

Глядя на эту таблицу вы вспоминаете, что мы живём в рыночной экономике и цена, если не каждый день, то каждый месяц может меняться.
К примеру, сезонные повышения цен или наоборот скидки (перед началом учебного года или новым годом).
Так же вы знаете что у магазина есть отдел финансовой отчётности, в котором сотрудникам нужно строить различные отчёты (о доходах например) за разные промежутки времени,
в том числе и запрошлые месяцы идаже прошлые года.

Как бы вы доработали архитектуру таблицы, чтобы обеспечить историческое хранение стоимости товара?""",
    3: """Требуется проверить запрос на корректность и исправить там ошибки, если они есть.
Существует учебная схема HR, содержащая таблицы: employees, departments и locations
Необходимо получить все отделы, расположенные в Сеуле в которых все сотрудники, не
имеющие менеджера, зарабатывают в общей сложности более 100000.
Для решения задачи был написан следующий запрос:

SELECT
DEPARTMENT_ID,
DEPARTMENT_NAME,
SUM(SALARY) TOTAL_SALARY
FROM
EMPLOYEE E,
DEPARTMENTS D
WHERE E.DEPARTMENT_ID = D.DEPARTMENT_ID
AND MANAGER_ID = NULL
AND LOCATION_ID = (SELECT LOCATION_ID
FROM LOCATIONS
WHERE CITY = 'SEOUL')
AND SUM(SALARY) >= 100000
GROUP BY DEPARTMENT_NAME""",
    4: "В некоторой комнате на пол уронили карандаш. Объясните почему вы не можете через него перепрыгнуть?"
}

def parser_prompt(text: str, questions_dict: dict) -> str:
    """
        Создает промпт для LLM для извлечения ответов
        на 4 указанных вопроса    
    """
    logger.debug("Формирование промпта для LLM.")
    # Строки вопросов для промпта
    questions_formatted = ""
    for num, text in questions_dict.items():
        clean_text = re.sub(r'\s+', ' ', text).strip()
        questions_formatted += f"Вопрос {num}: {clean_text}\n\n"
        
    prompt = f"""
    Ты опытный Data Engineer. Ниже приведен документ, содержащий ответы на 4 вопроса. Тебе нужно найти и извлечь эти ответы.

    Вопросы:
    {questions_formatted}

    Прочитай документ внимательно и найди фрагменты, соответствующие каждому из 4 вопросов.
    Ответы могут быть пронумерованы (1., 2., 3., 4.), или просто следовать под формулировкой вопроса, или быть помечены как "Ответ 1:", "Ответ 2:" и т.д.
    Иногда ответ на один вопрос может частично находиться в тексте другого вопроса.

    Твоя задача - извлечь *только* текст ответа для каждого вопроса. Не добавляй к ответу сам вопрос или пояснения типа "Ответ:". Если ответ на какой-то вопрос отсутствует, верни для него пустую строку "".

    Верни результат строго в формате JSON:
    {{"вопрос 1": "...", "вопрос 2": "...", "вопрос 3": "...", "вопрос 4": "..."}}.

    Документ:
    ---
    {text}
    ---

    JSON:
    """
    logger.debug("Промпт сформирован.")
    return prompt

def preprocess_answer(answer_text: str, question_num: int) -> str:
    """
    Применяет правила к извлеченному ответу для его очистки.
    """
    logger.debug(f"Предобработка ответа на вопрос {question_num}")
    if not answer_text:
        logger.debug(f"Ответ на вопрос {question_num} пуст")
        return ""
    
    # Убираем лишние пробелы
    processed = answer_text.strip()
    # Для поиска используем часть текста вопроса
    question_snippet = QUESTIONS[question_num].lower()[:50].split('.')[0].strip()
    if processed.lower().startswith(question_snippet):
        # Обрезаем до конечной позиции фрагмента
        escaped_snippet = re.escape(question_snippet)
        match = re.match(re.compile(escaped_snippet, re.IGNORECASE), processed)
        if match:
            processed = processed[match.end():].strip()
            # Убираем возможные лишние слова и символы
            processed = re.sub(r'^(?:ответ\s*:?\s*|[\d]+\.\s*)', '', processed, flags=re.IGNORECASE)
            logger.debug(f"Обрезанное начало: {processed[:50]}...")
            
    # Для 3 вопроса извлекаем только блок SQL
    if question_num == 3:
        logger.debug("Применение специальной обработки для SQL-запроса (вопрос 3)")
        # Ищем SELECT до ;
        sql_match = re.search(r'(SELECT.*?;)', processed, re.IGNORECASE | re.DOTALL)
        if sql_match:
            processed = sql_match.group(1).strip()
            logger.debug(f"Найден SQL-запрос: {processed}")
        else:
            logger.debug("SQL-запрос не найден в ответе на вопрос 3")
    # Убираем лишние переносы строк внутри текста, оставляя по одному
    processed = re.sub(r'\n\s*\n', '\n', processed) 
    logger.debug(f"Предобработка ответа на вопрос {question_num} завершена: {processed[:100]}...")     
    return processed  

def llm_parser(file_path: Path, generator: TaskGenerator) -> dict:
    """
    Читает документ, отправляет промпт LLM, парсит JSON,
    затем применяет правила очистки к каждому ответу.
    """
    logger.info(f"Начало обработки файла: {file_path}")
    content = ""
    # Если файл.pdf
    if file_path.suffix.lower() == ".pdf":
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                # Извлекаем текст из всех страниц
                content = "\n".join(page.extract_text() for page in reader.pages)
            logger.info(f"Текст из PDF успешно извлечён. Длина: {len(content)} символов")
        except Exception as e:
            logger.error(f"Ошибка при чтении PDF файла {file_path}: {e}")
            return {f"вопрос {i}": "" for i in range(1, 5)}        
    
    # Если файл.docx используем python-docx
    elif file_path.suffix.lower() == '.docx':
        logger.info(f"Чтение DOCX файла: {file_path}")
        try:
            doc = Document(file_path)
            # Извлекаем текст из параграфов
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"Текст из DOCX успешно извлечён. Длина: {len(content)} символов")
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
            return {f"вопрос {i}": "" for i in range(1, 5)}
    # Читаем .txt и .md
    elif file_path.suffix.lower() in ['.txt', '.md']:
        logger.info(f"Чтение текстового файла: {file_path}")
        try:
            if not os.path.getsize(file_path):
                logger.warning(f"Файл {file_path} пуст")
                content = ""
            else:
                with open(file_path, "rb") as file:
                    result = chardet.detect(file.read())
                    encoding = result["encoding"]
                    logger.info(f"Определенная кодировка: {encoding}")
                with open(file_path, "r", encoding=encoding) as file:
                    content = file.read()
                    logger.info(f"Текст из {file_path.suffix} успешно извлечён. Длина: {len(content)} символов")
        except FileNotFoundError:
            logger.error(f"Error: File {file_path} not found")
            return {f"вопрос {i}": "" for i in range(1, 5)}
        except IOError as e:
            logger.error(f"IOError при чтении {file_path}: {e}")
            return {f"вопрос {i}": "" for i in range(1, 5)}
    else:
        logger.error(f"Неподдерживаемый тип файла: {file_path.suffix}. Поддерживаются .txt, .md, .docx, .pdf")
        return {f"вопрос {i}": "" for i in range(1, 5)}

    logger.info("Формирование промпта для LLM")
    prompt = parser_prompt(content, QUESTIONS)

    # Генерация ответа от LLM
    logger.info("Отправка запроса к LLM для генерации")
    response = generator.generate_tasks(
        prompt,
        num_return_sequences=1,
        max_length=2000,
        temperature=0.1,
        top_p=0.95
    )[0]
    logger.info("Генерация от LLM завершена")

    # Парсинг сгенрированного json
    json_str_to_parse = response.strip() # Убираем пробелы по краям
    logger.debug(f"Строка для парсинга JSON: {json_str_to_parse[:200]}...")
    try:
        # Парсим строку response как JSON
        parsed_json = json.loads(json_str_to_parse)
        logger.info("JSON успешно распарсен напрямую из ответа модели")
        # Проверяем, есть ли нужные ключи
        expected_keys = [f"вопрос {i}" for i in range(1, 5)]
        if all(key in parsed_json for key in expected_keys):
            logger.debug("Все ожидаемые ключи найдены в JSON")
            # Применяем правила очистки
            result = {}
            for i in range(1, 5):
                key = f"вопрос {i}"
                raw_answer = parsed_json.get(key, "")
                logger.debug(f"Обработка ответа на {key}: {raw_answer[:50]}...")
                # Применяем правила очистки
                clean_answer = preprocess_answer(raw_answer, i)
                result[key] = clean_answer
            logger.info("Обработка всех ответов завершена")
            return result
        else:
            logger.info("Обработка всех ответов завершена")
            # Возвращаем пустой словарь или частично заполненный, если ключи не совпадают
            return {f"вопрос {i}": "" for i in range(1, 5)}

    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON напрямую: {e}")
        logger.info("Пробуем найти JSON внутри строки...")
      
        # Убираем маркеры кода, если они есть
        cleaned_response = re.sub(r'```(?:json)?\s*', '', response).strip()
        logger.debug(f"Строка после удаления маркеров: {cleaned_response[:200]}...")
        
        # Поиск первой { и последней } (fallback)
        first_brace = cleaned_response.find('{')
        last_brace = cleaned_response.rfind('}')

        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = cleaned_response[first_brace:last_brace+1]
            logger.debug(f"Найден потенциальный JSON-блок: {json_str[:200]}...")
            try:
                parsed_json = json.loads(json_str)
                logger.info("JSON успешно распарсен после поиска скобок")
                expected_keys = [f"вопрос {i}" for i in range(1, 5)]
                if all(key in parsed_json for key in expected_keys):
                    logger.debug("Все ожидаемые ключи найдены в найденном JSON")
                    result = {}
                    for i in range(1, 5):
                         key = f"вопрос {i}"
                         raw_answer = parsed_json.get(key, "")
                         logger.debug(f"Обработка ответа на {key} из найденного JSON: {raw_answer[:50]}...")
                         # Применяем правила очистки
                         clean_answer = preprocess_answer(raw_answer, i)
                         result[key] = clean_answer
                    logger.info("Обработка всех ответов из найденного JSON завершена")
                    return result
                else:
                    logger.warning(f"Найденный JSON не содержит все ожидаемые ключи (вопрос 1-4). Ключи: {list(parsed_json.keys())}")
                    return {f"вопрос {i}": "" for i in range(1, 5)}
            except json.JSONDecodeError as e2:
                logger.error(f"Ошибка парсинга найденного JSON-блока: {e2}")
                logger.debug(f"Попытка распарсить строку: {json_str[:200]}...")
        else:
            logger.warning("Не найдены фигурные скобки для извлечения JSON")

        # Если ничего не сработало
        logger.error(f"JSON не найден в ответе LLM для {file_path.name}")
        logger.debug(f"Ответ LLM: {response[:500]}...") # Печатаем начало для отладки
        return {f"вопрос {i}": "" for i in range(1, 5)}
    


path = "1.docx"
output = "1.json"

file_to_parse = Path(f"candidate_answers/{path}")   
output_filename = Path(f"parsed_answers/{output}")   
   
generator = TaskGenerator("qwen3-4b-instruct-2507")
parsing_result = llm_parser(file_to_parse, generator)
with open(output_filename, "w", encoding="utf-8") as file:
    json.dump(parsing_result, file, ensure_ascii=False, indent=4)
    logger.info(f"Результат сохранен в: {output_filename}")

    
print("Результат парсинга:")
print(json.dumps(parsing_result, ensure_ascii=False, indent=4))

'''
def parse_all_docs(generator_model: str="qwen3-4b-instruct-2507"):
    """
    Обрабатывает все подходящие файлы в папке candidate_answers,
    парсит их с помощью llm_parser и сохраняет результаты в parsed_answers.
    """
    source_dir = Path("candidate_answers")
    target_dir = Path("parsed_answers")
    
    if not source_dir.exists():
        logger.error(f"Папка {source_dir} не существует")
        return
    # Создаем папку для результатов, если ее нет
    target_dir.mkdir(exist_ok=True)
    
    # Инициализируем генератор
    generator = TaskGenerator(generator_model)
    supported_extensions = {".txt", ".md", ".docx", ".pdf"}
    
    # Парсим все доступные файлы
    for file_path in source_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            logger.info(f"Обработка файла: {file_path.name}")
            output_filename = target_dir / f"{file_path.stem}.json"
            parsing_result = llm_parser(file_path, generator)
            
            # Сохраняем результат в json
            try:
                with open(output_filename, "w", encoding="utf-8") as file:
                    json.dump(parsing_result, file, ensure_ascii=False, indent=4)
                logger.info(f"Результат сохранен в: {output_filename}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении {output_filename} : {e}")
    logger.info("Обработка всех файлов завершена")
    
if __name__ == "__main__":
    parse_all_docs()
'''