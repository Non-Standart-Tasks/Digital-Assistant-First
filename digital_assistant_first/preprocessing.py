# Импорты стандартной библиотеки
import os
import json
import pymupdf
import logging
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # Обновленный импорт
from langchain_community.llms import Ollama
from pathlib import Path
from utils.paths import CONTENT_DIR

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Константы для путей и конфигураций
PDF_DIR = CONTENT_DIR / 'pdf'
OUTPUT_JSON = CONTENT_DIR / 'global_function.json'

# Функция для обработки одного PDF файла
def process_single_pdf(pdf_file: Path) -> str:
    """Извлечение и очистка текста из PDF файла."""
    try:
        with pymupdf.open(os.path.normpath(pdf_file)) as doc:
            text = chr(12).join([page.get_text() for page in doc])
            paragraphs = text.split('\x0c')
            formatted_paragraphs = [
                ' '.join(paragraph.replace('\n', ' ').replace('\xa0', ' ').split())
                for paragraph in paragraphs
            ]
            return '\n\n'.join(formatted_paragraphs)
    except Exception as e:
        logging.error(f"Ошибка обработки {pdf_file}: {e}")
        return ""

# Функция для генерации JSON объекта с использованием LLM
def generate_json_from_text(llm, text: str) -> dict:
    """Generate JSON from formatted text using the LLM."""
    prompt_template = """
    Ask one global question about the following text and use the whole text as the answer. Format them into a dictionary object where the question is the key, and the answer is the value.

    No line breaks or extra spaces within the dictionary structure.
    No escape characters like `\\` or `\n` in the output.
    Double quotes around both the key and the value, with all necessary special characters properly encoded.

    Example:  "Как установить ВТБ Онлайн на Андроиде?": "Перейдите в интернет-банк, нажмите на кнопку и следуйте инструкции online.vtb.ru, чтобы скачать для Android."
    
    Give me ONLY the dictionary file with only Russian text:

    {formatted_text}

    dictionary:
    """   
    prompt = PromptTemplate(input_variables=["formatted_text"], template=prompt_template)
    chain = prompt | llm

    try:
        generated_json = chain.invoke(text)
        json_dict = dict([generated_json.split(": ", 1)])  # split по первому двоеточию

        return json_dict

    except Exception as e:
        logging.error(f"Error generating JSON from text: {e}")
        return {}

# Функция для сохранения данных в JSON
def save_json_data(output_file: Path, data: dict) -> None:
    """Сохранить или добавить JSON данные в выходной файл."""
    if not data:
        return

    try:
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []  # Инициализация пустого списка, если файл не существует

        if isinstance(existing_data, list):
            existing_data.append(data)
        else:
            existing_data = [existing_data, data]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
        logging.info(f"JSON данные сохранены в {output_file}")

    except Exception as e:
        logging.error(f"Ошибка сохранения JSON данных: {e}")

# Функция для постобработки данных
def postprocess(input_path: Path = OUTPUT_JSON):
    if not input_path.exists():
        print(f"Файл не найден: {input_path}")
        return
    else:
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)


    def clean_text(text):
        return text.replace('\\"', '"').replace('\n', ' ').strip()

    # Обработка каждого элемента в списке
    cleaned_entry = {}

    for entry in data:
        for key, value in entry.items():
            # Очистка ключей и значений
            cleaned_key = clean_text(key)
            cleaned_value = clean_text(value)
                
            # Сохранение очищенных данных
            cleaned_entry[cleaned_key] = cleaned_value

    # Сохранение очищенных записей в JSON файл
    with open(input_path, 'w', encoding='utf-8') as json_file:
        json.dump(cleaned_entry, json_file, ensure_ascii=False, indent=4)

# Основная функция обработки
def process_pdf_files(path: Path, output_file: Path) -> None:
    """Обработать все PDF файлы в указанной директории."""
    llm = Ollama(model="llama3.1:8b-instruct-q8_0")

    for pdf_file in path.glob('*.pdf'):
        logging.info(f"Обработка {pdf_file}")
        formatted_text = process_single_pdf(pdf_file)

        if formatted_text:
            json_data = generate_json_from_text(llm, formatted_text)
            if json_data:
                save_json_data(output_file, json_data)

    postprocess()

if __name__ == "__main__":
    process_pdf_files(PDF_DIR, OUTPUT_JSON)