# Импорты стандартной библиотеки
import os
import pprint
import json
import time
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from QA import QA_dictinary
from utils.kv_faiss import KeyValueFAISS
from utils.paths import LOG_DIR
from ollama import Client  # Импортируем Client

from utils.logging import setup_logging, log_api_call

logger = setup_logging(logging_path="../logs/digital_assistant.log")

# Загружаем переменные окружения из файла .env
load_dotenv()

# Получаем токен из переменных окружения
token = os.getenv("HF_API_TOKEN")
if not token:
    raise ValueError(
        "HF_API_TOKEN не найден в переменных окружения. Пожалуйста, установите его в файле .env."
    )

# Входим в Hugging Face
login(token=token)


def generate_response(
    query: str = "как настроить VTB Pay Drupal",
    model: str = "llama3.2:latest",
    temperature: float = 0.1,
    save_log: bool = True,
    show_log: bool = True,
    return_answer: bool = False,
    question_type: str = "All",
    config=None,
) -> dict:
    """
    Генерация ответа на запрос с использованием библиотеки LangChain.

    Аргументы:
        query (str): Запрос, для которого нужно сгенерировать ответ.
        model (str): Название модели Hugging Face, которая будет использоваться для генерации ответа.
        temperature (float): Температура для генерации ответа.
        save_log (bool): Сохранять ли ответ в файл журнала.
        show_log (bool): Печатать ли ответ в консоль.
        return_answer (bool): Возвращать ли сгенерированный ответ.
        question_type (str): Тип вопросов для извлечения из QA_dictinary.
        config (dict): Конфигурационный словарь.

    Возвращает:
        dict: Словарь, содержащий ответ, использованную модель и время выполнения.
    """
    try:
        splitter_type = config["Splitter"]["Type"]
        assert splitter_type in ["character", "json"]

        # Создаем клиент Ollama
        ollama_client = Client()

        # Получаем список доступных моделей
        data = ollama_client.list()["models"]

        names = [item["name"] for item in data]

        available_models = names

        # Если модель не найдена, загружаем её
        if model not in available_models:
            print(f"Модель '{model}' не найдена. Загружаем...")
            ollama_client.pull(model)

        # Инициализируем модель Ollama
        llm = Ollama(model=model, temperature=temperature)

        # Загружаем и подготавливаем документы
        loader = TextLoader("content/global_js.json")
        documents = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name=config["Embedding"])

        # Разбиваем документы на части и создаем хранилище FAISS
        if splitter_type == "character":
            docs = CharacterTextSplitter(
                chunk_size=int(config["Splitter"]["Chunk_size"]), chunk_overlap=0
            ).split_documents(documents)

            vector_store = FAISS.from_documents(docs, embeddings)
        elif splitter_type == "json":
            logical_chunks: dict = json.loads(documents[0].page_content)
            kv_dict = {k: f"{k}:\n{str(v)}" for k, v in logical_chunks.items()}
            docs = [Document(k) for k, _ in logical_chunks.items()]

            vector_store = KeyValueFAISS.from_documents(
                docs, embeddings
            ).add_value_documents(kv_dict)

        # Настройка цепочки RetrievalQA
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="refine",
            retriever=vector_store.as_retriever(),
        )

        # Запрос к системе
        start_time = time.time()
        response = rag_chain.invoke(query)
        end_time = time.time()

        # Логируем успешный ответ
        log_api_call(logger, source="LLM", request=query, response=response["result"])

        execution_time = end_time - start_time

        # Если show_log=True, выводим ответ
        if show_log:
            docs = rag_chain.retriever.vectorstore.similarity_search(query)
            pprint.pprint(f"query: {query}")
            pprint.pprint(f"context:\n" + "\n".join([doc.page_content for doc in docs]))
            pprint.pprint(f"response: {response['result']}")

        # Если save_log=True, сохраняем ответ в файл журнала
        if save_log:
            response_data = {
                "model": model,
                "result": response["result"],
                "query": query,
                "human_result": QA_dictinary.get(question_type, {}).get(
                    query, "No human result found."
                ),
                "temperature": temperature,
                "execution_time": f"{execution_time:.2f} seconds",
            }

            output_file_path = LOG_DIR / "response.json"

            # Загружаем существующие данные журнала, если они есть
            if output_file_path.exists():
                with open(output_file_path, "r", encoding="utf-8") as file:
                    try:
                        existing_data = json.load(file)
                        if isinstance(existing_data, list):
                            all_responses = existing_data
                        elif isinstance(existing_data, dict):
                            all_responses = [existing_data]
                    except json.JSONDecodeError:
                        all_responses = []
            else:
                all_responses = []

            # Добавляем новый ответ
            all_responses.append(response_data)

            # Записываем обратно в файл журнала
            with open(output_file_path, "w", encoding="utf-8") as file:
                json.dump(all_responses, file, ensure_ascii=False, indent=4)

            print(f"Ответ сохранен в {output_file_path}")

        if return_answer:
            return response["result"]

    except Exception as e:
        # Логируем ошибку
        log_api_call(logger, source="LLM", request=query, response="", error=str(e))
        raise


def get_inference(
    model: str = "llama3.2:latest",
    temperature: float = 0.1,
    query: str = "Как установить ВТБ Онлайн на Android",
    return_answer: bool = False,
) -> dict:
    """
    Упрощенная функция для получения ответа от модели LLM.

    Аргументы:
        model (str): Название модели Hugging Face, которая будет использоваться для генерации ответа.
        temperature (float): Температура для генерации ответа.
        query (str): Запрос, для которого нужно сгенерировать ответ.

    Возвращает:
        dict: Словарь, содержащий ответ, использованную модель и время выполнения.
    """
    default_config = {
        "Embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "Splitter": {
            "Type": "character",
            "Chunk_size": 4000,
        },
    }
    return generate_response(
        model=model,
        temperature=temperature,
        query=query,
        show_log=False,
        return_answer=return_answer,
        config=default_config,
    )
