import docker
import time
import logging
import pytest
import requests
import re
import xmlrpc.client
from docker.errors import BuildError, APIError, ImageNotFound

# Инициализация логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Настройка консольного обработчика логов
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Настройка форматирования логов
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Добавление обработчика к логгеру
logger.addHandler(console_handler)


@pytest.fixture(scope="module")
def docker_client():
    """Инициализация Docker клиента."""
    client = docker.from_env()
    yield client
    client.close()  # Закрываем клиент после завершения тестов


@pytest.fixture(scope="module")
def build_image(docker_client):
    """Сборка Docker-образа для тестов или использование уже существующего."""
    image_tag = "my-ollama-app"
    try:
        # Пытаемся получить образ по тегу
        logger.debug(f"Проверяем наличие Docker-образа с тегом '{image_tag}'...")
        image = docker_client.images.get(image_tag)
        logger.info(
            f"Образ '{image_tag}' уже существует. Используем существующий образ."
        )
        return image
    except ImageNotFound:
        logger.info(f"Образ '{image_tag}' не найден. Начинаем сборку образа...")
        try:
            image, logs = docker_client.images.build(path=".", tag=image_tag, rm=True)
            for log in logs:
                if "stream" in log:
                    logger.debug(log["stream"].strip())
            logger.info(f"Успешно собран образ '{image_tag}'.")
            return image
        except (BuildError, APIError) as e:
            logger.error(f"Не удалось собрать Docker-образ: {str(e)}")
            pytest.fail(f"Не удалось собрать Docker-образ: {str(e)}")
    except APIError as e:
        logger.error(f"Ошибка API Docker: {str(e)}")
        pytest.fail(f"Ошибка API Docker: {str(e)}")


@pytest.fixture(scope="module")
def start_container(docker_client, build_image):
    """Запуск контейнера для тестов."""
    container = docker_client.containers.run(
        build_image.id, detach=True, ports={"8080/tcp": 8080}
    )
    time.sleep(10)  # Даем время для запуска сервиса
    yield container
    container.stop()
    container.remove()


@pytest.fixture(scope="module")
def supervisor_client():
    """Подключение к Supervisor через XML-RPC."""
    try:
        server = xmlrpc.client.ServerProxy("http://127.0.0.1:9001/RPC2")
        # Проверка доступности Supervisor
        server.supervisor.getVersion()
        return server
    except Exception as e:
        pytest.fail(f"Не удалось подключиться к Supervisor: {str(e)}")


@pytest.fixture(scope="module")
def ensure_ollama_pull_success(start_container):
    """Проверка, что процесс ollama_pull завершился успешно внутри контейнера."""
    timeout = 30000  # Максимальное время ожидания в секундах
    check_interval = 10  # Интервал проверки в секундах

    start_time = time.time()
    while time.time() - start_time < timeout:
        container_status = start_container.logs().decode("utf-8").strip()
        logger.debug(f"Текущий статус процесса 'ollama_pull': {container_status}")

        # Проверка, завершился ли процесс успешно
        if "exited" in container_status and "exit status 0" in container_status:
            logger.info("Процесс 'ollama_pull' успешно завершился.")
            return True

        time.sleep(check_interval)

    # Если время ожидания истекло, и процесс не завершился успешно
    logger.error("Время ожидания завершения процесса 'ollama_pull' истекло.")
    pytest.fail("Время ожидания завершения процесса 'ollama_pull' истекло.")


def test_image_build(build_image):
    """Тестируем успешность сборки образа."""
    assert build_image is not None, "Сборка Docker-образа не удалась"


def test_ollama_service_running(start_container):
    """Тестируем, что сервис Ollama работает внутри контейнера, проверяя логи FastAPI."""
    # Выполнение команды tail -n 100 для получения последних 100 строк из логов
    exit_code, output = start_container.exec_run("tail -n 100 /var/log/fastapi_err.log")

    # Декодирование байтового вывода в строку
    container_logs = output.decode("utf-8")

    # Логирование полученных логов для отладки
    logger.debug(f"Содержимое /var/log/fastapi_err.log:\n{container_logs}")

    # Проверка наличия строки, указывающей на успешный запуск Uvicorn
    assert "Uvicorn running on http://0.0.0.0:8080" in container_logs, (
        "Сервер Ollama не был запущен успешно",
        container_logs,
    )


def test_send_request_after_ollama_pull(ensure_ollama_pull_success):
    """Отправка запроса на сервер FastAPI после успешного завершения ollama_pull."""
    try:
        logger.info("Отправка POST запроса на эндпоинт /message.")
        response = requests.post(
            url="http://127.0.0.1:8080/message",
            json={
                "user_id": 4,
                "message": "Как установить ВТБ онлйан на Андроид? Дай короткую инструкцию",
            },
        )
        logger.debug(f"Получен ответ: {response.status_code} - {response.text}")
        assert (
            response.status_code == 200
        ), f"Ожидался статус 200, получен {response.status_code}"

        # Дополнительные проверки содержимого ответа
        response_data = response.json()

        assert "user_id" in response_data, "Key 'user_id' is missing in the response"
        assert "message" in response_data, "Key 'message' is missing in the response"

        logger.info("Отправка POST запроса на эндпоинт /message прошла успешно.")
        logger.info("Запрос успешно обработан сервером FastAPI.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при отправке запроса: {str(e)}")
        pytest.fail(f"Ошибка при отправке запроса: {str(e)}")
    except ValueError:
        logger.error("Не удалось декодировать JSON ответ от сервера.")
        pytest.fail("Не удалось декодировать JSON ответ от сервера.")
