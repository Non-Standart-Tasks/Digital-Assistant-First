from agents import Runner
import time
from digital_assistant_first.multiagent_system.models import *
from dataclasses import dataclass
from agents import Agent, function_tool, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, WebSearchTool, RunContextWrapper
from pydantic import BaseModel
from openai import AsyncOpenAI
import time
from dataclasses import dataclass
import logging
from pathlib import Path
import asyncio
from digital_assistant_first.utils.logging import setup_logging
from digital_assistant_first.utils.check_serp_response import APIKeyManager
from digital_assistant_first.internet_search import yandex_search
from digital_assistant_first.utils.serperapi_apikey_selection import SerperAPIKeySelector
import requests
import json 
from streamlit_app import load_config_yaml
config = load_config_yaml()
deepseek_api_key = config['deepseek_api_key']

client = AsyncOpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
logger = setup_logging(logging_path="logs/digital_assistant.log")
# serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")
serper_api_key_selector = SerperAPIKeySelector()

set_tracing_disabled(disabled=True)

class WebSearches(BaseModel):
    searches: list[str]

class NamesOfVars(BaseModel):
    names: list[str]

class AddressOfVars(BaseModel):
    address: str
    category: str
    num_of_vars: int

class LinksOfVars(BaseModel):
    links: list[str]


@dataclass
class UserInfo:  
    instructions: str

@function_tool
async def fetch_user_instructions(wrapper: RunContextWrapper[UserInfo]) -> str:  
    return wrapper.context.instructions

agent_internet_first_step = Agent(
    name="Assistant",
    instructions="""Ответь на вопрос пользователя, используя интернет и контекст. Если запрос связан с выводом каких-либо мест, то выведи столько вариантов, сколько попросил пользовательл
    если явно количество не указано, то выведи 5 вариантов.
    """,
    model="gpt-4o-mini",
    tools=[
        WebSearchTool(search_context_size='low')]
    #model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client),
    )

agent_address = Agent(
    name="Assistant",
    instructions="""Проверь запрос и выведи адрес если он был в запросе. К примеру "Найти санатории в Казани" - укажи адрес "Казань". 
    Определи категорию запроса пользователя и верни ТОЛЬКО одну из следующих категорий без дополнительных пояснений:
        - рестораны (если запрос о ресторанах, кафе, еде в общественных местах)
        - бары (если запрос о барах, пабах, винных барах)
        - кальянные (если запрос о кальянных, кальян-барах)
        - доставка_еды (если запрос о доставке еды, заказе еды на дом)
        - банкет (если запрос о проведении банкета, юбилея, корпоратива, аренде зала для торжества)
        - кейтеринг (если запрос о выездном обслуживании, доставке готовых блюд на мероприятие)
        - спектакль (если запрос о театре, спектаклях, концертах, выставках, фестивалях)
        - концерты (если запрос о концертах, музыкальных исполнителях, музыкальных фестивалях)
        - шоу (если запрос о шоу, спектаклях, концертах, выставках, фестивалях)
        - выставки (если запрос о выставках, музеях, галереях)
        - музеи (если запрос о музеях, галереях)
        - парки_развлечений (если запрос о парках, аттракционах, развлекательных центрах)
        - подбор_такси (если запрос о подборе такси)
        - подбор_трансфера (если запрос о подборе трансфера)
        - аренда_транспорта_без_водителя (если запрос о аренде транспорта без водителя)
        - аренда_транспорта_с_водителем (если запрос о аренде транспорта с водителем)
        - маршруты (если запрос о том, как построить маршрут, проложить путь между местами)
        - поездки (если запрос о поездках на машинах, такси, аренде автомобилей, авиабилетах, перелетах из одного города в другой, железнодорожных билетах)
        - другое (если запрос не подходит ни под одну из перечисленных категорий)
    Также определи количество заведений, которые нужно вывести в ответе.
    """,
    
    
    model="gpt-4o-mini",
    output_type=AddressOfVars
    )

agent_get_names_of_vars = Agent(
    name="Assistant",
    instructions="""Получи список названий заведений из ответа пользователя.
    """,
    model="gpt-4o-mini",
    output_type=NamesOfVars
    )

agent_get_address_of_vars = Agent(
    name="Assistant",
    instructions="""Проверь запрос и выведи адрес если он был в запросе.
    """,
    model="gpt-4o-mini",
    output_type=AddressOfVars
    )


category_rules = {
    'рестораны':    """
                    Адрес:
                    Режим работы: 
                    Тип кухни: 
                    Средний чек: 
                    Сайт: 
                    Сайт на рейтинг: 
                    Ссылка на отзывы: 
                    """,
    'бары':         """
                    Адрес:
                    Режим работы:
                    Тип кухни:
                    Средний чек:
                    Сайт:
                    Сайт на рейтинг:
                    Ссылка на отзывы:
                    """,
    'кальянные':    """
                    Адрес:
                    Режим работы:
                    Средний чек:
                    Сайт:
                    Сайт на рейтинг:
                    Ссылка на отзывы:
                    """,
    'доставка_еды': """
                    Адрес:
                    Режим работы:
                    Тип кухни:
                    Средний чек:
                    Наличие депозита:
                    Наличие отдельного зала, количество человек:
                    Сайт:
                    Сайт на рейтинг:
                    Ссылка на отзывы:
                    """,
    'банкет':       """
                    Адрес:
                    Режим работы:
                    Тип кухни:
                    Средний чек:
                    Наличие депозита:
                    Наличие отдельного зала, количество человек:
                    Сайт:
                    Сайт на рейтинг:
                    Ссылка на отзывы:
                    """,
    'кейтеринг':    """ 
                    Адрес предлагаемой площадки:
                    Режим работы:
                    Минимальное количество заказов:
                    Сайт или контактная информация:
                    Специальные предложения или акции:
                    Ссылка на отзывы/портфолио:
                    """,
    'спектакли':    """
                    Название спектакля:
                    Возрастное ограничение:
                    Театр:
                    Адрес места проведения:
                    Дата и время:
                    Жанр:
                    Стоимость билета от:
                    Доступные места:
                    Сайт площадки:
                    Специальные предложения или акции:
                    Ссылка на отзывы/описание спектакля:
                    """,

    'концерты':     """
                    Исполнитель/группа:
                    Дата и время:
                    Длительность:
                    Место проведения:
                    Адрес места проведения:
                    Возрастное ограничение:
                    
                    Жанр:
                    Стоимость билета от:
                    Доступные места:
                    Сайт площадки или контактная информация:
                    Специальные предложения или акции:
                    Ссылка на отзывы/описание концерта:
                    """,

    'шоу':          """
                    Возрастное ограничение:
                    Место проведения:
                    Адрес места проведения:
                    Дата и время:
                    Возрастное ограничение:
                    Формат:
                    Длительность:
                    Стоимость билета от:
                    Доступные места:
                    Сайт площадки или контактная информация:
                    Специальные предложения или акции:
                    Ссылка на отзывы/описание шоу:
                    """,

    'выставки':     """
                    Тип выставки: 
                    Тематика: 
                    Даты и время:
                    Место проведения: 
                    Адрес места проведения:
                    Организатор: 
                    Целевая аудитория: Возрастное ограничение:
                    Стоимость билетов от и до:
                    Режим работы выставки:
                    Постоянная или временная выставка:Даты проведения выставки, если временная:
                    Сайт для покупки билетов: 
                    Дополнительные мероприятия:
                    Краткое описание выставки: 
                    Политика возврата билетов: 
                    """,
    
    'музеи':        """
                    Тип музея: 
                    Тематика: 
                    Возрастное ограничение:
                    Город: 
                    Дни и часы работы: 
                    Входные билеты: 
                    Специальные экспозиции: 
                    Услуги: 
                    Сайт музея для покупки билетов: 
                    Рекомендации: 
                    Наличие групповых экскурсий: 
                    Наличие индивидуальных экскурсий: 
                    Краткое описание:
                    Политика возврата билетов:
                    """,

    'парки_развлечений': """
                    Тип парка:  
                    Тематика:  
                    Даты и время работы парка:  
                    Адрес парка:  
                    Целевая аудитория:  Возрастное ограничение:
                    Возрастное ограничение:
                    Стоимость билетов от и до:  
                    Специальные мероприятия или аттракционы:  
                    Сайт для покупки билетов:  
                    Дополнительные услуги:  
                    Краткое описание парка:  
                    Наличие и стоимость fastpass:
                    Наличие и стоимость комплексных билетов:
                    Политика возврата билетов:
                    """,

    'подбор_такси': """
                    Сайт службы такси:
                    Контакты службы такси:
                    Основной адрес службы такси:
                    Способы заказа:
                    Стоимость поездки от и до:
                    Способы оплаты:
                    Тип автомобиля:
                    Время бесплатного ожидания:
                    Условия платного ожидания:
                    Условия отказа от поездки:
                    Специальные предложения или акции:
                    Ссылка на отзывы/описание службы:
                    """,
    'подбор_трансфера': """
                    Сайт службы трансфера:  
                    Контакты службы трансфера:
                    Стоимость поездки от и до:  
                    Стоимость детского удерживающего устройства:
                    Стоимость встречи с табличкой:
                    Способы оплаты:  
                    Тип автомобиля:  
                    Время бесплатного ожидания:  
                    Условия платного ожидания: 
                    Политика отказа от поездки: 
                    Специальные предложения или акции:  
                    Ссылка на отзывы/описание службы:  
    """,
    'аренда_транспорта_с_водителем': """
                    Сайт службы аренды:  
                    Контакты службы аренды:
                    Марка и модель автомобиля:
                    Стоимость аренды (за час/день):
                    Стоимость часа на подачу автомобиля:
                    Промежутки времени аренды:   
                    Способы оплаты:  
                    Политика отказа от аренды:  
                    Специальные предложения или акции:  
                    Ссылка на отзывы/описание службы:  
    """,
    'аренда_транспорта_без_водителя': """
                    Сайт службы аренды:  
                    Стоимость аренды (за день/неделю):  
                    Место получения автомобиля:
                    Место возврата автомобиля:
                    Дата и время забора автомобиля:
                    Дата и время сдачи автомобиля:
                    Способы оплаты:  
                    Тип автомобиля:  
                    Тип трансмиссии:
                    Условия страховки:  
                    Размер франшизы: 
                    Политика топлива (полный/пустой бак):  
                    Условия возврата автомобиля:  
                    Наличие лимита по километражу:
                    Тип топлива:
                    Порядок оплаты аренды и необходимость внесения депозита: 
                    Порядок возврата депозита:
                    Штрафная политика: 
                    Специальные предложения или акции:  
                    Ссылка на отзывы/описание службы:  
                    """,
    'другое': """
    Сайт:
    Контакты: 
    Ссылка на отзывы/описание службы:
    """
}

# async def fetch_internet_data(link, address_of_vars):
#     _, serpapi_key = serpapi_key_manager.get_best_api_key()
        
#     yandex_res = await yandex_search(link + ' ' + address_of_vars, serpapi_key)
#     return str(yandex_res)

async def fetch_serper_data(link, address_of_vars, api_key):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": f"{link} {address_of_vars}",
    "gl": "ru",
    "hl": "ru",
    "num": 10
    })
    headers = {
    'X-API-KEY': api_key,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    serper_api_key_selector.track_key(response, api_key)

    return str(response.text)

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("deepsearch")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(log_dir / "deepsearch.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


async def process_establishment(name: str, links: list[str], address_of_vars: str, 
                             category_of_vars: str,
                             status_placeholder, logger) -> str:
    try:
        global_text = f"Название заведения: {name}\n"  # Добавляем название в начало текста
        # Поисковые запросы делаем последовательно
        for link in links:
            try:
                status_placeholder.info(f"🔍 Поиск в интернете по запросу: {link}...")
                api_key_chosen = serper_api_key_selector.get_best_key()
                text = await fetch_serper_data(link, address_of_vars, api_key_chosen)
                global_text += '\n' + text
                time.sleep(3)
            except Exception as e:
                logger.error(f"Error during search for {link}: {str(e)}")
                continue
                
        if global_text:
            # Добавляем название заведения в инструкции для суммаризации
            agent = Agent(
                name="Assistant",
                instructions=f"""Суммаризируй информацию ТОЛЬКО для заведения "{name}". 
                Игнорируй любую информацию о других заведениях.
                Используй следующий формат:

                Название: {name}
                Остальные поля возьми отсюда:
                {category_rules[category_of_vars]}

                Обязательно старайся вставлять ссылки из текста, если они есть. Свои не выдумывай.
                """,
                model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
            )
            status_placeholder.info(f"📝 Готовим информацию по предложениям...")
            summarization = await Runner.run(agent, global_text)
            result = summarization.final_output
            
            # Проверяем, что в результате есть нужное название
            if name.lower() not in result.lower():
                logger.warning(f"Summarization result doesn't contain establishment name: {name}")
                return ""
                
            return result
        return ""
    except Exception as e:
        logger.error(f"Error processing establishment {name}: {str(e)}")
        return ""

async def create_agent_and_get_links(name: str, category_rules: dict, category_of_vars: str, client: AsyncOpenAI, address_of_vars: str) -> list[str]:
    agent_creating_links = Agent(
        name="Assistant",
        instructions=f"""сформулируй запросы по каждому пункту для поиска в интернете {name} правильным образом.

        Вот пункты:
        {category_rules[category_of_vars]}

        И укажи {address_of_vars} в запросе чтобы более точно искать информацию. Если адрес не указан, то не указывай его.

        Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
        """,
        model="gpt-4o-mini",
        #model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client),
        output_type=LinksOfVars
    )
    
    result = await Runner.run(agent_creating_links, name)
    return result.final_output.links

async def agent_critique1(summarization_text: str, num_of_vars: int, client: AsyncOpenAI) -> str:
    
    agent_critique = Agent(
        name="Assistant",
        instructions=f"""Сделай форматирование текста, выдели каждый пункт жирным, а информацию по нему не жирным. Не ставь пункты в виде списка или точек.
        Не добавляй никаких пояснений, просто выводи текст.

        Проверь, что заведений должно быть ровно {num_of_vars} 
        """,
        model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
        )
    
    critique = await Runner.run(agent_critique, summarization_text)
    return critique.final_output

async def deepsearch(user_input: str, status_placeholder, config):    
    logger = setup_logging()
    
    internet_first_step = await Runner.run(agent_internet_first_step, user_input)
    address_of_vars = await Runner.run(agent_address, user_input)
    address_of_vars, category_of_vars, num_of_vars = address_of_vars.final_output.address, address_of_vars.final_output.category, address_of_vars.final_output.num_of_vars
    
    status_placeholder.info(f"🔍 Найдены следующие варианты: {internet_first_step.final_output}")
    time.sleep(2)

    names_of_vars = await Runner.run(agent_get_names_of_vars, internet_first_step.final_output)
    names_of_vars = names_of_vars.final_output.names

    # Параллельно создаем агентов и получаем ссылки
    link_tasks = [
        create_agent_and_get_links(name, category_rules, category_of_vars, client, address_of_vars)
        for name in names_of_vars
    ]
    link_results = await asyncio.gather(*link_tasks)

    
    names_and_links = {
        name: links 
        for name, links in zip(names_of_vars, link_results)
    }
    
    logger.debug(f"Names and links dictionary: {names_and_links}")
    
    # Параллельно обрабатываем заведения (только агенты работают параллельно)
    establishment_tasks = [
        process_establishment(
            name, links, address_of_vars, category_of_vars, status_placeholder, logger
        )
        for name, links in names_and_links.items()
    ]
    
    summarization_results = await asyncio.gather(*establishment_tasks)

    summarization_text = ''.join(filter(None, summarization_results))
    status_placeholder.info(f"📝 Проверяем корректное форматирование вывода...")
    critique = await agent_critique1(summarization_text, num_of_vars=num_of_vars, client=client)

    return critique