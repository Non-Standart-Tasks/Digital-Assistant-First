from agents import Runner
from langchain_community.tools import DuckDuckGoSearchResults
import time
from digital_assistant_first.multiagent_system.models import *
from dataclasses import dataclass
from agents import Agent, function_tool, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, WebSearchTool, RunContextWrapper
from pydantic import BaseModel
from openai import AsyncOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import time
from dataclasses import dataclass
import logging
from pathlib import Path
import asyncio
from functools import partial
from digital_assistant_first.utils.logging import setup_logging, log_api_call
from digital_assistant_first.utils.check_serp_response import APIKeyManager
from digital_assistant_first.internet_search import yandex_search

client = AsyncOpenAI(api_key="sk-13dd9563da2d435ebd38c2693dd78c6f", base_url="https://api.deepseek.com")
logger = setup_logging(logging_path="logs/digital_assistant.log")
serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")

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
        - ивенты (если запрос о мероприятиях, концертах, выставках, фестивалях)
        - маршруты (если запрос о том, как построить маршрут, проложить путь между местами)
        - поездки (если запрос о поездках на машинах, такси, аренде автомобилей, авиабилетах, железнодорожных билетах)
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

agent_restaurants_format = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.

    Название: 
    Адрес:
    Режим работы: 
    Тип кухни: 
    Средний чек: 
    Сайт: 
    Сайт на рейтинг: 
    Ссылка на отзывы: 
    
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )


agent_bars_format = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.

    Название: 
    Адрес:
    Режим работы: 
    Специализация: 
    Средний чек: 
    Сайт: 
    Сайт на рейтинг: 
    Ссылка на отзывы: 

    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )

agent_hookah_format = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.

    Название: 
    Адрес:
    Режим работы: 
    Ассортимент: 
    Средний чек: 
    Сайт: 
    Сайт на рейтинг: 
    Ссылка на отзывы: 
        
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )

agent_delivery_format = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.
    
    Название: 
    Тип кухни: 
    Время доставки: 
    Минимальная сумма заказа: 
    Стоимость доставки: 
    Сайт: 
    Приложение: 
    Телефон: 
    Ссылка на отзывы: 
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )

agent_banquet_format = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.
    
    Название: 
    Адрес:
    Вместимость: 
    Тип кухни: 
    Средний чек за банкет: 
    Аренда зала: 
    Сайт: 
    Контакты: 
    Ссылка на отзывы: 
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )


agent_catering_format = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.
    
    Название: 
    Специализация: 
    Минимальный заказ: 
    Стоимость: 
    Дополнительные услуги: 
    Сайт: 
    Контакты: 
    Ссылка на отзывы: 
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )


agent_get_links_of_vars = Agent(
    name="Assistant",
    instructions="""Используя fetch_user_instructions, сформулируй запросы по каждому пункту для поиска в интернете правильным образом.

    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    tools=[fetch_user_instructions],
    model="gpt-4o",
    output_type=LinksOfVars,
    )

agent_summarization = Agent(
    name="Assistant",
    instructions="""Используй функцию fetch_user_instructions чтобы получить инуструкцию как должен выглядеть формат данных для каждого заведения. 
    Посмотри на текст и отформатируй его в единый стиль. 
    Соблюдай отступы и пробелы между пунктами.
    
    Удали все ненужные заведения из списка если это не относится к категории. 
    """,
    #model="gpt-4o-mini",

    model=OpenAIChatCompletionsModel(model='deepseek-reasoner', openai_client=client)
    )

agent_critique = Agent(
    name="Assistant",
    instructions="""Сделай форматирование текста, выдели каждый пункт жирным, а информацию по нему не жирным. 
    Не добавляй никаких пояснений, просто выводи текст.
    """,
    #model="gpt-4o-mini",
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )


agent_restaurants_summarization = Agent(
    name="Assistant",
    instructions="""Расположи полученную информацию в единую структуру по пунктам, представленным ниже. Ничего от себя не добавляй. Никаких примечаний. 

    Название: 
    Адрес:
    Режим работы: 
    Тип кухни: 
    Средний чек: 
    Сайт: 
    Сайт на рейтинг: 
    Ссылка на отзывы: 
    
    Учти, что в ответе может быть только ОДНО заведение.
    """,
    #model="gpt-4o-mini"
     model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )


agent_bars_summarization = Agent(
    name="Assistant",
    instructions="""Тебе на вход дана информация из разных интернет-запросов по запросу о барах которую нужно привести в единуюу структуру по пунктам, представленным ниже.

    Название: [вставь название бара]
    Адрес: [вставь полный адрес]
    Режим работы: [часы работы, если есть данные]
    Специализация: [вставь тип бара, если указано]
    Средний чек: [вставь стоимость среднего чека, если есть данные]
    Сайт: [вставь официальный сайт, если есть]
    Сайт на рейтинг: [вставь ссылку на страницу с рейтингом]
    Ссылка на отзывы: [вставь ссылку на отзывы] 
    """,
    #model="gpt-4o-mini"
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )

agent_hookah_summarization = Agent(
    name="Assistant",
    instructions="""Посмотри на текст и отформатируй его в единый стиль. Проверь, что все пункты представлены следующим образом: Не удаляй информацию и не коверкай ее, просто сделай правильный формат если он не такой.

    Название: 
    Адрес:
    Режим работы: 
    Ассортимент: 
    Средний чек: 
    Сайт: 
    Сайт на рейтинг: 
    Ссылка на отзывы: 
        
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    #model="gpt-4o-mini"
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )

agent_delivery_summarization = Agent(
    name="Assistant",
    instructions="""Посмотри на текст и отформатируй его в единый стиль. Проверь, что все пункты представлены следующим образом: Не удаляй информацию и не коверкай ее, просто сделай правильный формат если он не такой.
    
    Название: 
    Тип кухни: 
    Время доставки: 
    Минимальная сумма заказа: 
    Стоимость доставки: 
    Сайт: 
    Приложение: 
    Телефон: 
    Ссылка на отзывы: 
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    #model="gpt-4o-mini"
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )

agent_banquet_summarization = Agent(
    name="Assistant",
    instructions="""Посмотри на текст и отформатируй его в единый стиль. Проверь, что все пункты представлены следующим образом: Не удаляй информацию и не коверкай ее, просто сделай правильный формат если он не такой.
    
    Название: 
    Адрес:
    Вместимость: 
    Тип кухни: 
    Средний чек за банкет: 
    Аренда зала: 
    Сайт: 
    Контакты: 
    Ссылка на отзывы: 
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    #model="gpt-4o-mini"
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )


agent_catering_summarization = Agent(
    name="Assistant",
    instructions="""Посмотри на текст и отформатируй его в единый стиль. Проверь, что все пункты представлены следующим образом: Не удаляй информацию и не коверкай ее, просто сделай правильный формат если он не такой.
    
    Название: 
    Специализация: 
    Минимальный заказ: 
    Стоимость: 
    Дополнительные услуги: 
    Сайт: 
    Контакты: 
    Ссылка на отзывы: 
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    #model="gpt-4o-mini"
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )


agent_dict = {
    'рестораны': [agent_restaurants_format, agent_restaurants_summarization],
    'бары': [agent_bars_format, agent_bars_summarization],
    'кальянные': [agent_hookah_format, agent_hookah_summarization],
    'доставка_еды': [agent_delivery_format, agent_delivery_summarization],
    'банкет': [agent_banquet_format, agent_banquet_summarization],
    'кейтеринг': [agent_catering_format, agent_catering_summarization]
}


category_rules = {
    'рестораны': """Адрес:
                    Режим работы: 
                    Тип кухни: 
                    Средний чек: 
                    Сайт: 
                    Сайт на рейтинг: 
                    Ссылка на отзывы: """,
    'бары': """Адрес:
                    Режим работы: 
                    Специализация: 
                    Средний чек: 
                    Сайт: 
                    Сайт на рейтинг: 
                    Ссылка на отзывы: """,
    'кальянные': """Адрес:
                    Режим работы: 
                    Ассортимент: 
                    Средний чек: 
                    Сайт: 
                    Сайт на рейтинг: 
                    Ссылка на отзывы: """,
    'доставка_еды': """Тип кухни: 
                        Время доставки: 
                    Минимальная сумма заказа: 
                    Стоимость доставки: 
                    Сайт: 
                    Приложение: 
                    Телефон: 
                    Ссылка на отзывы: """,
    'банкет':       """Адрес:
                    Вместимость: 
                    Тип кухни: 
                    Средний чек за банкет: 
                    Аренда зала: 
                    Сайт: 
                    Контакты: 
                    Ссылка на отзывы:""",
    
    'кейтеринг': """ 
                    Специализация: 
                    Минимальный заказ: 
                    Стоимость: 
                    Дополнительные услуги: 
                    Сайт: 
                    Контакты: 
                    Ссылка на отзывы: """,
    'другое':       """Сайт:
                    Контакты: """
}

async def fetch_internet_data(link, address_of_vars):
    _, serpapi_key = serpapi_key_manager.get_best_api_key()
        
    yandex_res = await yandex_search(link + ' ' + address_of_vars, serpapi_key)
    return str(yandex_res)

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

async def search_link(link: str, address: str, status_placeholder, logger) -> str:
    try:
        status_placeholder.info(f"🔍 Поиск в интернете по запросу: {link}...")
        search = DuckDuckGoSearchResults()
        text = search.invoke(link + ' ' + address)
        await asyncio.sleep(2)  # Заменяем time.sleep на asyncio.sleep
        return text
    except Exception as e:
        logger.error(f"Error during search for {link}: {str(e)}")
        return ""

async def process_establishment(name: str, links: list[str], address_of_vars: str, 
                             category_of_vars: str, agent_dict: dict,
                             status_placeholder, logger) -> str:
    try:
        global_text = f"Название заведения: {name}\n"  # Добавляем название в начало текста
        # Поисковые запросы делаем последовательно
        for link in links:
            try:
                status_placeholder.info(f"🔍 Поиск в интернете по запросу: {link}...")
                #search = DuckDuckGoSearchResults()
                #text = search.invoke(link + ' ' + address_of_vars)
                text = await fetch_internet_data(link, address_of_vars)
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

        И укажи {address_of_vars} в запросе чтобы более точно искать информацию.

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
            name, links, address_of_vars, category_of_vars, 
            agent_dict, status_placeholder, logger
        )
        for name, links in names_and_links.items()
    ]
    
    summarization_results = await asyncio.gather(*establishment_tasks)

    summarization_text = ''.join(filter(None, summarization_results))
    status_placeholder.info(f"📝 Проверяем корректное форматирование вывода...")
    critique = await agent_critique1(summarization_text, num_of_vars=num_of_vars, client=client)

    return critique