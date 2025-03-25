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

client = AsyncOpenAI(api_key="sk-13dd9563da2d435ebd38c2693dd78c6f", base_url="https://api.deepseek.com")

set_tracing_disabled(disabled=True)

class WebSearches(BaseModel):
    searches: list[str]

class NamesOfVars(BaseModel):
    names: list[str]

class AddressOfVars(BaseModel):
    address: str
    category: str

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
    instructions="""Используй функцию fetch_user_instructions чтобы получить инуструкцию как должен выглядеть формат данных для каждого заведения. 
    Посмотри на текст и отформатируй его в единый стиль. 
    Соблюдай отступы и пробелы между пунктами.
    
    Удали все ненужные заведения из списка если это не относится к категории. 
    """,
    #model="gpt-4o-mini",
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client)
    )


agent_restaurants_critique = Agent(
    name="Assistant",
    instructions="""Посмотри на текст и отформатируй его в единый стиль. Проверь, что все пункты представлены следующим образом: Не удаляй информацию и не коверкай ее, просто сделай правильный формат если он не такой.

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
    instructions="""Посмотри на текст и отформатируй его в единый стиль. Проверь, что все пункты представлены следующим образом: Не удаляй информацию и не коверкай ее, просто сделай правильный формат если он не такой.

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
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )

agent_delivery_format = Agent(
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
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )

agent_banquet_format = Agent(
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
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )


agent_catering_format = Agent(
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
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )


agent_dict = {
    'рестораны': agent_restaurants_format,
    'бары': agent_bars_format,
    'кальянные': agent_hookah_format,
    'доставка_еды': agent_delivery_format,
    'банкет': agent_banquet_format,
    'кейтеринг': agent_catering_format
}

async def deepsearch(user_input: str, status_placeholder, config):    
    
    #UserInstructions = UserInfo(instructions=config["FORMAT_INSTRUCTIONS"]['рестораны'])
    #testing = await Runner.run(agent_testing, user_input, context=UserInstructions)
    #assert False, testing
    
    
    internet_first_step = await Runner.run(agent_internet_first_step, user_input)
    # Получаем категорию запроса и парсим адрес (чтобы если рестораны в Казани - то Казань указывалась в адресе)
    address_of_vars = await Runner.run(agent_address, user_input)
    address_of_vars, category_of_vars = address_of_vars.final_output.address, address_of_vars.final_output.category
    
    status_placeholder.info(f"🔍 Найдены следующие варианты: {internet_first_step.final_output}")
    time.sleep(1)

    # Получаем список названий заведений
    names_of_vars = await Runner.run(agent_get_names_of_vars, internet_first_step.final_output)
    names_of_vars = names_of_vars.final_output.names
    
    names_and_links = {}
    status_placeholder.info(f"🔍 Создаем интернет-запросы...")
    
    UserInstructions = UserInfo(instructions=config["FORMAT_INSTRUCTIONS"][category_of_vars])
    
    for name in names_of_vars:
        print(category_of_vars)
        links_and_addresses_of_vars = await Runner.run(agent_dict[category_of_vars], name)
        
        #links_and_addresses_of_vars = await Runner.run(agent_get_links_of_vars, name, context=UserInstructions)#
        names_and_links[name] = links_and_addresses_of_vars.final_output.links
        #assert False, names_and_links
    summarization_text = ''
    for name, links in names_and_links.items():
        for link in links:
            status_placeholder.info(f"🔍 Поиск в интернете по запросу: {link}...")
            search = DuckDuckGoSearchResults()
            text = search.invoke(link + ' ' + address_of_vars)
            time.sleep(2)
        
        summarization = await Runner.run(agent_summarization, text)
        print('text', text)
        result_summarization = summarization.final_output
        summarization_text += result_summarization

    print('sum_text', summarization_text)
    critique = await Runner.run(agent_critique, summarization_text, context=UserInstructions)
    result_critique = critique.final_output
    status_placeholder.info("✅ Предложения сформированы")
    #print('result_critique', result_critique)
    #return summarization_text
    return result_critique