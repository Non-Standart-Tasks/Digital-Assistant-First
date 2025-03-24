from agents import Agent, function_tool, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, WebSearchTool, RunContextWrapper
from pydantic import BaseModel
from openai import AsyncOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import time


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

@function_tool
async def fetch_format_instructions(category: str) -> str:  
    return config["FORMAT_INSTRUCTIONS"][category]

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

agent_get_links_of_vars = Agent(
    name="Assistant",
    instructions="""На основе названия заведения, сформулируй запросы по каждому пункту для поиска в интернете правильным образом, чтобы он мог покрыть следующую информацию:
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )

agent_summarization = Agent(
    name="Assistant",
    instructions="""На основе полученных данных преобразуй их в следующий формат:
    
    {bullet_points}

    """,
    model="gpt-4o-mini",

    #model=OpenAIChatCompletionsModel(model='deepseek-reasoner', openai_client=client)
    )

agent_critique = Agent(
    name="Assistant",
    instructions="""Посмотри на текст и отформатируй его в единый стиль. 
    Не нужно писать что-то дополнительное в конце, вроде ### Изменения в стиле: и т.д.
    Учитывай категорию запроса: {category}. Удали все ненужные заведения из списка если это не относится к категории. 
    """,
    model="gpt-4o-mini",
    #model=OpenAIChatCompletionsModel(model='deepseek-reasoner', openai_client=client)
    )


