from agents import Agent, function_tool, Runner, set_tracing_disabled, OpenAIChatCompletionsModel
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI
from langchain_community.tools import DuckDuckGoSearchResults



client = AsyncOpenAI(api_key="sk-13dd9563da2d435ebd38c2693dd78c6f", base_url="https://api.deepseek.com")

set_tracing_disabled(disabled=True)

class WebSearches(BaseModel):
    searches: list[str]


@function_tool
async def web_search(web_search: WebSearches) -> str:
    search = DuckDuckGoSearchResults()
    return search.invoke(web_search.searches)


agent_web_search = Agent(
    name="Агент создания запросов",
    instructions="""You are a financial research planner. Given a request for financial analysis, "
    "produce a set of web searches to gather the context needed. Aim for recent "
    "headlines, earnings calls or 10‑K snippets, analyst commentary, and industry background. "
    "Output between 5 and 15 search terms to query for.
    """,
    model="gpt-4o-mini",
    #model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client),
    output_type=WebSearches
    )



agent_orchestrator = Agent(
    name="Assistant",
    instructions="""Ты агент оркестратор который направляет запрос для агента создания запросов чтобы он создал правильные запросы в интернет. 
    А потом ты с помощью инструмента web_search отправяешь эти запросы в интернет.
    """,
    model="gpt-4o-mini",
    handoffs=[agent_web_search],
    #model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client),
    )

agent_analyst = Agent(
    name="Агент анализа",
    instructions="""Ты агент анализа который анализирует текст и выдает ответ пользователю в следующем формате:
    Название санатория: 
    Адрес: 
    Класс: 
    Питание: 
    Программа лечения: 
    Необходимые документы: 
    Бюджет:
    Удобства: 
    Расположение: 
    Политика отмены бронирования: 
    Дополнительные услуги: 
    Сайт с информацией по бронированию:
    """,
    #model="gpt-4o-mini",
    model=OpenAIChatCompletionsModel(model='deepseek-chat', openai_client=client),
    )


async def main():
    result = await Runner.run(agent_orchestrator, "Подбери лучшие санатории в Минеральных водах")
    assert False, result.final_output
    
    text = ''
    for i in result.final_output.searches[:1]:
        search = DuckDuckGoSearchResults()
        text += search.invoke(i)
    result_analyst = await Runner.run(agent_analyst, text)
    print(result_analyst.final_output)


if __name__ == "__main__":
    asyncio.run(main())