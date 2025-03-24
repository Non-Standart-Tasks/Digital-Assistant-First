from agents import Agent, function_tool, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, WebSearchTool
from pydantic import BaseModel
import asyncio
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
    Также укажи категорию запроса - например "пивные рестораны", "кальянные", "бургеры", "санатории", "гостиницы" и т.д.""",
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

agent_summarization = Agent(
    name="Assistant",
    instructions="""На основе полученных данных преобразуй их в следующий формат:
    Название:
    Адрес:
    Режим работы:
    Тип кухни:
    Средний чек:
    Сайт:
    Сайт на рейтинг:
    Ссылка на отзывы:
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
    #model="gpt-4o-mini",

    model=OpenAIChatCompletionsModel(model='deepseek-reasoner', openai_client=client)
    )


async def main():
    total_start_time = time.time()
    
    user_input = "сделай подборку ресторанов москвы с хорошим выбором пива, вина, бургеров и закусок"

    step_start = time.time()
    internet_first_step = await Runner.run(agent_internet_first_step, user_input)
    print(f"Первый шаг: {time.time() - step_start:.2f} секунд")
    
    step_start = time.time()
    address_of_vars = await Runner.run(agent_address, user_input)

    #assert False, address_of_vars.final_output.category

    address_of_vars, category_of_vars = address_of_vars.final_output.address, address_of_vars.final_output.category
    #category_of_vars = address_of_vars.final_output.category


    print(f"Получение адреса: {time.time() - step_start:.2f} секунд")
    
    
    step_start = time.time()
    names_of_vars = await Runner.run(agent_get_names_of_vars, internet_first_step.final_output)
    names_of_vars = names_of_vars.final_output.names
    print(f"Получение названий: {time.time() - step_start:.2f} секунд")
    
    step_start = time.time()
    names_and_links = {}
    for name in names_of_vars:
        links_and_addresses_of_vars = await Runner.run(agent_get_links_of_vars, name)

        names_and_links[name] = links_and_addresses_of_vars.final_output.links

    print(f"Формирование поисковых запросов: {time.time() - step_start:.2f} секунд")
    
    step_start = time.time()
    summarization_text = ''
    for name, links in names_and_links.items():
        for link in links:
            print(link)
            search = DuckDuckGoSearchResults()
            text = search.invoke(link + ' ' + address_of_vars)
            time.sleep(2)
        
        summarization = await Runner.run(agent_summarization, text)
        with open("text.txt", "a", encoding="utf-8") as file:
            file.write(text + "\n\n") 
        result_summarization = summarization.final_output
        print(result_summarization)
        #assert False, result_summarization
        summarization_text += result_summarization

    critique = await Runner.run(agent_critique, summarization_text, context=category_of_vars)
    result_critique = critique.final_output
    
    print(f"Поиск в интернете: {time.time() - step_start:.2f} секунд")
    
    #step_start = time.time()
    #result_summarization = await Runner.run(agent_summarization, text)
    #print(f"Суммаризация: {time.time() - step_start:.2f} секунд")
    
    #print(summarization_text)
    print(result_critique)
    print(f"\nОбщее время выполнения: {time.time() - total_start_time:.2f} секунд")
    
    with open("text.txt", "w", encoding="utf-8") as file:
        file.write(summarization_text)

    with open("critique.txt", "w", encoding="utf-8") as file:
        file.write(result_critique)

    #text = ''
    #for i in result.final_output.searches[:1]:
    #    search = DuckDuckGoSearchResults()
    #    text += search.invoke(i)
    #result_analyst = await Runner.run(agent_analyst, text)
    #print(result_analyst.final_output)


if __name__ == "__main__":
    asyncio.run(main())