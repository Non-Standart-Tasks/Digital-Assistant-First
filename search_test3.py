from agents import Agent, function_tool, Runner, RunContextWrapper
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import time
from dataclasses import dataclass



@dataclass
class UserInfo:
    instructions: str

@dataclass
class LinksOfVars(BaseModel):
    links: list[str]

@function_tool
async def fetch_user_instructions(wrapper: RunContextWrapper[UserInfo]) -> str:  
    return wrapper.context.instructions

UserInstructions = UserInfo(instructions=
                            
    """
                            
    Название: [название заведения]
    Адрес: [полный адрес]
    Режим работы: [часы работы, если есть данные]
    Тип кухни: [какая кухня представлена]
    Средний чек: [стоимость среднего чека, если есть данные]
    Сайт: [официальный сайт, если есть]
    Сайт на рейтинг: [ссылка на страницу с рейтингом]
    Ссылка на отзывы: [ссылка на отзывы]
                            
    """)


agent_test = Agent(
    name="Assistant",
    instructions="""сформулируй запросы по каждому пункту для поиска в интернете правильным образом.

    Название: [название заведения]
    Адрес: [полный адрес]
    Режим работы: [часы работы, если есть данные]
    Тип кухни: [какая кухня представлена]
    Средний чек: [стоимость среднего чека, если есть данные]
    Сайт: [официальный сайт, если есть]
    Сайт на рейтинг: [ссылка на страницу с рейтингом]
    Ссылка на отзывы: [ссылка на отзывы]
    
    
    Только не как ссылку уже готовую, а как запрос для поиска в интернете. Например 'Санаторий MAYRVEDA Минеральные воды адрес', 'Санаторий MAYRVEDA Минеральные воды режим работы' и т.д." 
    """,
    #tools=[fetch_user_instructions],
    model="gpt-4o-mini",
    output_type=LinksOfVars
    )


async def main():
    #assert False, UserInstructions
    result = await Runner.run(starting_agent=agent_test, input="Подбери мне 2 пивных ресторана в Казани", context=UserInstructions)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())