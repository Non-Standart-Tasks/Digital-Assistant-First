from agents import Agent, InputGuardrail,GuardrailFunctionOutput, Runner,WebSearchTool
from pydantic import BaseModel
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="sk-13dd9563da2d435ebd38c2693dd78c6f", base_url="https://api.deepseek.com")

restaurant_agent = Agent(
    name="Рестораны",
    instructions="""Вы — цифровой помощник сервиса ВТБ Консьерж, нацеленный на предоставление точной, творчески оформленной информации. Если ответа на вопрос нет в контексте, честно сообщите об этом, но предложите альтернативные пути решения или дополнительные варианты, если это возможно.

    Используйте историю сообщений для построения ответа, если она предоставлена:
    {context}

    Не генерируй галлюцинации. Если информации нет в контексте, то не предоставляй другую.
    
    ВАЖНО: При ответе на запрос о ресторанах используй следующий формат для представления информации о каждом заведении:

    Название: [название заведения]
    Адрес: [полный адрес]
    Режим работы: [часы работы, если есть данные]
    Тип кухни: [какая кухня представлена]
    Средний чек: [стоимость среднего чека, если есть данные]
    Сайт: [официальный сайт, если есть]
    Сайт на рейтинг: [ссылка на страницу с рейтингом]
    Ссылка на отзывы: [ссылка на отзывы]

    Представляй информацию о каждом заведении в этом формате, с разделением и ясной структурой.
    """,
    tools=[
        WebSearchTool(search_context_size='low')],
    model="gpt-4o-mini"
)

other_agent = Agent(
    name="Другое",
    instructions="""Если запрос не связан с ресторанами, то выводи "насрать" """,
    model="gpt-4o-mini"
)

classify_agent = Agent(
    name="Классификация",
    instructions="""определи агента, который должен обработать запрос""",
    handoffs=[restaurant_agent, other_agent],
    model="gpt-4o-mini"
)

async def main():
    result = await Runner.run(classify_agent, "Лучшие  в мире")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())