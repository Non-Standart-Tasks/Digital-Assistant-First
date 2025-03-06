from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
import logfire
from typing import List, Dict, Any
import json
from pathlib import Path

logfire.configure(send_to_logfire='if-token-present')

class Box(BaseModel):
    number_of_restaurants: int
    name_of_restaurants: str
    answer: str

roulette_agent = Agent(  
    'openai:gpt-4o-mini',
    result_type=Box,
    system_prompt=(
        'use the check_user_question tool to determine if the user question is related to restaurants. '
        'If it is, use the get_restaurants tool to get a list of restaurants and answer the user question. '
        'In other cases, answer the user question as best as possible.'
    )
)

@roulette_agent.tool
def check_user_question(ctx: RunContext, user_question: str) -> str:
    return f"Analyzing question: '{user_question}'"

@roulette_agent.tool
def get_restaurants(ctx: RunContext, json_path=None) -> List[Dict[str, Any]]:
    print('get_restaurants_by_category')
    """
    Загружает локальный JSON с ресторанами и возвращает отфильтрованный список
    по введенной категории.
    """
    
    if json_path is None:
        json_path = "restaurants.json"
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert False, data
        restaurants = data.get("restaurants", [])
    except Exception as e:
        print("No yandex-restaurants file in system", e)
        restaurants = []
    
    print(restaurants)
    return restaurants

result = roulette_agent.run_sync('''Хороший ресторан в Москве''')
print(result.data)