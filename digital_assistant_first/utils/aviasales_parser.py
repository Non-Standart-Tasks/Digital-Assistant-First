import json
from typing import Optional, List, Dict
from datetime import date
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, Controller, BrowserConfig
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Post(BaseModel):
    airline: str
    price: str
    departure_time: str
    arrival_time: str
    depature_airport: str
    arrival_airport: str
    departure_date: str
    arrival_date: str
    return_date: str = ''
    number_transfers : str = ''
    baggage_in_price: bool
    baggage_additional_price: str

class Posts(BaseModel):
    posts: List[Post]

class AviasalesHandler:
    def __init__(self):
        self.config = BrowserConfig(headless=True, disable_security=True)
        self.browser = Browser(config=self.config)
        self.controller = Controller(output_model=Posts)

    def format_model_output(self, response):
        if hasattr(response, "content"):
            content = response.content
        elif hasattr(response, "message"):
            content = response.message.content
        else:
            content = str(response)
        result = content.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
        result = result.replace("'", '"')
        result = result.replace(',"', '", "')
        try:
            json_result = json.loads(result)
        except Exception as e:
            print(f'Exception: \'{result}\'')
            json_result = result

        print("Result: ", json_result)
        return json_result
    
    def load_airports(self) -> Dict[str, str]:
        airports = {}
        with open('airport-codes.json', 'r', encoding='utf-8') as f:
            try:
                airport_list = json.load(f)
                for airport in airport_list:
                    if isinstance(airport, dict) and 'name' in airport and 'code' in airport:
                        airports[airport['name'].lower()] = airport['code']
            except json.JSONDecodeError as e:
                print(f'Error decoding JSON: {e}')
        return airports

    def get_airport_codes(self, text: str, model, config) -> List[Optional[str]]:
        formatted_prompt = config["system_prompt_airport"].format(input=text)
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": text}
        ]
        
        response = model.invoke(messages, stream=False)
        cities_data = self.format_model_output(response)
        cities = [cities_data['city1'], cities_data['city2']]
        
        airports = self.load_airports()
        
        codes = []
        for city in cities:
            city_lower = city.lower()
            if city_lower == 'notfound':
                codes.append(None)
                continue
            found = False
            for airport_name, code in airports.items():
                if city_lower in airport_name:
                    codes.append(code)
                    found = True
                    break
            if not found:
                codes.append(None)
        return cities, codes

    def construct_aviasales_url(
        self,
        from_city: str,
        to_city: str,
        depart_date: str,
        return_date: str,
        adult_passengers: int = 1,
        child_passengers: int = 0,
        travel_class: str = "",
    ) -> Optional[str]:
        """Construct Aviasales URL based on parameters"""
        try:
            if child_passengers == 0:
                child_passengers = ''
            class_suffix = (
                travel_class + str(adult_passengers) + str(child_passengers) if travel_class else str(adult_passengers) + str(child_passengers)
            )
            aviasales_url = f"https://www.aviasales.ru/search/{from_city}{depart_date}{to_city}{return_date}{class_suffix}"
            aviasales_url = aviasales_url.replace(" ", "")
            return aviasales_url
        except Exception as e:
            print(f"Error constructing URL: {e}")
            return None

    def aviasales_request(self, model, config, user_input):
        _, airport_codes = self.get_airport_codes(text=user_input, model=model, config=config)
        formatted_prompt_tickets = config["system_prompt_tickets"].format(
            user_input=user_input,
            airport_codes=airport_codes,
            now_date=date.today()
        )
        messages = [
            {"role": "system", "content": formatted_prompt_tickets},
            {"role": "user", "content": user_input},
        ]
        response = model.invoke(messages, stream=False)
        tickets_need = self.format_model_output(response)
        return tickets_need

    async def get_info_aviasales_url_async(self, aviasales_url: str, user_input: str):
        initial_actions = [{'open_tab': {'url': aviasales_url}}]
        task = f'''Go to the site, find five flights, click on each and return detailed info about each in text format.
                   Check the price of each flight carefully. Here is the user query: {user_input}'''
        model = ChatOpenAI(model='gpt-4o')
        agent = Agent(task=task, llm=model, initial_actions=initial_actions, controller=self.controller, browser=self.browser)
        history = await agent.run()
        result = history.final_result()
        if result:
            parsed: Posts = Posts.model_validate_json(result)
            output_text = ""
            for post in parsed.posts:
                output_text += f"""
--------------------------------
Авиалинии: {post.airline}
Цена: {post.price}
Аэропорт, дата, время отправления: {post.depature_airport,post.departure_date,post.departure_time}
Аэропорт, дата и время прибытия: {post.arrival_airport,post.arrival_date,post.arrival_time}
Багаж включен в билет: {post.baggage_in_price,post.baggage_additional_price}
Дата возвращения: {post.return_date}
Кол-во пересадок: {post.number_transfers}
"""
            return output_text
        else:
            return 'No result'

    def get_info_aviasales_url(self, aviasales_url: str, user_input: str):
        import asyncio
        return asyncio.run(self.get_info_aviasales_url_async(aviasales_url=aviasales_url, user_input=user_input))
