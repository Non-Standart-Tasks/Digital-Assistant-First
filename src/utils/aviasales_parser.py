import json
from typing import Optional, List
from datetime import date
from PIL import Image
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
    return_date: str = ''
    baggage: str
    baggage_additional_price: str
    url_fliht: str
    summary_info: str

class Posts(BaseModel):
    posts: List[Post]

class AviasalesHandler:
    def __init__(self):
        self.config = BrowserConfig(headless=True, disable_security=True)
        self.browser = Browser(config=self.config)
        self.controller = Controller(output_model=Posts)

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
        formatted_prompt_tickets = config["system_prompt_tickets"].format(
            user_input=user_input, now_date=date.today()
        )
        messages = [
            {"role": "system", "content": formatted_prompt_tickets},
            {"role": "user", "content": user_input},
        ]
        response = model.invoke(messages, stream=False)
        if hasattr(response, "content"):
            content = response.content
        elif hasattr(response, "message"):
            content = response.message.content
        else:
            content = str(response)
        analysis = content.strip()
        if analysis.startswith("```json"):
            analysis = analysis[7:]
        if analysis.endswith("```"):
            analysis = analysis[:-3]
        analysis = analysis.strip()
        tickets_need = json.loads(analysis)
        return tickets_need

    async def get_info_aviasales_url_async(self, aviasales_url: str):
        initial_actions = [{'open_tab': {'url': aviasales_url}}]
        task = 'Go to the site, find five flights and return detailed info about all of them in text format.'
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
airline: {post.airline}
price: {post.price}
departure_time: {post.departure_time}
arrival_time: {post.arrival_time}
depature_airport: {post.depature_airport}
arrival_airport: {post.arrival_airport}
departure_date: {post.departure_date}
return_date: {post.return_date}
baggage: {post.baggage}
baggage_additional_price: {post.baggage_additional_price}
url_fliht: {post.url_fliht}
summary_info: {post.summary_info}
"""
            return output_text
        else:
            return 'No result'

    def get_info_aviasales_url(self, aviasales_url: str):
        import asyncio
        return asyncio.run(self.get_info_aviasales_url_async(aviasales_url))
