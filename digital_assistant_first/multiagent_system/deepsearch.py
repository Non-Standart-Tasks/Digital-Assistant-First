from agents import Runner
from langchain_community.tools import DuckDuckGoSearchResults
import time
from digital_assistant_first.multiagent_system.models import *

 # Определение категории запроса с помощью агента - синхронная версия
def categorize_request(user_input, model):
    category_prompt = """
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
    
    Запрос пользователя: {user_input}
    """
    
    messages = [
        {"role": "system", "content": category_prompt.format(user_input=user_input)}
    ]
    
    # Явно указываем stream=False
    response = model.invoke(messages, stream=False)
    
    if hasattr(response, "content"):
        category = response.content.strip().lower()
    elif hasattr(response, "message"):
        category = response.message.content.strip().lower()
    else:
        category = str(response).strip().lower()

    return category


async def deepsearch(user_input: str, status_placeholder, config):
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
    for name in names_of_vars:
        links_and_addresses_of_vars = await Runner.run(agent_get_links_of_vars, name)#
        names_and_links[name] = links_and_addresses_of_vars.final_output.links
    
    summarization_text = ''
    for name, links in names_and_links.items():
        for link in links:
            status_placeholder.info(f"🔍 Поиск в интернете по запросу: {link}...")
            search = DuckDuckGoSearchResults()
            text = search.invoke(link + ' ' + address_of_vars)
            time.sleep(2)
        
        summarization = await Runner.run(agent_summarization, text, context=config["FORMAT_INSTRUCTIONS"][category_of_vars])
        result_summarization = summarization.final_output
        print(config["FORMAT_INSTRUCTIONS"][category_of_vars])
        summarization_text += result_summarization

    critique = await Runner.run(agent_critique, summarization_text, context=category_of_vars)
    result_critique = critique.final_output
    status_placeholder.info("✅ Предложения сформированы")
    return result_critique