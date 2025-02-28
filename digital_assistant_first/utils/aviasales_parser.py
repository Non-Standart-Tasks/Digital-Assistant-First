import os
import json
import requests
from serpapi import GoogleSearch
from datetime import datetime
import re
from bs4 import BeautifulSoup
from typing import Optional, Dict, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from twocaptcha import TwoCaptcha

def construct_aviasales_url(
    from_city: str,
    to_city: str,
    depart_date: str,
    return_date: str,
    passengers: int = 1,
    travel_class: str = "",
) -> Optional[str]:
    """Construct Aviasales URL based on parameters"""

    try:
        # Add class suffix if specified
        class_suffix = (
            travel_class + str(passengers) if travel_class else str(passengers)
        )

        print(
            f"https://www.aviasales.ru/search/{from_city}{depart_date}{to_city}{return_date}{class_suffix}"
        )

        return f"https://www.aviasales.ru/search/{from_city}{depart_date}{to_city}{return_date}{class_suffix}"
    except Exception as e:
        print(f"Error constructing URL: {e}")
        return None

def get_llm_response(prompt):
    """Get response from LLM"""
    url = "https://gptunnel.ru/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {gpt_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": """You are a knowledgeable travel assistant. 
                You help users plan their trips by providing detailed recommendations 
                about places to visit. When providing recommendations, include relevant 
                details about each place, such as historical significance, 
                cultural importance, or unique features. Write recommendations only in RUSSIAN!!!
                Provide some information in Markdown table format.
                Add neccessary links to hotels/places/attractions!
                
                If prompt ask to analyze the prompt, then follow the instructions in it. Write analysis in ENGLISH.
                """,
            },
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Failed to get LLM response {response}")
    
# def process_query(self, user_query: str) -> str:
#     """Process user query and return formatted response"""
#     analysis_prompt = f"""
#     Analyze this travel query: "{user_query}"
#     1. What is the main destination? Write the 3-letter code (IATA code) of the city
#     2. What is the departure city? If not mentioned, assume MOW
#     3. What are the travel dates? Write only in ddmm format (for example 25 May: 2505) If not mentioned, for start date 2101 and 2701 for end date
#     4. How many passengers? Write only number. If not mentioned, assume 1
#     5. What is the preferred travel class (economy - '' /comfort - 'w'/business - 'c'/first - 'f')? If not mentioned, assume economy - ''
#     6. Are there any specific interests or preferences mentioned?
#     7. What is the budget level (budget/business/vip)?
#     Provide your analysis in JSON format only in ENGLISH with keys: "destination", "departure_city", "start_date", "end_date", 
#     "passengers", "travel_class", "interests", "budget_level"
#     """

#     try:
#         analysis = self.get_llm_response(analysis_prompt)
#         analysis = analysis.strip()
#         if analysis.startswith("```json"):
#             analysis = analysis[7:]  # Remove ```json
#         if analysis.endswith("```"):
#             analysis = analysis[:-3]  # Remove trailing ```
#         analysis = analysis.strip()
#         print(analysis)
#         analysis_data = json.loads(analysis)

#         # Get flight options
#         aviasales_url = self.construct_aviasales_url(
#             analysis_data["departure_city"],
#             analysis_data["destination"],
#             analysis_data["start_date"],
#             analysis_data["end_date"],
#             analysis_data["passengers"],
#             analysis_data.get("travel_class", ""),
#         )

#         # Search for places
#         search_results = self.search_places(analysis_data["destination"])

#         # Generate recommendations using LLM
#         recommendation_prompt = f"""
#         Create a comprehensive travel plan for {analysis_data["destination"]} based on a {analysis_data["budget_level"]} budget.
        
#         Please provide three different travel plans only in RUSSIAN!!!:

#         1. Budget Travel Plan (Economy class, hostels/budget hotels, public transport)
#         - Daily budget: $50-100 for accommodation and activities
#         - Focus on free attractions and budget-friendly options
        
#         2. Business Travel Plan (Business class, 4-star hotels, mix of transport)
#         - Daily budget: $200-500 for accommodation and activities
#         - Focus on comfort and efficiency
        
#         3. VIP Travel Plan (First class, 5-star hotels, private transport)
#         - Daily budget: $1000+ for accommodation and activities
#         - Focus on luxury experiences and exclusive access
        
#         Search results for attractions:
#         {json.dumps(search_results.get('top_sights', {}).get('sights', []), ensure_ascii=False, indent=2)}

#         For each plan include:
#         1. Accommodation options
#         2. Daily activities and attractions
#         3. Travel recommendations of transportation (in Russian "Передвижение")
#         4. Dining suggestions
#         5. Estimated total budget

#         Format each section with markdown and include practical details like:
#         - Specific hotel names and prices
#         - Restaurant recommendations with price ranges
#         - Activity costs and booking tips

#         VERY VITAL! Представь три разных плана путешествия только на РУССКОМ языке в виде таблицы с тремя столбцами: Budget, Business, VIP.
#         Формат таблицы:
#         "Категория	Budget (Эконом)	Business (Бизнес)	VIP (Люкс)"
#         Для каждого столбца (Budget, Business, VIP) укажи:
#         Конкретные названия отелей, ресторанов, достопримечательностей.
#         Примерные цены на проживание, питание, транспорт и мероприятия.
#         Практические советы по бронированию (например, ссылки на сайты или приложения).
#         VITAL: Вся информация должна быть представлена в виде таблицы, чтобы клиенту было удобно сравнивать три варианта путешествия.
#         В ТАБЛИЦЕ НЕ ПИШИ ТЕКСТ ЖИРНЫМ ШРИФТОМ ИЛИ КУРСИВОМ!!!
        
#         End with booking links and contact information for recommended services in separate text!

#         At the end add the link for checking available flights at Aviasales site: {aviasales_url}. Here customer can already find the right routes of fligtes and prices

#         Write text only in RUSSIAN!
#         """

#         screenshot_path = "images/screenshot.png"
#         self.take_screenshot(aviasales_url, screenshot_path)

#         final_response = self.get_llm_response(recommendation_prompt)

#         final_response += f" [Посмотреть на сайте]({aviasales_url})"
#         # final_response += f"\n\n[![Скриншот сайта]({screenshot_path})]({aviasales_url})\n\n"

#         return final_response, aviasales_url

#     except Exception as e:
#         print(f"Error processing query: {e}")
#         return f"Sorry, an error occurred while processing your request: {str(e)}"

def fetch_page_text(url: str) -> str:
    """Fetch the text content of the given URL and return it."""
    service = Service("/Users/ivan/Downloads/chromedriver-mac-arm64/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Add more Chrome options to make the browser more realistic
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    captcha_key = os.getenv("CAPTCHA_KEY")

    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Set page load timeout
        driver.set_page_load_timeout(30)

        # Navigate to the URL
        driver.get(url)
        print("Page loaded successfully")

        # Wait for any initial loading to complete
        time.sleep(5)

        # try:
        #     # Wait for reCAPTCHA iframe to be present (if applicable)
        #     WebDriverWait(driver, 10).until(
        #         EC.presence_of_element_located(
        #             (By.CSS_SELECTOR, 'iframe[src*="recaptcha"]')
        #         )
        #     )
        #     print("Found reCAPTCHA iframe")

        #     # Handle reCAPTCHA if detected (you can remove this part if not needed)
        #     iframe = driver.find_element(
        #         By.CSS_SELECTOR, 'iframe[src*="recaptcha"]'
        #     )
        #     iframe_src = iframe.get_attribute("src")

        #     sitekey_match = re.search(r"k=([^&]+)", iframe_src)

        #     if sitekey_match:
        #         sitekey = sitekey_match.group(1)
        #         print(f"Found sitekey: {sitekey}")

        #         solver = TwoCaptcha(captcha_key)
        #         result = solver.recaptcha(sitekey=sitekey, url=url, invisible=1)

        #         driver.execute_script(
        #             f"document.querySelector('textarea[name=\"g-recaptcha-response\"]').innerHTML = '{result['code']}';"
        #         )
        #         print("Inserted reCAPTCHA response")

        # except Exception as captcha_error:
        #     print(f"reCAPTCHA handling failed: {captcha_error}")

        # # Wait for main content to load with multiple possible conditions
        # try:
        #     WebDriverWait(driver, 30).until(
        #         EC.any_of(
        #             EC.presence_of_element_located((By.CLASS_NAME, "product-list")),
        #             EC.presence_of_element_located(
        #                 (By.CLASS_NAME, "search-results")
        #             ),
        #             EC.presence_of_element_located((By.CLASS_NAME, "tickets-list")),
        #             EC.presence_of_element_located((By.TAG_NAME, "main")),
        #         )
        #     )
        #     print("Main content loaded")
        # except Exception as e:
        #     print(f"Timeout waiting for main content: {e}")

        # Give the page more time to fully render if necessary
        # time.sleep(10)

        # Get page source and extract text content from it
        page_source = driver.page_source

        return page_source

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    finally:
        driver.quit()

def parse_aviasales(url):
    """Fetch the text content of the given URL and return it."""
    service = Service("/Users/ivan/Downloads/chromedriver-mac-arm64/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    # Add more Chrome options to make the browser more realistic
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    # Инициализация драйвера
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    
    # Ждем загрузки результатов
    time.sleep(5)
    
    try:
        # Находим все карточки с рейсами
        flight_cards = driver.find_elements(By.CSS_SELECTOR, "[data-test-id='ticket']")
        
        flights_data = []
        
        for card in flight_cards:
            try:
                # Получаем авиакомпанию
                airline = card.find_element(By.CSS_SELECTOR, "[data-test-id='carrier']").text
                
                # Получаем цену
                price = card.find_element(By.CSS_SELECTOR, "[data-test-id='price']").text
                
                # Получаем времена вылета и прилета
                departure_time = card.find_element(By.CSS_SELECTOR, "[data-test-id='departure-time']").text
                arrival_time = card.find_element(By.CSS_SELECTOR, "[data-test-id='arrival-time']").text
                
                # Получаем аэропорты
                airports = card.find_elements(By.CSS_SELECTOR, "[data-test-id='airport-code']")
                departure_airport = airports[0].text
                arrival_airport = airports[-1].text
                
                flight_info = {
                    'airline': airline,
                    'price': price,
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'departure_airport': departure_airport,
                    'arrival_airport': arrival_airport
                }
                
                flights_data.append(flight_info)
                
            except Exception as e:
                print(f"Ошибка при парсинге карточки: {e}")
                continue
                
        return flights_data
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    try:
        url = "https://www.aviasales.ru/search/MOW2105BCN27051"
        results = parse_aviasales(url)

        # Вывод результатов
        for flight in results:
            print("\nРейс:")
            print(f"Авиакомпания: {flight['airline']}")
            print(f"Цена: {flight['price']}")
            print(f"Время вылета: {flight['departure_time']}")
            print(f"Время прилета: {flight['arrival_time']}")
            print(f"Аэропорт вылета: {flight['departure_airport']}")
            print(f"Аэропорт прилета: {flight['arrival_airport']}")
    except Exception as e:
        print("Exception: " + e)
