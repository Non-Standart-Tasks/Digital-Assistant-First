import os
import json
import requests
from typing import Optional
from selenium.webdriver.support import expected_conditions as EC
import time
import http.client
from openai import OpenAI
import base64
from PIL import Image

from dotenv import load_dotenv

load_dotenv()


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

        aviasales_url = f"https://www.aviasales.ru/search/{from_city}{depart_date}{to_city}{return_date}{class_suffix}"
        aviasales_url = aviasales_url.replace(" ", "")

        return aviasales_url
    except Exception as e:
        print(f"Error constructing URL: {e}")
        return None


def analyze_aviasales_url(
    webpage_url,
    robot_id="09a7d6f0-2dd1-4547-b1df-b0c5963d1b86",
):
    """
    Получает скриншот веб-страницы и анализирует его через GPT-4V.

    Args:
        webpage_url (str): URL страницы для анализа
        browse_ai_key (str): API ключ Browse.ai
        openai_key (str): API ключ OpenAI
        robot_id (str): ID робота Browse.ai

    Returns:
        str: Текстовый анализ изображения от GPT-4V
    """

    openai_key = os.getenv('OPENAI_API_KEY')
    browse_ai_key = os.getenv('BROWSE_AI_KEY')
    client = OpenAI(api_key=openai_key)

    browse_api_url = f"https://api.browse.ai/v2/robots/{robot_id}/tasks"

    payload = {"inputParameters": {"originUrl": webpage_url}}
    headers = {"Authorization": f"Bearer {browse_ai_key}"}

    response = requests.post(browse_api_url, json=payload, headers=headers)
    response_data = json.loads(response.text)
    task_id = response_data["result"]["id"]

    while True:
        conn = http.client.HTTPSConnection("api.browse.ai")
        conn.request("GET", f"/v2/robots/{robot_id}/tasks/{task_id}", headers=headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))

        if data["result"]["status"] == "successful":
            break
        elif data["result"]["status"] == "failed":
            raise Exception("Failed to capture screenshot")

        time.sleep(3)

    screenshot_url = data["result"]["capturedScreenshots"]["Screenshot"]["src"]

    response = requests.get(screenshot_url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to download screenshot. Status code: {response.status_code}"
        )

    file_name = "screenshot.png"
    with open(file_name, "wb") as file:
        file.write(response.content)

    with open(file_name, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze the image and check if there is a section in the image in the form of a small table 'Прямые рейсы' 
                        at the very top of the image? Output the response as a JSON file with the field 'answer' and the values 'true' or 'false'!
                        Also determine very carefully the exact location of the table by pixels and add to the output how many pixels you need to crop the image 
                        on each side to leave only the "Direct flights" table! Set this information in the fields 'left', 'top', 'right', 'bottom'.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    content = response.choices[0].message.content

    analysis = content.strip()
    if analysis.startswith("```json"):
        analysis = analysis[7:]  # Remove ```json
    if analysis.endswith("```"):
        analysis = analysis[:-3]  # Remove trailing ```
    analysis = analysis.strip()
    direct_flights = json.loads(analysis)

    if direct_flights["answer"]:
        image = Image.open(file_name)
        left = 50
        top = 250
        right = image.width - 200
        bottom = image.height - 3400

        cropped_image = image.crop((left, top, right, bottom))

        cropped_image.save(file_name)

        result = True
    else:
        result = False

    return result
