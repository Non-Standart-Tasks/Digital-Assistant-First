import requests
import json
from pathlib import Path
from pathlib import Path
import base64
import json
from io import BytesIO
from PIL import Image, ImageDraw
from pydantic import BaseModel, field_validator

class Offer(BaseModel):
    category: str
    description: str
    url: str
    image: str  # just a raw string

    @field_validator("image")
    def check_base64(cls, v):
        try:
            # Attempt to decode
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 string.")
        return v


def tobase64(path: Path) -> str:
    with open(path, "rb") as f:
        file_data = f.read()
    return base64.b64encode(file_data).decode("utf-8")


def test_compile():
    # Sample offers data
    offers = [
        Offer(
            category="Новинки",
            description="Описание первого предложения с интересными деталями",
            url="https://example.com/offer1",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
        Offer(
            category="Акции",
            description="Описание второго предложения со специальными условиями",
            url="https://example.com/offer2",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
        Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
                Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
                        Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
                                Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
                                        Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
                                                Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump(),
                                                        Offer(
            category="Специальные предложения",
            description="Описание третьего предложения с выгодными условиями",
            url="https://example.com/offer3",
            image=tobase64(Path("/home/anton/repos/offer-service-parse/input.jpg"))
        ).model_dump()
    ]

    # Send POST request to the compile endpoint
    response = requests.post(
        "http://localhost:8000/compile",
        json=offers,
        headers={"Content-Type": "application/json"}
    )

    # Check if request was successful
    if response.status_code == 200:
        # Save the PDF
        with open("output.pdf", "wb") as f:
            f.write(response.content)
        print("PDF successfully generated and saved as 'output.pdf'")
    else:
        print(f"Error: {response.status_code}")
        # print(f"Response: {response.text}")

if __name__ == "__main__":
    test_compile()