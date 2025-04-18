from fastapi import FastAPI, Request
import uvicorn
import uuid

app = FastAPI()

# Глобальный словарь, где key = user_id, value = список офферов
DATABASE = {}

@app.post("/generate_link")
async def generate_link(request: Request):
    """
    Принимает JSON массива офферов и возвращает уникальную ссылку на страницу Streamlit.
    Формат оффера:
    [
      {
        "category": "...",
        "description": "...",
        "url": "...",
        "image": "..."  
      }, ...
    ]
    """
    offers = await request.json()

    # Генерируем уникальный user_id
    user_id = str(uuid.uuid4())

    # Сохраняем офферы в словаре
    DATABASE[user_id] = offers

    # Формируем ссылку на наше Streamlit-приложение
    # Допустим, оно будет крутиться на http://localhost:8501/
    link = f"https://offer.vtb.msut.me/?user_id={user_id}"

    return {"link": link}

@app.get("/get_offers/{user_id}")
async def get_offers(user_id: str):
    """Возвращает сохранённый JSON офферов для данного user_id"""
    data = DATABASE.get(user_id, [])
    return data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
