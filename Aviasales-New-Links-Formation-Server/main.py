from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
import httpx

app = FastAPI()

@app.get("/")
async def get_link(search_id: str, url_num: int):
    url = f"https://api.travelpayouts.com/v1/flight_searches/{search_id}/clicks/{url_num}.json?marker=621748"
    print(url)
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch redirect URL")
    data = response.json()
    redirect_url = data.get("url")
    if not redirect_url:
        raise HTTPException(status_code=500, detail="URL not found in response")
    return RedirectResponse(url=redirect_url)

# Запускать с Uvicorn на порту 19777
# uvicorn script_name:app --host 0.0.0.0 --port 19777