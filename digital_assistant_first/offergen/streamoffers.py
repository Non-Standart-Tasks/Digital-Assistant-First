import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw

# Устанавливаем широкую верстку
st.set_page_config(layout="wide")

API_BASE = "http://185.221.163.214:8001"  # Адрес, по которому крутится FastAPI

# -----------------------------------------------------------------------
# Функция для установки фонового изображения
def set_bg_image(image_file: str):
    """
    Устанавливает фоновое изображение из локального файла image_file.
    Файл должен быть, например, PNG или JPEG.
    """
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------
# Применяем наше фоновое изображение:
set_bg_image("background_empty.png")  # поменяйте на свой файл

# -----------------------------------------------------------------------
# Функция для закругления углов (используйте при необходимости)
def round_corners(img: Image.Image, corner_radius: int = 30) -> Image.Image:
    img = img.convert("RGBA")
    width, height = img.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)

    draw.rectangle([(corner_radius, 0), (width - corner_radius, height)], fill=255)
    draw.rectangle([(0, corner_radius), (width, height - corner_radius)], fill=255)

    draw.pieslice([(0, 0), (corner_radius * 2, corner_radius * 2)], 180, 270, fill=255)
    draw.pieslice([(width - corner_radius * 2, 0), (width, corner_radius * 2)], 270, 360, fill=255)
    draw.pieslice([(0, height - corner_radius * 2), (corner_radius * 2, height)], 90, 180, fill=255)
    draw.pieslice([(width - corner_radius * 2, height - corner_radius * 2), (width, height)], 0, 90, fill=255)

    rounded = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    rounded.paste(img, mask=mask)

    return rounded

# -----------------------------------------------------------------------
# Функция для загрузки офферов по user_id из FastAPI
@st.cache_data
def fetch_offers_by_user_id(u_id: str):
    """Делаем запрос к нашему FastAPI, чтобы получить офферы."""
    response = requests.get(f"{API_BASE}/get_offers/{u_id}")
    if response.status_code == 200:
        return response.json()
    return []

# -----------------------------------------------------------------------
# Читаем user_id из query_params
query_params = st.experimental_get_query_params() 
user_id = query_params.get("user_id", [None])[0]

if not user_id:
    st.warning("Не передан user_id в URL. Пример: ?user_id=abc-123.")
else:
    # Загружаем офферы из FastAPI
    offers = fetch_offers_by_user_id(user_id)

    # Сохраняем в сессию (чтобы можно было удалять карточки)
    if "offers" not in st.session_state:
        st.session_state.offers = offers.copy()

    st.title("Предложения")

    # Две колонки для отображения карточек
    cols = st.columns(2)

    # Список индексов для удаления
    remove_indices = []

    # Рисуем каждую карточку
    for i, offer in enumerate(st.session_state.offers):
        col = cols[i % 2]  # Выбираем одну из двух колонок
        with col:
            with st.container():
                # Декодируем Base64
                try:
                    img_data = base64.b64decode(offer["image"])
                    img = Image.open(io.BytesIO(img_data))
                    # при желании можно также вызвать round_corners(img)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Неверная картинка: {e}")

                # Категория
                st.caption(offer.get("category", ""))
                # Описание
                st.markdown(f"**{offer.get('description', '')}**")
                # Ссылка
                url = offer.get('url', '#')
                if url:
                    st.markdown(f"[Подробнее]({url})", unsafe_allow_html=True)

                # Кнопка удаления
                if st.button("Удалить", key=f"del_{i}"):
                    remove_indices.append(i)

    # Удаляем все карточки, на которые нажали «Удалить»
    for index in sorted(remove_indices, reverse=True):
        del st.session_state.offers[index]
        st.rerun()
