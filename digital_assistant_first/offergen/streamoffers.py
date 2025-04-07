import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw
import streamlit.components.v1 as components
import warnings
import sys
import os

# Фильтрация всех warnings
warnings.filterwarnings("ignore")

# Перенаправление stderr (для подавления выводов в консоль)
sys.stderr = open(os.devnull, "w")

# Настройка уровня логирования
os.environ["STREAMLIT_LOGGING_LEVEL"] = "ERROR"
# Устанавливаем широкую верстку
st.set_page_config(layout="wide")

API_BASE = "http://185.221.163.214:8001"  # Адрес FastAPI

# -----------------------------------------------------------------------
# Закругление углов (опционально)
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
# Получаем офферы из FastAPI
@st.cache_data
def fetch_offers_by_user_id(u_id: str):
    """Получение офферов по user_id."""
    response = requests.get(f"{API_BASE}/get_offers/{u_id}")
    if response.status_code == 200:
        return response.json()
    return []

# -----------------------------------------------------------------------
# Чтение user_id из query params
query_params = st.experimental_get_query_params()
user_id = query_params.get("user_id", [None])[0]

if not user_id:
    st.warning("Не передан user_id в URL. Пример: ?user_id=abc-123")
else:
    offers = fetch_offers_by_user_id(user_id)

    # Инициализация состояния
    if "offers" not in st.session_state:
        st.session_state.offers = offers.copy()
        st.session_state.edit_mode = [False] * len(offers)

    if "show_controls" not in st.session_state:
        st.session_state.show_controls = True

    # Кнопка включения/выключения режима редактирования
    if st.button("🔧 Показать/Скрыть режим редактирования"):
        st.session_state.show_controls = not st.session_state.show_controls

    st.title("Предложения" + (" (режим редактирования)" if st.session_state.show_controls else ""))

    # Две колонки
    cols = st.columns(2)
    remove_indices = []

    # Рендер карточек
    for i, offer in enumerate(st.session_state.offers):
        col = cols[i % 2]
        with col:
            with st.container():
                # Картинка
                try:
                    img_data = base64.b64decode(offer["image"])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка загрузки изображения: {e}")

                # Загрузка нового изображения
                if st.session_state.show_controls:
                    uploaded_file = st.file_uploader(
                        label="Перетащите новое изображение для замены",
                        type=["png", "jpg", "jpeg"],
                        key=f"file_uploader_{i}"
                    )

                    if uploaded_file is not None:
                        try:
                            new_img = Image.open(uploaded_file)
                            buffered = io.BytesIO()
                            new_img.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                            st.session_state.offers[i]["image"] = img_base64
                            st.success("Изображение обновлено!")
                        except Exception as e:
                            st.error(f"Не удалось обработать изображение: {e}")

                # Категория
                st.caption(offer.get("category", ""))

                # Кнопки перемещения
                if st.session_state.show_controls:
                    col_move1, col_move2 = st.columns([1, 1])
                    with col_move1:
                        if i > 0 and st.button("<-", key=f"move_up_{i}"):
                            st.session_state.offers[i], st.session_state.offers[i - 1] = (
                                st.session_state.offers[i - 1],
                                st.session_state.offers[i],
                            )
                            st.session_state.edit_mode[i], st.session_state.edit_mode[i - 1] = (
                                st.session_state.edit_mode[i - 1],
                                st.session_state.edit_mode[i],
                            )
                            st.rerun()

                    with col_move2:
                        if i < len(st.session_state.offers) - 1 and st.button("->", key=f"move_down_{i}"):
                            st.session_state.offers[i], st.session_state.offers[i + 1] = (
                                st.session_state.offers[i + 1],
                                st.session_state.offers[i],
                            )
                            st.session_state.edit_mode[i], st.session_state.edit_mode[i + 1] = (
                                st.session_state.edit_mode[i + 1],
                                st.session_state.edit_mode[i],
                            )
                            st.rerun()

                # Редактирование описания
                if st.session_state.edit_mode[i]:
                    edited_description = st.text_area(
                        f"Описание оффера #{i + 1}",
                        value=offer.get("description", ""),
                        height=150,
                        key=f"description_{i}"
                    )
                    if st.button("Сохранить", key=f"save_{i}"):
                        st.session_state.offers[i]["description"] = edited_description
                        st.session_state.edit_mode[i] = False
                        st.success("Описание обновлено")
                        st.rerun()
                else:
                    st.markdown(offer.get("description", ""), unsafe_allow_html=True)
                    if st.session_state.show_controls and st.button("Редактировать", key=f"edit_{i}"):
                        st.session_state.edit_mode[i] = True
                        st.rerun()

                # Ссылка
                url = offer.get("url", "#")
                if url:
                    st.markdown(f"[Подробнее]({url})", unsafe_allow_html=True)

                # Кнопка удаления
                if st.session_state.show_controls and st.button("Удалить", key=f"del_{i}"):
                    remove_indices.append(i)

    # Удаление карточек
    for index in sorted(remove_indices, reverse=True):
        del st.session_state.offers[index]
        del st.session_state.edit_mode[index]
        st.rerun()

    # ---------------------------------------------------------
    # JS-код для экспорта страницы в PDF без кнопок
    hide_js = """
    <script>
    function hideElementsBeforePrint() {
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            if (btn.innerText.includes("режим редактирования") || btn.innerText.includes("Скачать в PDF")) {
                btn.style.display = 'none';
            }
        });
    }
    window.hideElementsBeforePrint = hideElementsBeforePrint;

    function exportToPDF() {
        hideElementsBeforePrint();
        window.print();
    }
    </script>
    """

    st.markdown(hide_js, unsafe_allow_html=True)

    # Кнопка PDF (сама исчезает при печати)
    components.html(
        """
        <style>
        @media print {
            html, body {
                zoom: 70%;
                margin: 0 !important;
                padding: 0 !important;
            }
            body {
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }

            #pdf-download-button {
                display: none !important;
            }
        }
        </style>

        <div style="text-align: right; margin-top: 2em;">
            <button id="pdf-download-button" onclick="exportToPDF()" style="padding: 0.5em 1em; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer;">
                📄 Скачать в PDF
            </button>
        </div>

        <script>
        function exportToPDF() {
            // Скрыть саму кнопку перед печатью
            const thisButton = document.getElementById("pdf-download-button");
            if (thisButton) {
                thisButton.style.display = "none";
            }

            // Скрыть другие кнопки вне iframe (если есть)
            const parentButtons = parent.document.querySelectorAll('button');
            parentButtons.forEach(btn => {
                if (btn.innerText.includes("режим редактирования") || btn.innerText.includes("Скачать в PDF")) {
                    btn.style.display = 'none';
                }
            });

            // Пауза, чтобы DOM успел обновиться, и запуск печати
            setTimeout(() => {
                parent.window.print();
            }, 300);
        }
        </script>
        """,
        height=150,
        scrolling=False
    )