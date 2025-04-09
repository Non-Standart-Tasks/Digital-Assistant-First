import streamlit as st
import requests
import base64
import io
from PIL import Image, ImageDraw
import warnings
import sys
import os
import json
import subprocess
import uuid
import shutil
import tempfile
from pathlib import Path
import importlib.util
import math
from fastapi import FastAPI, Response
from pydantic import BaseModel, field_validator
from io import BytesIO

# -----------------------------------------------------------------------
# Фильтрация всех warnings
warnings.filterwarnings("ignore")

# Перенаправление stderr (для подавления выводов в консоль)
sys.stderr = open(os.devnull, "w")

# Настройка уровня логирования
os.environ["STREAMLIT_LOGGING_LEVEL"] = "ERROR"

# Устанавливаем широкую верстку
st.set_page_config(layout="wide")

# Определяем ROOT_DIR для доступа к файлам
try:
    from digital_assistant_first.utils.paths import ROOT_DIR
except ImportError:
    # Если не удается импортировать, определяем ROOT_DIR относительно текущего файла
    ROOT_DIR = Path(__file__).parent.parent.parent

API_BASE = "http://185.221.163.214:8001"  # Адрес FastAPI

# -----------------------------------------------------------------------
# Код из main.py для прямого использования без импорта

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

def round_corners(img: Image.Image, corner_radius: int = 30) -> Image.Image:
    """
    Given a PIL Image, return a new Image with rounded corners applied.
    """
    # Ensure RGBA mode so alpha can be preserved
    img = img.convert("RGBA")
    width, height = img.size
    
    # Create mask for rounded corners
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw two rectangles to cover main area
    draw.rectangle([(corner_radius, 0), (width - corner_radius, height)], fill=255)
    draw.rectangle([(0, corner_radius), (width, height - corner_radius)], fill=255)
    
    # Draw four circles for the corners
    draw.pieslice([(0, 0), (corner_radius * 2, corner_radius * 2)], 180, 270, fill=255)
    draw.pieslice([(width - corner_radius * 2, 0), (width, corner_radius * 2)], 270, 360, fill=255)
    draw.pieslice([(0, height - corner_radius * 2), (corner_radius * 2, height)], 90, 180, fill=255)
    draw.pieslice([(width - corner_radius * 2, height - corner_radius * 2), (width, height)], 0, 90, fill=255)
    
    # Apply rounded corner mask
    rounded = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    rounded.paste(img, mask=mask)
    
    return rounded

# Обновим функцию escape_latex для минимального экранирования
def escape_latex(text: str) -> str:
    """
    Минимальное экранирование только критичных символов LaTeX, сохраняя кириллицу.
    """
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # Экранируем только самые критичные спецсимволы LaTeX
    for char in ['%', '$', '#', '_', '{', '}']:
        text = text.replace(char, '\\' + char)
        
    # Заменяем обратные слэши на прямые (это особенность XeLaTeX)
    text = text.replace('\\\\', '\\')
    
    # Заменяем некоторые символы на более безопасные эквиваленты
    text = text.replace('&', '\\&')
    text = text.replace('^', '\\^{}')
    text = text.replace('~', '\\~{}')
    
    # Заменяем переносы строк на пробелы
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text

# Находим функцию compile_tex
def compile_tex(offers: list[Offer]):
    # 1. Generate a unique file ID to avoid collisions during parallel requests
    file_id = str(uuid.uuid4())

    # 2. Create a directory for this compilation to keep files together (optional but cleaner)
    workdir = f"work_{file_id}"
    os.makedirs(workdir, exist_ok=True)
    
    try:
        # 3. Decode and save each offer's image to a unique file
        #    We will collect the filenames and build the LaTeX body dynamically.
        image_paths = []
        for i, offer in enumerate(offers):
            img_filename = f"image_{i}.jpg"
            img_path = os.path.join(workdir, img_filename)

            img = Image.open(BytesIO(base64.b64decode(offer.image)))
            rounded_img = round_corners(img, corner_radius=30)
            rounded_img.save(img_path, format="PNG")

            image_paths.append(img_filename)

        # 4. Build the LaTeX header with XeLaTeX для поддержки кириллицы
        latex_header = r"""
\documentclass[a4paper]{article}

% Минимальный набор пакетов для XeLaTeX с поддержкой кириллицы
\usepackage{fontspec}
\usepackage[russian]{babel}
\usepackage{graphicx}
\usepackage[most]{tcolorbox}
\usepackage{geometry}
\usepackage{eso-pic}
\usepackage{tikz}
\usepackage{hyperref}
\usepackage{multicol}

% Базовые шрифты с поддержкой кириллицы
\setmainfont{Liberation Serif}
\setsansfont{Liberation Sans}

\geometry{top=2cm, bottom=0.7cm, left=0.7cm, right=0.7cm}

\definecolor{darkgray}{RGB}{77,76,76}
\definecolor{headingcol}{RGB}{172,150,134}

\newcommand\BackgroundPic{
    \AddToShipoutPicture*{
        \AtPageLowerLeft{
            \includegraphics[width=\paperwidth,height=\paperheight]{background_empty.pdf}
        }
    }
}

\newcommand{\offercard}[4]{%
    \begin{tcolorbox}[
        width=\linewidth,
        colframe=white,
        colback=white,
        boxrule=0.5pt,
        arc=5mm,
        left=5pt, right=5pt, top=5pt, bottom=5pt,
    ]
        \sffamily
        \includegraphics[width=\textwidth]{#1}
        
        \vspace{0.2cm}{\footnotesize\textcolor{gray}{#2}}
        \vspace{0.2cm}
        
        \textbf{#3}
        
        \vspace{0.2cm}
        \hfill\href{#4}{\textcolor{darkgray}{\footnotesize Подробнее}}
    \end{tcolorbox}
    \vspace{0.5cm}
}

\newcommand{\insertheading}[1]{
    \begin{center}
    \sffamily
    \color{headingcol}
    \Huge
    #1
    \end{center}
    \vspace{1cm}
}

\begin{document}
"""

        # 5. Build the LaTeX body with simplified approach
        latex_body = []

        # We'll create pages with two columns of offers
        chunk_size = 6  # offers per page
        num_pages = math.ceil(len(offers) / chunk_size)

        for page_idx in range(num_pages):
            # Start a new page with background + heading
            latex_body.append(r"\BackgroundPic")
            latex_body.append(r"\insertheading{Предложения}" + "\n")

            # Determine which offers are on this page
            start = page_idx * chunk_size
            end = min(start + chunk_size, len(offers))
            page_offers = offers[start:end]
            page_images = image_paths[start:end]
            
            # Begin page layout with two columns
            latex_body.append(r"\begin{minipage}[t]{0.48\textwidth}")
            
            # Left column offers (even indices)
            for i in range(0, len(page_offers), 2):
                if i < len(page_offers):
                    offer = page_offers[i]
                    img_path = page_images[i]
                    
                    # Используем обновленную функцию escape_latex, которая сохраняет кириллицу
                    category = escape_latex(offer.category)
                    description = escape_latex(offer.description)
                    
                    latex_body.append(f"\\offercard{{{img_path}}}{{{category}}}{{{description}}}{{{offer.url}}}")
            
            latex_body.append(r"\end{minipage}\hfill")
            
            # Right column offers (odd indices)
            latex_body.append(r"\begin{minipage}[t]{0.48\textwidth}")
            
            for i in range(1, len(page_offers), 2):
                if i < len(page_offers):
                    offer = page_offers[i]
                    img_path = page_images[i]
                    
                    # Используем обновленную функцию escape_latex, которая сохраняет кириллицу
                    category = escape_latex(offer.category)
                    description = escape_latex(offer.description)
                    
                    latex_body.append(f"\\offercard{{{img_path}}}{{{category}}}{{{description}}}{{{offer.url}}}")
            
            latex_body.append(r"\end{minipage}")
            
            # Add a new page if this isn't the last page
            if page_idx < num_pages - 1:
                latex_body.append(r"\newpage")

        # 6. Wrap up with the document end
        latex_footer = r"\end{document}"

        # Combine everything into a full LaTeX string
        full_latex = latex_header + "\n".join(latex_body) + latex_footer

        # Write the .tex source to file
        tex_filename = os.path.join(workdir, f"{file_id}.tex")
        with open(tex_filename, "w", encoding="utf-8") as f:
            f.write(full_latex)

        # Save the LaTeX source for debugging
        debug_tex = os.path.join(workdir, "debug.tex")
        with open(debug_tex, "w", encoding="utf-8") as f:
            f.write(full_latex)

        # Copy background file
        background_pdf = os.path.join(ROOT_DIR, "digital_assistant_first", "offergen", "background_empty.pdf")
        if not os.path.exists(background_pdf):
            print(f"Файл фона не найден: {background_pdf}")
            return {
                "status": "error",
                "message": f"Файл фона не найден: {background_pdf}"
            }
            
        # Копируем файл фона в рабочую директорию
        dest_background = os.path.join(workdir, "background_empty.pdf")
        shutil.copyfile(background_pdf, dest_background)
        
        # Обновим команду запуска xelatex с дополнительными параметрами
        # Run xelatex вместо pdflatex - принудительно используем xelatex с параметрами
        command = [
            "xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-output-driver=xdvipdfmx -V 7",  # Используем специальный драйвер с большей совместимостью
            os.path.basename(tex_filename)
        ]
        print(f"Запуск команды XeLaTeX: {command}")
        
        # Запускаем компиляцию дважды для правильного форматирования
        for run_index in range(2):
            try:
                print(f"Запуск XeLaTeX (проход {run_index+1}/2)...")
                process = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=workdir
                )
                
                print(f"xelatex выполнен с кодом: {process.returncode}")
                
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при компиляции LaTeX (проход {run_index+1}): {e}")
                print(f"STDOUT: {e.stdout.decode('utf-8', errors='ignore')[:500]}")
                print(f"STDERR: {e.stderr.decode('utf-8', errors='ignore')[:500]}")
                
                # Сохраняем полный вывод для диагностики
                error_log = os.path.join(workdir, f"xelatex_error_run{run_index+1}.log")
                with open(error_log, "w", encoding="utf-8") as f:
                    f.write(f"STDOUT:\n{e.stdout.decode('utf-8', errors='ignore')}\n\n")
                    f.write(f"STDERR:\n{e.stderr.decode('utf-8', errors='ignore')}\n")
                    
                print(f"Полный лог ошибки сохранен в: {error_log}")
                
                # Проверяем наличие .log файла от XeLaTeX
                log_file = os.path.join(workdir, f"{file_id}.log")
                if os.path.exists(log_file):
                    try:
                        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                            log_content = f.read()
                        print(f"Найден лог-файл XeLaTeX, ищем ошибки...")
                        
                        # Ищем типичные ошибки в логе
                        error_lines = []
                        for line in log_content.split("\n"):
                            if "Error" in line or "error" in line or "!" in line.strip()[:1]:
                                error_lines.append(line)
                        
                        if error_lines:
                            print("Найдены следующие ошибки в логе XeLaTeX:")
                            for line in error_lines[:10]:  # Показываем первые 10 ошибок
                                print(f"  {line}")
                    except Exception as log_error:
                        print(f"Ошибка чтения лог-файла: {log_error}")
                
                # Если это первый запуск, продолжаем и пробуем второй раз
                if run_index == 0:
                    print("Первая компиляция завершилась с ошибкой, пробуем второй проход...")
                    continue
                
                # Если это второй запуск и он завершился с ошибкой, возвращаем ошибку
                return {
                    "status": "error",
                    "message": "Failed to compile LaTeX.",
                    "latex_stdout": e.stdout.decode("utf-8", errors='ignore'),
                    "latex_stderr": e.stderr.decode("utf-8", errors='ignore')
                }

        # Read the resulting PDF
        pdf_filename = os.path.join(workdir, f"{file_id}.pdf")
        if not os.path.exists(pdf_filename):
            print(f"PDF файл не найден после компиляции: {pdf_filename}")
            
            # Показываем содержимое директории для отладки
            print(f"Файлы в {workdir}: {os.listdir(workdir)}")
            
            return {
                "status": "error",
                "message": "PDF file not found after compilation."
            }
        
        # Чтение PDF-файла
        try:
            with open(pdf_filename, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                
            if not pdf_bytes:
                print(f"PDF файл пуст: {pdf_filename}")
                return {
                    "status": "error",
                    "message": "PDF file is empty."
                }
                
            return pdf_bytes
            
        except Exception as e:
            print(f"Ошибка при чтении PDF файла: {e}")
            return {
                "status": "error",
                "message": f"Error reading PDF file: {e}"
            }
            
    except Exception as e:
        import traceback
        print(f"Общая ошибка в compile_tex: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

# -----------------------------------------------------------------------
# Функция для преобразования изображений в base64 для PDF
def convert_image_to_base64(image_path):
    """Преобразует изображение в base64 для LaTeX"""
    try:
        if not os.path.exists(image_path):
            print(f"Файл изображения не найден: {image_path}")
            return None
            
        # Открываем и преобразуем изображение
        with open(image_path, 'rb') as img_file:
            # Сначала пробуем непосредственное чтение файла
            img_bytes = img_file.read()
            img_str = base64.b64encode(img_bytes).decode()
            print(f"Успешно преобразован файл в base64: {image_path}, размер: {len(img_str)} символов")
            return img_str
    except Exception as e:
        # Если непосредственное чтение не сработало, пробуем через PIL
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')  # Конвертируем в RGB для совместимости
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"Преобразовано через PIL: {image_path}, размер: {len(img_str)} символов")
            return img_str
        except Exception as pil_error:
            print(f"Ошибка при конвертации изображения в base64: {e}")
            print(f"Ошибка PIL: {pil_error}")
            return None

# -----------------------------------------------------------------------
# Функция для создания PDF с использованием напрямую функции compile_tex
def create_pdf_with_latex(offers):
    """
    Создает PDF с использованием напрямую функции compile_tex
    """
    try:
        if not offers:
            st.error("Нет офферов для создания PDF")
            return None
            
        # Преобразуем офферы в формат, ожидаемый функцией compile_tex
        latex_offers = []
        
        for offer in offers:
            # Проверяем и очищаем данные оффера
            category = offer.get("category", "").strip()
            description = offer.get("description", "").strip()
            url = offer.get("url", "#").strip()
            
            # Ограничиваем длину полей для предотвращения ошибок LaTeX (только если очень длинные)
            if len(description) > 500:
                description = description[:497] + "..."
            if len(category) > 100:
                category = category[:97] + "..."
                
            # Получаем путь к изображению
            image_path = find_image_path(offer)
            if not image_path:
                image_path = get_placeholder_image_path()
                if not image_path:
                    st.error("Не найден файл заглушки для изображения")
                    return None
            
            # Конвертируем изображение в base64
            image_base64 = convert_image_to_base64(image_path)
            if not image_base64:
                st.warning(f"Не удалось конвертировать изображение в base64: {image_path}")
                continue
            
            # Создаем объект Offer
            latex_offer = Offer(
                category=category,
                description=description,
                url=url,
                image=image_base64
            )
            
            latex_offers.append(latex_offer)
        
        if not latex_offers:
            st.error("Все изображения офферов некорректны, нечего выводить в PDF")
            return None
            
        # Печатаем диагностическую информацию
        print(f"Подготовлено {len(latex_offers)} офферов для PDF")
        for i, offer in enumerate(latex_offers):
            print(f"Оффер {i+1}: категория='{offer.category}', описание='{offer.description[:30]}...'")
            
        # Вызываем функцию compile_tex напрямую
        response = compile_tex(latex_offers)
        
        # Проверяем тип ответа
        if isinstance(response, dict) and "status" in response and response["status"] == "error":
            # Если вернулась ошибка
            error_msg = response.get("message", "Неизвестная ошибка")
            st.error(f"Ошибка при создании PDF: {error_msg}")
            if "latex_stdout" in response:
                print(f"LaTeX STDOUT: {response['latex_stdout']}")
            if "latex_stderr" in response:
                print(f"LaTeX STDERR: {response['latex_stderr']}")
            return None
        elif response is None:
            st.error("Функция compile_tex вернула None")
            return None
        else:
            # Если вернулись байты PDF или что-то еще
            print(f"PDF успешно создан, размер: {len(response) if response else 'None'} байт")
            return response
    except Exception as e:
        st.error(f"Ошибка при создании PDF: {e}")
        import traceback
        error_details = traceback.format_exc()
        st.error(error_details)
        print(error_details)
        return None

# -----------------------------------------------------------------------
# Функция для поиска изображения и улучшенная обработка
def find_image_path(offer):
    """
    Находит путь к изображению оффера. Возвращает путь или None.
    Не создает Streamlit элементы!
    """
    if "image" not in offer:
        print(f"Отсутствует поле 'image' в оффере: {offer.get('offer_url', 'unknown')}")
        return None
        
    image_data = offer["image"]
    
    # Пропускаем, если это base64 (начинается с iVBOR)
    if isinstance(image_data, str) and (image_data.startswith("iVBOR") or len(image_data) > 1000):
        print(f"Изображение в формате base64 для оффера: {offer.get('offer_url', 'unknown')}")
        return None
    
    # Случай 1: Обработка абсолютного пути (начинается с "/")
    if isinstance(image_data, str) and image_data.startswith("/"):
        if os.path.exists(image_data):
            print(f"Найден абсолютный путь к изображению: {image_data}")
            return image_data
        else:
            print(f"Абсолютный путь существует в данных, но файл не найден: {image_data}")
            
    # Случай 2: Обработка относительного пути
    elif isinstance(image_data, str) and "/" in image_data and not image_data.startswith("iVBOR"):
        # Пробуем сформировать абсолютный путь
        full_path = ROOT_DIR / image_data.lstrip("/")
        if os.path.exists(full_path):
            print(f"Найден относительный путь к изображению: {full_path}")
            return str(full_path)
        else:
            print(f"Относительный путь существует в данных, но файл не найден: {full_path}")
            
    # Пробуем поискать по ID оффера в папке изображений
    if "offer_url" in offer:
        try:
            # Извлекаем ID из URL
            offer_id = offer["offer_url"].split("/")[-1]
            if offer_id.isdigit():
                # Форматируем ID с ведущими нулями
                img_filename = f"{int(offer_id):09d}.jpg"  # Формат 000000123.jpg
                img_path = f"/root/For_Prod/pre-master/content/json/offers/images/{img_filename}"
                
                if os.path.exists(img_path):
                    print(f"Найдено изображение по ID оффера: {img_path}")
                    return img_path
                else:
                    print(f"Попытка найти изображение по ID оффера не удалась: {img_path}")
        except Exception as e:
            print(f"Ошибка при поиске изображения по ID оффера: {e}")
    
    # Если не нашли изображение, возвращаем None
    print(f"Изображение не найдено для оффера: {offer.get('offer_url', 'unknown')}")
    return None

# -----------------------------------------------------------------------
# Функция для получения базового изображения-заглушки
def get_placeholder_image_path():
    """Возвращает путь к изображению-заглушке или None."""
    placeholder_paths = [
        ROOT_DIR / "digital_assistant_first" / "offergen" / "background_empty.png",
        ROOT_DIR / "digital_assistant_first" / "offergen" / "background_empty.pdf"
    ]
    
    for path in placeholder_paths:
        if os.path.exists(path):
            return str(path)
            
    print(f"Путь к ROOT_DIR: {ROOT_DIR}")
    print(f"Содержимое директории offergen: {os.listdir(ROOT_DIR / 'digital_assistant_first' / 'offergen')}")
    return None

# -----------------------------------------------------------------------
# Получение офферов по user_id из FastAPI
@st.cache_data
def fetch_offers_by_user_id(u_id: str):
    """Возвращает список офферов в JSON-формате по user_id."""
    response = requests.get(f"{API_BASE}/get_offers/{u_id}")
    if response.status_code == 200:
        return response.json()
    return []

# -----------------------------------------------------------------------
def main():
    user_id = st.query_params.get("user_id")

    if not user_id:
        st.warning("Не передан user_id в URL. Пример: ?user_id=abc-123")
        return

    # Получаем офферы (список словарей)
    offers = fetch_offers_by_user_id(user_id)

    # Инициализируем состояние (при первом заходе)
    if "offers" not in st.session_state:
        st.session_state.offers = offers.copy()
        st.session_state.edit_mode = [False] * len(offers)
        st.session_state.show_controls = True

    # Кнопка включения/выключения режима редактирования
    if st.button("🔧 Показать/Скрыть режим редактирования"):
        st.session_state.show_controls = not st.session_state.show_controls
        st.rerun()

    # Заголовок
    st.title("Предложения" + (" (режим редактирования)" if st.session_state.show_controls else ""))

    # Две колонки для карточек
    cols = st.columns(2)
    remove_indices = []

    # Рендер карточек
    for i, offer in enumerate(st.session_state.offers):
        col = cols[i % 2]
        with col:
            with st.container():
                # Находим путь к изображению
                image_path = find_image_path(offer)
                
                # Если путь найден, отображаем изображение
                if image_path:
                    try:
                        st.image(image_path, width=1200)
                    except Exception as e:
                        st.error(f"Ошибка при загрузке изображения: {e}")
                        # Показываем заглушку
                        placeholder = get_placeholder_image_path()
                        if placeholder:
                            st.image(placeholder, width=1200)
                elif "image" in offer and isinstance(offer["image"], str):
                    # Если путь не найден, но есть содержимое в offer["image"],
                    # проверяем, может быть это base64
                    try:
                        if offer["image"].startswith("iVBOR") or len(offer["image"]) > 1000:
                            img_data = base64.b64decode(offer["image"])
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, width=1200)
                        else:
                            # Не найдено изображение, используем заглушку
                            placeholder = get_placeholder_image_path()
                            if placeholder:
                                st.image(placeholder, width=1200)
                    except Exception as e:
                        st.error(f"Ошибка при декодировании изображения: {e}")
                        # Используем заглушку
                        placeholder = get_placeholder_image_path()
                        if placeholder:
                            st.image(placeholder, width=1200)
                else:
                    # Если нет никакого изображения, используем заглушку
                    placeholder = get_placeholder_image_path()
                    if placeholder:
                        st.image(placeholder, width=1200)

                # Если включен режим редактирования
                if st.session_state.show_controls:
                    # Создаем уникальный ключ для file_uploader
                    unique_key = f"uploader_{i}_{hash(str(offer.get('offer_url', '')))}"
                    
                    # Инициализируем ключи для отслеживания состояния загрузки
                    if f"uploaded_{unique_key}" not in st.session_state:
                        st.session_state[f"uploaded_{unique_key}"] = False
                    
                    uploaded_file = st.file_uploader(
                        label="Перетащите новое изображение для замены",
                        type=["png", "jpg", "jpeg"],
                        key=unique_key
                    )
                    
                    # Проверяем, загружен ли файл и еще не обработан
                    if uploaded_file is not None and not st.session_state[f"uploaded_{unique_key}"]:
                        try:
                            # Помечаем, что файл обрабатывается
                            st.session_state[f"uploaded_{unique_key}"] = True
                            
                            # Получаем ID оффера для формирования имени файла
                            offer_id = offer.get("offer_url", "").split("/")[-1]
                            
                            # Проверяем, является ли ID числовым
                            if offer_id and offer_id.isdigit():
                                # Используем ID из offer_url для имени файла
                                img_filename = f"{int(offer_id):09d}.jpg"  # Формат 000000123.jpg
                            else:
                                # Если ID не числовой или отсутствует, генерируем уникальный ID только для имени файла
                                # НЕ изменяем offer_url
                                random_id = str(uuid.uuid4().int)[:9]  # Берем первые 9 цифр UUID
                                img_filename = f"{int(random_id):09d}.jpg"  # Формат 000000123.jpg
                            
                            # Загружаем изображение
                            new_img = Image.open(uploaded_file)
                            
                            # Преобразуем изображение в RGB перед сохранением в JPEG формате
                            if new_img.mode == 'RGBA':
                                new_img = new_img.convert('RGB')
                            
                            # Формируем путь для сохранения
                            img_path = f"/root/For_Prod/pre-master/content/json/offers/images/{img_filename}"
                            
                            # Сохраняем изображение
                            new_img.save(img_path, "JPEG")
                            st.success(f"Изображение сохранено как {img_filename}")
                            
                            # Обновляем путь в данных оффера
                            st.session_state.offers[i]["image"] = img_path
                            
                            # Обновляем страницу только один раз после сохранения
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ошибка при загрузке изображения: {e}")
                            # Сбрасываем флаг загрузки при ошибке
                            st.session_state[f"uploaded_{unique_key}"] = False
                    # Если файл был сброшен пользователем, сбрасываем флаг
                    elif uploaded_file is None and st.session_state[f"uploaded_{unique_key}"]:
                        st.session_state[f"uploaded_{unique_key}"] = False

                # Категория
                st.caption(offer.get("category", ""))

                # Кнопки перемещения карточки (вверх/вниз)
                if st.session_state.show_controls:
                    col_move1, col_move2 = st.columns(2)
                    with col_move1:
                        if i > 0 and st.button("<-", key=f"move_up_{i}"):
                            (st.session_state.offers[i],
                             st.session_state.offers[i - 1]) = (st.session_state.offers[i - 1],
                                                               st.session_state.offers[i])
                            (st.session_state.edit_mode[i],
                             st.session_state.edit_mode[i - 1]) = (st.session_state.edit_mode[i - 1],
                                                                   st.session_state.edit_mode[i])
                            st.rerun()
                    with col_move2:
                        if i < len(st.session_state.offers) - 1 and st.button("->", key=f"move_down_{i}"):
                            (st.session_state.offers[i],
                             st.session_state.offers[i + 1]) = (st.session_state.offers[i + 1],
                                                               st.session_state.offers[i])
                            (st.session_state.edit_mode[i],
                             st.session_state.edit_mode[i + 1]) = (st.session_state.edit_mode[i + 1],
                                                                   st.session_state.edit_mode[i])
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
                    # Показываем описание
                    st.markdown(offer.get("description", ""), unsafe_allow_html=True)
                    # Кнопка "Редактировать"
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

    # Удаляем карточки, которые были помечены
    for index in sorted(remove_indices, reverse=True):
        del st.session_state.offers[index]
        del st.session_state.edit_mode[index]
        st.rerun()

    # Разделительная черта
    st.write("---")

    # Заголовок для блока PDF
    st.subheader("Сформировать PDF")

    # Копируем background_empty.pdf если нужно
    background_pdf = ROOT_DIR / "digital_assistant_first" / "offergen" / "background_empty.pdf"
    if not os.path.exists(background_pdf):
        st.warning("Файл background_empty.pdf не найден. PDF может быть создан некорректно.")
        
    # Генерируем PDF
    if st.button("Создать и скачать PDF"):
        with st.spinner("Создание PDF..."):
            # Проверяем наличие background_empty.pdf
            background_pdf = ROOT_DIR / "digital_assistant_first" / "offergen" / "background_empty.pdf"
            if not os.path.exists(background_pdf):
                st.warning(f"Файл background_empty.pdf не найден. Путь: {background_pdf}")
                st.warning(f"Содержимое директории: {os.listdir(ROOT_DIR / 'digital_assistant_first' / 'offergen')}")
                
            pdf_data = create_pdf_with_latex(st.session_state.offers)
            
            if pdf_data:
                st.success("PDF создан успешно!")
                st.download_button(
                    label="⬇️ Скачать PDF",
                    data=pdf_data,
                    file_name="vtb_family_offers.pdf",
                    mime="application/pdf",
                )
            else:
                st.error("Не удалось создать PDF. Проверьте логи для получения дополнительной информации.")

# Запуск приложения
if __name__ == "__main__":
    main()