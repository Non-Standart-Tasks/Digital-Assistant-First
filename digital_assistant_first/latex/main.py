import subprocess
import os
import uuid
import math
import base64
from fastapi import FastAPI, Response
from pydantic import BaseModel, field_validator
from PIL import Image, ImageDraw
from io import BytesIO
import shutil

app = FastAPI()

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

@app.post("/compile")
def compile_tex(offers: list[Offer]):
    # 1. Generate a unique file ID to avoid collisions during parallel requests
    file_id = str(uuid.uuid4())

    # 2. Create a directory for this compilation to keep files together (optional but cleaner)
    workdir = f"work_{file_id}"
    os.makedirs(workdir, exist_ok=True)

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

    # Helper function to generate a single \CustomCard
    def custom_card(img_file, category, description, url):
        return rf"""\CustomCard{{{img_file}}}{{{category}}}{{{description}}}{{{url}}}"""

    # 4. Build the LaTeX header (exactly as in your snippet)
    latex_header = r"""
\documentclass[a4paper]{article}

\usepackage[T1,T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[polish,russian]{babel}
\usepackage[defaultsans]{opensans}
\usepackage{graphicx}
\usepackage[most]{tcolorbox}
\usepackage{geometry}
\usepackage{eso-pic}  % For adding the PDF background
\usepackage{tikz}
\usepackage{hyperref}

\pagenumbering{gobble}

\DeclareRobustCommand\ebseries{\fontseries{eb}\selectfont}
\DeclareRobustCommand\sbseries{\fontseries{sb}\selectfont}
\DeclareRobustCommand\ltseries{\fontseries{l}\selectfont}
\DeclareRobustCommand\clseries{\fontseries{cl}\selectfont}

\DeclareTextFontCommand{\texteb}{\ebseries}
\DeclareTextFontCommand{\textsb}{\sbseries}
\DeclareTextFontCommand{\textlt}{\ltseries}
\DeclareTextFontCommand{\textcl}{\clseries}

\geometry{top=2cm, bottom=0.7cm, left=0.7cm, right=0.7cm}

\definecolor{lightgray}{RGB}{222,217,211}
\definecolor{border}{RGB}{241,241,240}
\definecolor{darkgray}{RGB}{77,76,76}
\definecolor{headingcol}{RGB}{172,150,134}

\newcommand\BackgroundPic{
    \AddToShipoutPicture*{
        \AtPageLowerLeft{
            \includegraphics[width=\paperwidth,height=\paperheight]{background_empty.pdf}
        }
    }
}

\newcommand{\CustomCard}[4]{%
    \begin{tcolorbox}[
        enhanced,
        fuzzy shadow = {0mm}{0pt}{-1pt}{-1pt}{black!60!white},
        width=\linewidth,
        height=0.225\paperheight,
        colframe=border!90,
        colback=white,
        boxrule=0pt,
        arc=4mm,
        left=.5em, right=.5em, top=.1em, bottom=.1em,
    ]
        \sffamily
        \hspace{-.4em}\includegraphics[width=1.033\textwidth]{#1}
        
        {\vspace{0.1cm}\color{gray} \footnotesize {#2}}
        \vspace{0.2cm}

        \parbox[c][1.3\baselineskip][t]{\linewidth}{%
          \sbseries \color{darkgray} #3
        }
        
        \begin{flushright}
            \begin{tcolorbox}[
                colframe=darkgray,
                colback=darkgray,
                coltext=white,
                arc=1mm,
                width=2cm,
                height=0.55cm,
                left=0pt,
                right=0pt,
                top=0pt,
                bottom=0pt,
                halign=center
            ]
                \footnotesize \href{#4}{Подробнее}
            \end{tcolorbox}
        \end{flushright}
    \end{tcolorbox}
    \vspace{0.9cm}
}

\newcommand{\insertheading}[1]{
    \begin{center}
    \sffamily
    \ltseries
    \color{headingcol}
    \Huge
    #1
    \end{center}
    \vspace{1.5cm}
}

\begin{document}
"""

    # 5. Build the LaTeX body:
    #
    #    We will place Offers in pages of two columns.
    #    For each page: 3 offers in the left minipage, 3 in the right, 
    #    then move on to the next page, etc.
    #    Adjust chunking logic as needed.

    latex_body = []

    # We chunk the offers into groups of 6 per "page"
    chunk_size = 6
    num_pages = math.ceil(len(offers) / chunk_size)

    for page_idx in range(num_pages):
        # Start a new page with background + heading
        latex_body.append(r"\BackgroundPic")
        latex_body.append(r"\insertheading{Предложения}" + "\n")

        # Current page's chunk
        start = page_idx * chunk_size
        end = start + chunk_size
        page_offers = offers[start:end]
        page_images = image_paths[start:end]

        # We'll store the left column cards and right column cards in separate lists
        left_cards = []
        right_cards = []

        # For each offer in this chunk, place it in either left or right by index
        for i, (offer, img_path) in enumerate(zip(page_offers, page_images)):
            card_latex = custom_card(img_path, offer.category, offer.description, offer.url)
            if i % 2 == 0:
                # even index -> left column
                left_cards.append(card_latex)
            else:
                # odd index -> right column
                right_cards.append(card_latex)

        # Now put them into minipages
        latex_body.append(r"\begin{minipage}[t]{0.465\textwidth}")
        latex_body.append("\n".join(left_cards))
        latex_body.append(r"\end{minipage}%")

        latex_body.append(r"\hspace{0.43cm}")

        latex_body.append(r"\begin{minipage}[t]{0.465\textwidth}")
        latex_body.append("\n".join(right_cards))
        latex_body.append(r"\end{minipage}")

        if page_idx < num_pages - 1:
            latex_body.append(r"\newpage")


    # 6. Wrap up with the document end
    latex_footer = r"\end{document}"

    # Combine everything into a full LaTeX string
    full_latex = latex_header + "\n".join(latex_body) + latex_footer

    # 7. Write the .tex source to file
    tex_filename = os.path.join(workdir, f"{file_id}.tex")
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(full_latex)

    shutil.copyfile("background_empty.pdf", os.path.join(workdir, "background_empty.pdf"))
    # 8. Run pdflatex
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        os.path.basename(tex_filename)
    ]
    print(command)
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir  # Run in the working directory
        )
    except subprocess.CalledProcessError as e:
        # If LaTeX compilation fails, clean up and return an error
        return {
            "status": "error",
            "message": "Failed to compile LaTeX.",
            "latex_stdout": e.stdout.decode("utf-8"),
            "latex_stderr": e.stderr.decode("utf-8")
        }

    # 9. Read the resulting PDF
    pdf_filename = os.path.join(workdir, f"{file_id}.pdf")
    if not os.path.exists(pdf_filename):
        return {
            "status": "error",
            "message": "PDF file not found after compilation."
        }
    
    # crop_cmd = ["pdfcrop", "--margins", "0 0 0 0", os.path.basename(pdf_filename), os.path.basename(pdf_filename)]
    # subprocess.run(crop_cmd, check=True, cwd=workdir)
    with open(pdf_filename, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # 11. Return the PDF as a streaming response
    return Response(content=pdf_bytes, media_type="application/pdf")

