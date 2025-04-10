from fastapi.testclient import TestClient
import pytest

# main.py
import subprocess
import os
import uuid
from fastapi import FastAPI, Response
from pydantic import BaseModel, Base64Str

app = FastAPI()

class Offer(BaseModel):
    category: str
    description: str
    url: str
    image: Base64Str

@app.post("/compile")
def compile_tex(request: CompileRequest):
    # 1. Generate a unique file name to avoid collisions if multiple requests come in simultaneously
    file_id = str(uuid.uuid4())
    tex_filename = f"{file_id}.tex"
    pdf_filename = f"{file_id}.pdf"

    # 2. Write the LaTeX source to a file
    with open(tex_filename, "w", encoding="utf-8") as f:
        f.write(request.tex_content)

    # 3. Run pdflatex
    #    -interaction=nonstopmode ensures it won't stop for user input on errors
    #    -halt-on-error stops processing on the first error
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_filename
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # If LaTeX compilation fails, return a 400 or 500 error
        
        for ext in [".aux", ".log", ".out", ".tex", ".pdf"]:
            filepath = f"{file_id}{ext}"
            if os.path.exists(filepath):
                print(filepath)
                os.remove(filepath)
                print("Cleaned up auxiliary files")
        
        return {
            "status": "error",
            "message": "Failed to compile LaTeX.",
            "latex_stdout": e.stdout.decode("utf-8"),
            "latex_stderr": e.stderr.decode("utf-8")
        }

    # 4. Read the resulting PDF
    if not os.path.exists(pdf_filename):


        return {
            "status": "error",
            "message": f"PDF file not found after compilation."
        }
    
    with open(pdf_filename, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # 5. Clean up auxiliary files (optional)
    for ext in [".aux", ".log", ".out", ".tex", ".pdf"]:
        filepath = f"{file_id}{ext}"
        if os.path.exists(filepath):
            print(filepath)
            os.remove(filepath)
            print("Cleaned up auxiliary files")

    # 6. Return the PDF as a response
    return Response(content=pdf_bytes, media_type="application/pdf")

client = TestClient(app)

def test_compile_valid_latex():
    # Basic LaTeX document
    latex_content = r"""
\documentclass{article}
\begin{document}
Hello, World!
\end{document}
"""
    
    response = client.post(
        "/compile",
        json={"tex_content": latex_content}
    )
    
    os.makedirs("test_pdfs", exist_ok=True)
    with open(f"test_pdfs/test_compile_valid_latex.pdf", "wb") as f:
        f.write(response.content)
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 0  # Ensure we got some PDF content

def test_compile_invalid_latex():
    # Invalid LaTeX document with missing \end{document}
    latex_content = r"""
\documentclass{article}
\begin{document}
Hello, World!
"""
    
    response = client.post(
        "/compile",
        json={"tex_content": latex_content}
    )
    
    assert response.status_code == 200  # The endpoint returns 200 even for LaTeX errors
    assert "error" in response.json()["status"]
    assert "Failed to compile LaTeX" in response.json()["message"]