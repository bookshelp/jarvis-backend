import os
import re
import httpx
import fitz  # PyMuPDF
import tempfile
import validators
import json
import urllib.parse
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from bs4 import BeautifulSoup
from typing import List, Dict
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    university: str
    semester: str
    subject: str

class PDFRequest(BaseModel):
    pdf_url: HttpUrl

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set")

# Global HTTP client for reuse
HTTP_CLIENT = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

async def validate_url(url: str) -> bool:
    if not validators.url(url):
        return False
    if re.match(r'^https?://(localhost|127\.|192\.168|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))', url):
        return False
    return True

async def extract_text_from_pdf(pdf_url: str, max_pages: int = 20) -> str:
    try:
        response = await HTTP_CLIENT.get(pdf_url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        text = ""
        for page_num in range(min(max_pages, doc.page_count)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        os.remove(tmp_path)
        
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

async def evaluate_academic_quality(text: str) -> Dict:
    if not text or len(text) < 100:
        return {"score": 0, "summary": "Document too short", "quality": "invalid"}

    prompt = f"Evaluate this academic text. Return ONLY JSON with: 'score' (0-100), 'summary' (1 sentence), and 'quality' (poor/average/good/excellent).\n\nText: {text[:2500]}"

    # FIXED: Using x-goog-api-key header instead of Bearer token
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    try:
        response = await HTTP_CLIENT.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        return json.loads(result)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"score": 0, "summary": "Failed to evaluate", "quality": "unknown"}

async def search_duckduckgo(university: str, subject: str, semester: str) -> List[str]:
    try:
        # Construct search query
        query = f"{subject} {semester} {university} filetype:pdf"
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        response = await HTTP_CLIENT.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            # DuckDuckGo obfuscates links. We must extract the real URL from the 'uddg' parameter.
            if 'uddg=' in href:
                parsed_url = urllib.parse.urlparse(href)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                actual_url = query_params.get('uddg', [''])[0]
                
                if actual_url.lower().endswith('.pdf'):
                    pdf_links.append(actual_url)
            # Sometimes direct links appear
            elif href.lower().endswith('.pdf'):
                pdf_links.append(href)

        # Return top 3 unique links
        return list(dict.fromkeys(pdf_links))[:3]

    except Exception as e:
        print(f"Search Error: {e}")
        return []

@app.post("/research")
async def research(request: ResearchRequest):
    pdf_urls = await search_duckduckgo(request.university, request.subject, request.semester)
    return {
        "status": "success",
        "query": f"{request.subject} {request.semester} {request.university}",
        "pdf_urls": pdf_urls,
        "message": "Found PDFs" if pdf_urls else "No PDFs found for this query."
    }

@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    if not await validate_url(str(request.pdf_url)):
        raise HTTPException(status_code=400, detail="Invalid or unsafe URL")

    text = await extract_text_from_pdf(str(request.pdf_url))
    evaluation = await evaluate_academic_quality(text)

    return {
        "pdf_url": str(request.pdf_url),
        "text_length": len(text),
        "evaluation": evaluation
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gemini_configured": bool()}