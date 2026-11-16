import os
import re
import httpx
import pymupdf
import tempfile
import validators
import json
import urllib.parse
import asyncio
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from bs4 import BeautifulSoup
from typing import List, Dict

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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable not set")

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

        if not response.content.startswith(b'%PDF'):
            print(f"ERROR: Downloaded content is not a PDF.")
            raise HTTPException(status_code=400, detail="The URL did not return a valid PDF file.")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        doc = pymupdf.open(tmp_path)
        text = ""
        for page_num in range(min(max_pages, doc.page_count)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        os.remove(tmp_path)
        
        print(f"INFO: Successfully extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        print(f"PDF Extraction Error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

async def evaluate_academic_quality(text: str) -> Dict:
    if not text or len(text) < 100:
        print(f"WARNING: Text too short ({len(text)} chars). Skipping AI evaluation.")
        return {"score": 0, "summary": "Document too short or unreadable", "quality": "invalid"}

    prompt = f"Evaluate this academic text. Return a JSON object with exactly these three keys: 'score' (integer 0-100), 'summary' (a 1-sentence string), and 'quality' (a string: 'poor', 'average', 'good', or 'excellent'). Text: {text[:3000]}"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a helpful academic evaluator. You output JSON only."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = await HTTP_CLIENT.post(url, headers=headers, json=payload)
            
            if response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"Groq rate limit hit. Waiting {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            result_str = data["choices"][0]["message"]["content"]
            result_str = result_str.replace("```json", "").replace("```", "").strip()
            return json.loads(result_str)
            
        except Exception as e:
            print(f"Groq API Exception (Attempt {attempt + 1}): {e}")
            if isinstance(e, httpx.HTTPStatusError):
                print(f"Groq API Response: {e.response.text}")
            if attempt == max_retries - 1:
                return {"score": 0, "summary": "AI service is busy. Please try again later.", "quality": "unknown"}
            await asyncio.sleep(2 ** attempt)

async def search_duckduckgo(university: str, subject: str, semester: str) -> List[str]:
    try:
        query = f"{subject} {semester} {university} filetype:pdf"
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        
        response = await HTTP_CLIENT.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        pdf_links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'uddg=' in href:
                parsed_url = urllib.parse.urlparse(href)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                actual_url = query_params.get('uddg', [''])[0]
                if actual_url.lower().endswith('.pdf'):
                    pdf_links.append(actual_url)
            elif href.lower().endswith('.pdf'):
                pdf_links.append(href)

        unique_links = list(dict.fromkeys(pdf_links))[:3]
        
        if not unique_links:
            print("No PDFs found via DDG. Using fallback PDFs.")
            unique_links = ["https://arxiv.org/pdf/1706.03762.pdf"]
            
        return unique_links

    except Exception as e:
        print(f"Search Error: {e}")
        return ["https://arxiv.org/pdf/1706.03762.pdf"]

@app.post("/research")
async def research(request: ResearchRequest):
    pdf_urls = await search_duckduckgo(request.university, request.subject, request.semester)
    return {
        "status": "success",
        "query": f"{request.subject} {request.semester} {request.university}",
        "pdf_urls": pdf_urls,
        "message": "Found PDFs" if pdf_urls else "No PDFs found."
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
    key = os.getenv("GROQ_API_KEY", "")
    return {
        "status": "healthy",
        "groq_configured": bool(key),
        "key_length": len(key)
    }
