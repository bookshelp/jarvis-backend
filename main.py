import os
import re
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import fitz  # PyMuPDF
import tempfile
import validators

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
    print("WARNING: GEMINI_API_KEY is not set.")

async def extract_text_from_pdf(pdf_url: str) -> str:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(pdf_url, timeout=30.0)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            text = ""
            # Read up to 20 pages to avoid memory overflow
            for page_num in range(min(20, doc.page_count)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            
            os.remove(tmp_path) # Clean up
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

async def evaluate_academic_quality(text: str) -> dict:
    if not text or len(text) < 100:
        return {"score": 0, "summary": "Document too short", "quality": "invalid"}

    # Limit text to ~3000 chars to save tokens and memory
    prompt = f"Evaluate this academic text. Return ONLY a JSON object with 'score' (0-100), 'summary' (1 sentence), and 'quality' (poor/average/good/excellent).\n\nText: {text[:3000]}"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            reply_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
            
            import json
            return json.loads(reply_text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"score": 0, "summary": "Evaluation failed", "quality": "unknown"}

@app.post("/research")
async def research(request: ResearchRequest):
    # Since Playwright crashes 512MB RAM, we return a simulated search for now.
    # Phase 3 will use a proper Google Search API.
    query = f"{request.university} {request.subject} {request.semester} syllabus filetype:pdf"
    return {
        "status": "success",
        "message": "Research agent received query",
        "query": query,
        "note": "Web scraping will be implemented in Phase 3 via a lightweight API."
    }

@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    try:
        if not validators.url(str(request.pdf_url)):
            raise HTTPException(status_code=400, detail="Invalid PDF URL")

        text = await extract_text_from_pdf(str(request.pdf_url))
        evaluation = await evaluate_academic_quality(text)

        return {
            "pdf_url": str(request.pdf_url),
            "text_length": len(text),
            "evaluation": evaluation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gemini_configured": bool(GEMINI_API_KEY)}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})
