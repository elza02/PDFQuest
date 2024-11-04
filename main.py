# main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import Rag_with_py 
app = FastAPI()

# Mount static files for CSS and JavaScript
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: QueryRequest):
    # Process the question (e.g., call your RAG model)
    user_question = request.question
    # Generate a response for the demo
    ret = Rag_with_py.retrieving_chunks(user_question)
    response = Rag_with_py.llm_generation(user_question,ret)
    return JSONResponse({"response": response})
# Endpoint to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def get_chat():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)