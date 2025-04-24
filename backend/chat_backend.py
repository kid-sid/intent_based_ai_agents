from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from ai_agent import get_response_from_ai_agent

# pydantic model setup
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

ALLOWED_MODELS = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile",
                  "gpt-4o-mini"]

app = FastAPI(title="Langgraph AI agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Allow requests from your frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")  # Redirect to interactive docs

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content="", media_type="image/x-icon")  # Return an empty favicon response

@app.post("/chat")
def chat_endpoint(request: RequestState):
    try:
        query = request.messages[-1] if request.messages else ""
        response = get_response_from_ai_agent(
            llm_id=request.model_name,
            query=query,
            allow_search=request.allow_search,
            system_prompt=request.system_prompt,
            provider=request.model_provider
        )
        return {"response": response}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)