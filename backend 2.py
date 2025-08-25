# step1: setup pydantic Model (Schema validation)

from pydantic import BaseModel
from typing import List

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    message: List[str]
    allow_search: bool







# step2: setup AI Agent from Frontend Request
# step3: Run app & Explore Swagger UI docs











































































"""""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_agent import get_response_from_ai_agent

# Allowed model names
ALLOWED_MODELS = {
    "gpt-4o-mini": "OpenAI",
    "llama-3.3-70b-versatile": "Groq",
    "llama3-70b-8192": "Groq",
    "mixtral-8x7b-32768": "Groq",
}

# 1. Pydantic 
class RequestState(BaseModel):
    model_name: str = Field(..., description="Choose from: " + ", ".join(ALLOWED_MODELS))
    system_prompt: str = Field(..., description="System instruction for the agent")
    query: str = Field(..., description="The user’s single-query string")
    allow_search: bool = Field(False, description="Enable Tavily search tool")

# 2. Initialize FastAPI
app = FastAPI(title="LangGraph AI Agent", version="1.0.0")

# 3. Endpoint
@app.post("/chat")
async def chat_endpoint(req: RequestState):
    if req.model_name not in ALLOWED_MODELS:
        raise HTTPException(400, detail="Invalid model_name; choose one of: " + ", ".join(ALLOWED_MODELS))
    provider = ALLOWED_MODELS[req.model_name]
    try:
        reply = get_response_from_ai_agent(
            llm_id=req.model_name,
            query=req.query,
            allow_search=req.allow_search,
            system_prompt=req.system_prompt,
            provider=provider
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    return {"reply": reply}

# 4. Run with: uvicorn backend:app --reload --host 127.0.0.1 --port 9999
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=9999, reload=True)
    

    """