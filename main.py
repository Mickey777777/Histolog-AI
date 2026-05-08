from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag_pipeline

app = FastAPI()

class MessageRequest(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"health": "ok"}

@app.post("/histolog/ai/query")
async def query(request: MessageRequest):
    answer = rag_pipeline.ask(request.message)

    return {"query": query, "answer": answer[0], "usage": answer[1]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)