from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import rag_pipeline

app = FastAPI()

class MessageRequest(BaseModel):
    message: str
    king: str


@app.get("/")
async def root():
    return {"health": "ok"}

@app.post("/histolog/ai/query")
async def query(request: MessageRequest):
    if request.king not in rag_pipeline.KINGS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 왕입니다: {request.king}. 사용 가능: {list(rag_pipeline.KINGS.keys())}"
        )

    answer = rag_pipeline.ask(request.message, request.king)

    return {"query": request.message, "king": request.king, "answer": answer[0], "usage": answer[1]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
