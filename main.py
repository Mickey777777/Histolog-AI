from fastapi import FastAPI, HTTPException
import rag_pipeline

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/histolog/ai/query")
async def query(query: str):
    if(not query.strip()):
        raise HTTPException(status_code=400, detail="empty query")
    
    answer = rag_pipeline.ask(query)

    return {"query": query, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)