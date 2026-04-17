from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag_pipeline import pipeline  # Import the RAG pipeline function
import uvicorn

# Step 1: Define the FastAPI application
app = FastAPI()

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Running", "message": "Legal RAG Backend is live! Use /summarize/ for queries."}

# Step 2: Define a Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Step 3: Define the API endpoint
@app.post("/summarize/")
async def summarize(request: QueryRequest):
    try:
        # Call the RAG pipeline
        summaries = pipeline(request.query, top_k=request.top_k)
        return {"query": request.query, "summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Step 4: Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)