import sys
import time
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime

from controllers.analysis import router as analysis_router
from src.crawler import Crawler

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router)

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/marketlens")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client.get_database()
    news_collection = db.news
    # Create index on URL to prevent duplicates
    news_collection.create_index("url", unique=True)
except Exception as e:
    print(f"⚠️ Warning: Could not connect to MongoDB at {MONGO_URI}: {e}")
    news_collection = None

# Data Models
class NewsItem(BaseModel):
    articleDateTime: Optional[str]
    title: str
    url: Optional[str]
    news: Optional[str]
    summarizeNews: Optional[str]
    sentiment: Optional[str]
    impactScore: Optional[int]
    keyTakeaways: Optional[List[str]]
    detailedSummary: Optional[str]
    snippet: Optional[str]
    source: Optional[str]
    marketSentiment: Optional[float] = 0.0

class NewsResponse(BaseModel):
    id: str
    title: str
    source: str
    timestamp: str
    snippet: str
    url: str
    sentiment: str
    summary: str
    detailedSummary: str
    keyTakeaways: List[str]
    impactScore: int

@app.get("/api/news", response_model=List[NewsResponse])
def get_news():
    if news_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")
        
    cursor = news_collection.find().sort("articleDateTime", -1).limit(50)
    articles = []
    for doc in cursor:
        articles.append({
            "id": str(doc.get("_id")),
            "title": doc.get("title", "") or "Untitled",
            "source": doc.get("source", "Unknown"),
            "timestamp": doc.get("articleDateTime") or datetime.datetime.now().isoformat(),
            "snippet": doc.get("snippet") or doc.get("summarizeNews", "")[:200],
            "url": doc.get("url", "#"),
            "sentiment": doc.get("sentiment", "Neutral"),
            "summary": doc.get("summarizeNews", "") or "No summary available.",
            "detailedSummary": doc.get("detailedSummary", "") or doc.get("news", "")[:500],
            "keyTakeaways": doc.get("keyTakeaways") or [],
            "impactScore": doc.get("impactScore", 0)
        })
    return articles

@app.post("/api/news", status_code=201)
def create_news(item: NewsItem):
    if news_collection is None:
        raise HTTPException(status_code=503, detail="Database not available")

    # Check for duplicate URL
    if item.url and news_collection.find_one({"url": item.url}):
        # Return success slightly different to indicate existing
        return {"message": "Article already exists"}
    
    data = item.dict()
    data["createdAt"] = datetime.datetime.utcnow()
    try:
        news_collection.insert_one(data)
    except Exception as e:
         # Handle race conditions
         if "duplicate key" in str(e):
             return {"message": "Article already exists"}
         raise HTTPException(status_code=500, detail=str(e))
         
    return {"message": "News saved successfully"}

def run_crawler():
    loop = False
    interval = 300

    if "--loop" in sys.argv:
        loop = True
        idx = sys.argv.index("--loop")
        if idx + 1 < len(sys.argv):
            try:
                interval = int(sys.argv[idx + 1])
            except ValueError:
                pass

    print(f"Start crawler with Backend URL: {os.getenv('BACKEND_URL')}")
    crawler = Crawler()

    if loop:
        print(f"🚀 Crawler running in loop mode every {interval}s")

        while True:
            try:
                crawler.process_once()
            except Exception as e:
                print("❌ Crawl error:", e)

            print(f"⏳ Sleeping {interval}s...")
            time.sleep(interval)
    else:
        print("▶ Running single crawl cycle...")
        new = crawler.process_once()
        print(f"✅ Done. {new} new articles found.")


if __name__ == "__main__":
    if "--loop" in sys.argv or "--crawler" in sys.argv:
        run_crawler()
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
