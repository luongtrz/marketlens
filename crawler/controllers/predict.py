from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm.llm_connector import get_llm_client
from prompt_templates.predict_trend import generate_trend_prediction_prompt
import json

router = APIRouter()

class PredictTrendRequest(BaseModel):
    coin_pair: str
    market_data: str
    summarized_news: str

@router.post("/predict-trend")
async def predict_trend(request: PredictTrendRequest):
    try:
        client = get_llm_client("deepseek")
        prompt = generate_trend_prediction_prompt(
            request.coin_pair, request.market_data, request.summarized_news
        )
        response_text = client.generate_content(prompt)
        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to generate content from LLM")
        cleaned_response = response_text.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        try:
            result_json = json.loads(cleaned_response)
            return result_json
        except json.JSONDecodeError:
            return {"raw_response": response_text, "error": "Failed to parse JSON response"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
