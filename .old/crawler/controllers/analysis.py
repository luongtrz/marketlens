from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from llm.llm_connector import get_llm_client
from prompt_templates.analysis_web_raw_data import get_analysis_prompt
import json

router = APIRouter()

class AnalysisRequest(BaseModel):
    data: str

@router.post("/analysis")
async def analyze_data(request: AnalysisRequest):
    try:
        # Get the LLM client (defaulting to Gemini for now)
        client = get_llm_client("deepseek")
        
        # Generate the prompt
        prompt = get_analysis_prompt(request.data)
        
        # Get the response from the LLM
        response_text = client.generate_content(prompt)
        
        if not response_text:
            raise HTTPException(status_code=500, detail="Failed to generate content from LLM")
            
        # Try to parse the response as JSON
        # The prompt asks for strictly JSON, but LLMs can be chatty.
        # We might need to clean the response if it contains markdown code blocks.
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
            # If parsing fails, return the raw text wrapped in a structure
            return {"raw_response": response_text, "error": "Failed to parse JSON response"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
