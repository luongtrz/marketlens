import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import json
import asyncio

# Add the parent directory to sys.path to allow imports from crawler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.analysis import analyze_data, AnalysisRequest

class TestAnalysisController(unittest.IsolatedAsyncioTestCase):

    @patch('controllers.analysis.get_llm_client')
    async def test_analyze_data_success(self, mock_get_llm_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_llm_client.return_value = mock_client
        
        expected_json = {
            "date_time": "2023-10-27",
            "title": "Test News",
            "news": "This is a test news article.",
            "summarize_news": "Test summary",
            "market_sentiment": 0.8
        }
        
        # Mock the generate_content return value
        mock_client.generate_content.return_value = json.dumps(expected_json)
        
        # Create request
        request = AnalysisRequest(data="<html><body>Test Content</body></html>")
        
        # Call the function
        result = await analyze_data(request)
        
        # Assertions
        self.assertEqual(result, expected_json)
        mock_get_llm_client.assert_called_with("deepseek")
        mock_client.generate_content.assert_called_once()

    @patch('controllers.analysis.get_llm_client')
    async def test_analyze_data_json_markdown_stripping(self, mock_get_llm_client):
        # Setup mock
        mock_client = MagicMock()
        mock_get_llm_client.return_value = mock_client
        
        expected_json = {
            "title": "Markdown Test"
        }
        
        # Mock return value with markdown code blocks
        mock_client.generate_content.return_value = "```json\n" + json.dumps(expected_json) + "\n```"
        
        request = AnalysisRequest(data="<html>...</html>")
        result = await analyze_data(request)
        
        self.assertEqual(result, expected_json)

    @patch('controllers.analysis.get_llm_client')
    async def test_analyze_data_invalid_json(self, mock_get_llm_client):
        mock_client = MagicMock()
        mock_get_llm_client.return_value = mock_client
        
        raw_response = "Not a JSON string"
        mock_client.generate_content.return_value = raw_response
        
        request = AnalysisRequest(data="<html>...</html>")
        result = await analyze_data(request)
        
        self.assertEqual(result["error"], "Failed to parse JSON response")
        self.assertEqual(result["raw_response"], raw_response)

if __name__ == '__main__':
    unittest.main()


'''
import os
import json
import asyncio
from dotenv import load_dotenv
from controllers.analysis import analyze_data, AnalysisRequest

def test_analysis_controller():
    load_dotenv()
    
    # Path to test.html
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_dir, "test.html")
    
    if not os.path.exists(html_file_path):
        print(f"Error: {html_file_path} not found.")
        return

    with open(html_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    print(f"Read {len(html_content)} characters from test.html")

    # Create the request object
    request = AnalysisRequest(data=html_content)

    print("Calling analyze_data controller...")
    try:
        # Since analyze_data is async, we need to run it in an event loop
        result = asyncio.run(analyze_data(request))
        
        print("\n--- Analysis Result ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("-----------------------")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis_controller()

'''