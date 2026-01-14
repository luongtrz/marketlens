def get_analysis_prompt(cleaned_html_content):
    """
    Returns the prompt for analyzing cleaned HTML web data.
    """
    return f"""
You are an expert financial analyst and data extractor. Your task is to analyze the provided cleaned HTML content from a financial news website and extract structured data.

Input:
The input is cleaned HTML content where unnecessary tags like ads and scripts have been removed.

Output:
Provide the output strictly in valid JSON format with the following fields:
- "date_time": The date and time of the news event or article publication.
- "title": The headline or title of the news article.
- "news": The full text or main body of the news content.
- "summarize_news": A concise summary of the news article.
- "sentiment": One of "Positive", "Negative", or "Neutral".
- "impact_score": An integer from 0 (no impact) to 100 (high market moving potential).
- "key_takeaways": An array of 3 short strings summarizing key points.
- "detailed_summary": A detailed paragraph summary of the event.
- "short_summary": A one-sentence summary for previews.

Constraints:
- If a field cannot be found in the content, set its value to null.
- Do NOT hallucinate or invent information. Only extract what is explicitly present in the text.
- Return ONLY the JSON object, no markdown formatting or additional text.

Cleaned HTML Content:
{cleaned_html_content}
"""
