def generate_trend_prediction_prompt(coin_pair, market_data, summarized_news):
    """
    Generates a prompt for a financial analyst AI to predict trends based on
    coin pair, market data, and summarized news.

    Args:
        coin_pair (str): The cryptocurrency pair (e.g., "BTC/USDT").
        market_data (str or dict): Recent market data (price, volume, volatility, etc.).
        summarized_news (str): Summarized news relevant to the coin pair.

    Returns:
        str: The formatted prompt ready for LLM input.
    """
    
    prompt = f"""
    You are a financial analyst AI. You will receive the following inputs:
    - Coin pair (e.g., BTC/USDT)
    - Recent market data (price, volume, volatility, etc.)
    - Summarized news relevant to the coin pair

    Your tasks:
    1. Analyze the provided market data and news.
    2. Predict the short-term (next few days) and long-term (next few weeks/months) trend for the coin pair (bullish, bearish, or neutral).
    3. Provide a logical and detailed explanation for your predictions, referencing both market data and news.
    4. Clearly separate your analysis, prediction, and reasoning.

    Input Data:
    Coin Pair: {coin_pair}
    Market Data: {market_data}
    Summarized News: {summarized_news}

    Output Format:
    Analysis:
    [Detailed analysis of market data and news]

    Prediction:
    Short-term trend: [bullish/bearish/neutral]
    Long-term trend: [bullish/bearish/neutral]

    Reasoning:
    [Logical and detailed explanation for both predictions, referencing specific data points and news items]
    """
    return prompt.strip()