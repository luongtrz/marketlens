# Analysis for "API Quota Exceeded"

## Issue
- User is getting `429 Resource Exhausted` from Gemini API.
- Stack trace points to `generateMarketForecast` called from `Dashboard.tsx`.

## Investigation Goal
- Determine how many times `generateMarketForecast` is called on page load.
- Check if it's called for every coin in a list automatically.

## Hypothesis
- The dashboard likely iterates through a list of coins (e.g., BTC, ETH, SOL) and triggers a forecast for each one immediately upon mounting.
- With `gemini-1.5-flash` or similar, the free tier has loose limits but strict minute/day quotas. If it fires 5-10 requests at once, it might hit the rate limit or the daily quota if reloaded often.

## Plan
1.  Analyze `pages/Dashboard.tsx` to see the `useEffect` hooks.
2.  Propose a solution:
    -   **Disable auto-fetch**: Only fetch when user clicks "Analyze".
    -   **Debounce/Throttle**: If it's on type.
    -   **Limit initial batch**: Only fetch the active coin.
