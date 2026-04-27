# Classify Service

Factor classification service using LLM (Google Gemini API).

## Description

- **Port:** 3001
- **Purpose:** Nhận raw factor string, trả về group name
- **Matching Priority:** Exact match → Cache → LLM fallback

## API Endpoints

### POST `/classify`
Classify a single factor.

```json
{
  "factor": "string"
}
```

### POST `/classify/batch`
Classify multiple factors.

```json
{
  "factors": ["string1", "string2"]
}
```

### GET `/health`
Health check endpoint.

## Setup

```bash
npm install
```

## Environment Variables

```
CLASSIFY_PORT=3001
GEMINI_API_KEY=your_api_key_here
```

## Development

```bash
npm run dev
```

## Production

```bash
npm start
```

## Dependencies

- `node-fetch` - HTTP client
- `openai` - OpenAI SDK
- `ts-node` - TypeScript execution
- `typescript` - TypeScript compiler
