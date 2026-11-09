# Enterprise API Specification v2.4

## 1. Overview and Base URLs
The Enterprise AI Platform provides a high-throughput REST and streaming API.
- Production Base URL: `https://api.enterprise-gemma.io/v2`
- Staging Base URL: `https://staging-api.enterprise-gemma.io/v2`

## 2. Authentication and Rate Limits
All client requests must include a valid Bearer authentication token in the `Authorization` header:
`Authorization: Bearer <JWT_ACCESS_TOKEN>`

Key Parameters:
- Token expiration period: 3600 seconds (1 hour). Refresh tokens remain valid for 14 days.
- Rate Limit Quota: 1200 requests per minute per authenticated client ID.
- Burst Capacity: Up to 250 requests over a rolling 5-second window.
- Rate Limit Exceeded Response: HTTP 429 Too Many Requests.
- Standard Headers returned on rate limit: `Retry-After: <seconds>` and `X-RateLimit-Remaining: <integer>`.

## 3. Data Serialization and Correlation Identifiers
- Data format: Standard UTF-8 JSON payloads with `Content-Type: application/json`.
- Maximum request payload size: 15 megabytes (15,728,640 bytes).
- Mandatory Correlation Header: Every API request must supply `X-Request-Correlation-ID: <UUID-v4>`.
If this header is omitted, the API gateway automatically injects an auto-generated UUIDv4 and logs a compliance warning.
