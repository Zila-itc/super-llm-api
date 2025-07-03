# OpenAI-Compatible Server with gpt4free Fallback

A unified API server that provides OpenAI-compatible endpoints with multiple provider support and automatic fallback to gpt4free for free usage.

## Features

- 🔄 **Multiple Provider Support**: OpenAI, Anthropic, Google AI, and more
- 🆓 **Free Fallback**: Automatic fallback to gpt4free when paid providers fail
- 🎯 **Force Specific Provider**: Use specific provider or API key via request payload
- 📊 **Rate Limiting**: Built-in rate limiting for each provider
- 🔒 **OpenAI Compatible**: Works with any OpenAI-compatible client
- 🐳 **Docker Support**: Easy deployment with Docker
- 📝 **Comprehensive Logging**: Monitor provider usage and failures

## Quick Start

1. **Setup the server**:
   ```bash
   ./run.sh
   ```

2. **Configure your API keys** (copy `.env.template` to `.env`):
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   ```

3. **Start the server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Test the server**:
   ```bash
   python test_server.py
   ```

## Usage Examples

### Basic Usage
```python
import httpx

async def chat_example():
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/v1/chat/completions", json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello!"}]
        })
        print(response.json())
```

### Force Specific Provider
```python
payload = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "force_provider": "openai",  # Force OpenAI
    "use_fallback": False  # Don't fallback if fails
}
```

### Force Specific API Key
```python
payload = {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "force_api_key": "sk-your-specific-key-here"
}
```

### Use Free gpt4free
```python
payload = {
    "model": "g4f-gpt-4",  # Use g4f prefix
    "messages": [{"role": "user", "content": "Hello!"}]
}
```

## Configuration

### Environment Variables
```env
# Primary providers
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-google-key

# Server settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

### Advanced Configuration (config.yaml)
```yaml
providers:
  openai:
    priority: 1
    rate_limit: 60
    timeout: 30
    enabled: true
    models: ["gpt-4", "gpt-3.5-turbo"]
```

## API Endpoints

### Chat Completions
`POST /v1/chat/completions`

**Request Body:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "Hello!"}],
  "temperature": 0.7,
  "max_tokens": 150,
  "force_provider": "openai",  // Optional: force specific provider
  "force_api_key": "sk-...",   // Optional: force specific API key
  "use_fallback": true         // Optional: enable fallback (default: true)
}
```

### List Models
`GET /v1/models`

### Health Check
`GET /health`

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t openai-server .
docker run -p 8000:8000 --env-file .env openai-server
```

## Provider Priority

1. **Forced Provider/API Key** (if specified)
2. **OpenAI** (priority 1)
3. **Anthropic** (priority 2) 
4. **Google AI** (priority 3)
5. **gpt4free** (fallback)

## Rate Limiting

Each provider has configurable rate limits:
- OpenAI: 60 requests/minute
- Anthropic: 50 requests/minute
- Google: 60 requests/minute
- gpt4free: No limits

## Supported Models

### OpenAI
- gpt-4, gpt-4-turbo, gpt-3.5-turbo

### Anthropic
- claude-3-opus, claude-3-sonnet, claude-3-haiku

### Google AI
- gemini-pro, gemini-pro-vision

### gpt4free
- g4f-gpt-4, g4f-gpt-3.5-turbo, g4f-claude-3, g4f-gemini-pro

## Error Handling

The server automatically handles:
- Provider failures with fallback
- Rate limit exceeded
- Invalid API keys
- Network timeouts

## Development

### Adding New Providers

1. Add provider configuration to `load_providers()`
2. Implement provider-specific method (e.g., `call_newprovider()`)
3. Add to `call_provider()` method
4. Update models list

### Testing

```bash
# Run all tests
python test_server.py

# Test specific functionality
python client_example.py
```

## Troubleshooting

### Common Issues

1. **"All providers failed"**: Check API keys in `.env`
2. **Rate limit exceeded**: Wait or add more providers
3. **gpt4free not working**: Try different g4f providers in config

### Logs

Check server logs for detailed error information:
```bash
tail -f server.log
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details
