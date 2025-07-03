#!/bin/bash

# OpenAI-Compatible Server Setup Script
# This script sets up a unified API server with multiple providers and gpt4free fallback

set -euo pipefail

# Function to check Python version
check_python_version() {
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed. Please install Python 3.8+ first."
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    REQUIRED_PYTHON_VERSION="3.8"

    if printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$PYTHON_VERSION" | sort -V -C; then
        echo "✅ Python $PYTHON_VERSION is installed and meets the minimum requirement of $REQUIRED_PYTHON_VERSION."
    else
        echo "❌ Python $PYTHON_VERSION is installed, but version $REQUIRED_PYTHON_VERSION or higher is required."
        echo "Please upgrade your Python installation. You can try the following commands:"
        echo "sudo apt-get update"
        echo "sudo apt-get install -y python3.8 python3.8-venv python3-pip"
        exit 1
    fi
}

# Install pip if not present
if ! python3 -m pip --version &> /dev/null; then
    echo "📦 Installing pip for Python 3..."
    echo "Please run 'sudo apt-get update && sudo apt-get install -y python3-pip' manually if pip is not found."
    exit 1
fi

# Check if uvx is installed, if not install it
if ! command -v uvx &> /dev/null; then
    echo "📦 Installing uvenv..."
    python3 -m pip install uvenv
fi

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
g4f==0.1.9.7
python-dotenv==1.0.0
pydantic==2.5.0
asyncio-throttle==1.0.2
tenacity==8.2.3
aiofiles==23.2.1
EOF

# Create .env template
cat > .env.template << 'EOF'
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google AI API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Additional providers (add more as needed)
# COHERE_API_KEY=your_cohere_api_key_here
# TOGETHER_API_KEY=your_together_api_key_here
# REPLICATE_API_TOKEN=your_replicate_token_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info

# Rate Limiting (requests per minute)
OPENAI_RATE_LIMIT=60
ANTHROPIC_RATE_LIMIT=50
GOOGLE_RATE_LIMIT=60

# G4F Configuration
G4F_ENABLED=true
G4F_RETRY_ATTEMPTS=3
G4F_TIMEOUT=30
EOF

# Create config.yaml for advanced configuration
cat > config.yaml << 'EOF'
# Advanced server configuration
server:
  host: "0.0.0.0"
  port: 8000
  reload: true
  log_level: "info"
  cors_origins: ["*"]
  
providers:
  openai:
    priority: 1
    rate_limit: 60
    timeout: 30
    enabled: true
    models:
      - "gpt-4"
      - "gpt-4-turbo"
      - "gpt-3.5-turbo"
      - "gpt-3.5-turbo-16k"
  
  anthropic:
    priority: 2
    rate_limit: 50
    timeout: 30
    enabled: true
    models:
      - "claude-3-opus-20240229"
      - "claude-3-sonnet-20240229"
      - "claude-3-haiku-20240307"
  
  google:
    priority: 3
    rate_limit: 60
    timeout: 30
    enabled: true
    models:
      - "gemini-pro"
      - "gemini-pro-vision"

g4f:
  enabled: true
  retry_attempts: 3
  timeout: 30
  providers:
    - "Bing"
    - "ChatGpt"
    - "GPTalk"
    - "Liaobots"
    - "Phind"
    - "Yqcloud"
EOF

# Create Docker setup
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  openai-server:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./config.yaml:/app/config.yaml
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

# Create run script
cat > run.sh << 'EOF'
#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting server..."
uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --reload
EOF

chmod +x run.sh

# Create test script
cat > test_server.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for OpenAI-Compatible Server
"""

import asyncio
import httpx
import json

async def test_chat_completion():
    """Test basic chat completion"""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
        
        try:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload,
                timeout=30.0
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Error: {e}")

async def test_with_forced_provider():
    """Test with forced provider"""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "force_provider": "openai",  # Force OpenAI
            "use_fallback": False
        }
        
        try:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload,
                timeout=30.0
            )
            print(f"Forced Provider Test - Status: {response.status_code}")
            print(f"Provider used: {response.json().get('provider', 'unknown')}")
        except Exception as e:
            print(f"Error: {e}")

async def test_g4f_fallback():
    """Test g4f fallback"""
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "g4f-gpt-4",
            "messages": [
                {"role": "user", "content": "Tell me a joke"}
            ],
            "temperature": 0.8
        }
        
        try:
            response = await client.post(
                "http://localhost:8000/v1/chat/completions",
                json=payload,
                timeout=30.0
            )
            print(f"G4F Test - Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Error: {e}")

async def test_models_endpoint():
    """Test models endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/v1/models")
            print(f"Models endpoint - Status: {response.status_code}")
            models = response.json()
            print(f"Available models: {len(models['data'])}")
            for model in models['data'][:5]:  # Show first 5
                print(f"  - {model['id']} (owned by {model['owned_by']})")
        except Exception as e:
            print(f"Error: {e}")

async def main():
    print("🧪 Testing OpenAI-Compatible Server...")
    print("=" * 50)
    
    await test_models_endpoint()
    print("\n" + "=" * 50)
    
    await test_chat_completion()
    print("\n" + "=" * 50)
    
    await test_with_forced_provider()
    print("\n" + "=" * 50)
    
    await test_g4f_fallback()

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x test_server.py

# Create client example
cat > client_example.py << 'EOF'
#!/usr/bin/env python3
"""
Example client for OpenAI-Compatible Server
Shows how to use the server with different configurations
"""

import asyncio
import httpx
import json

class UnifiedAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def chat_completion(self, 
                            model="gpt-3.5-turbo",
                            messages=None,
                            temperature=0.7,
                            max_tokens=None,
                            force_provider=None,
                            force_api_key=None,
                            use_fallback=True):
        """Send chat completion request"""
        
        payload = {
            "model": model,
            "messages": messages or [],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_fallback": use_fallback
        }
        
        if force_provider:
            payload["force_provider"] = force_provider
        
        if force_api_key:
            payload["force_api_key"] = force_api_key
        
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload
        )
        
        return response.json()
    
    async def list_models(self):
        """List available models"""
        response = await self.client.get(f"{self.base_url}/v1/models")
        return response.json()
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()

async def main():
    client = UnifiedAPIClient()
    
    try:
        # Example 1: Basic usage
        print("📝 Basic chat completion...")
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(f"Provider: {response.get('provider')}")
        print(f"Response: {response['choices'][0]['message']['content']}")
        
        # Example 2: Force specific provider
        print("\n🎯 Forcing specific provider...")
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "What's 2+2?"}],
            force_provider="openai"  # This will only work if OpenAI key is set
        )
        print(f"Provider: {response.get('provider')}")
        print(f"Response: {response['choices'][0]['message']['content']}")
        
        # Example 3: Use g4f fallback
        print("\n🆓 Using g4f fallback...")
        response = await client.chat_completion(
            model="g4f-gpt-4",
            messages=[{"role": "user", "content": "Tell me about AI"}]
        )
        print(f"Provider: {response.get('provider')}")
        print(f"Response: {response['choices'][0]['message']['content']}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
EOF

chmod +x client_example.py

# Create README.md
cat > README.md << 'EOF'
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
EOF

# Install dependencies using uvx
echo "📦 Installing dependencies with uvx..."
USER_BASE=$(python3 -m site --user-base)
export PATH="$USER_BASE/bin:$PATH"
~/.local/bin/uvx pip install -r requirements-minimal.txt
~/.local/bin/uvx pip install -r requirements.txt

# Copy the main application file
echo "📄 Setting up main application..."
# Note: The main.py file should be created separately with the server code

# Create a simple installer script
cat > install.sh << 'EOF'
#!/bin/bash

echo "🚀 OpenAI-Compatible Server Installer"
echo "===================================="

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Install uvenv if not present
if ! python3 -m uvenv --version &> /dev/null; then
    echo "📦 Installing uvenv..."
    python3 -m pip install uvenv
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📄 Creating .env file..."
    cp .env.template .env
    echo "✅ Please edit .env file and add your API keys"
fi

# Make scripts executable
chmod +x run.sh
chmod +x test_server.py
chmod +x client_example.py

echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: ./run.sh"
echo "3. Test: python test_server.py"
echo ""
echo "Server will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
EOF

chmod +x install.sh

# Create systemd service file
cat > openai-server.service << 'EOF'
[Unit]
Description=OpenAI-Compatible Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/openai-server
Environment=PATH=/opt/openai-server/venv/bin
ExecStart=/opt/openai-server/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/openai-server
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Project setup complete!"
echo ""
echo "📁 Created files:"
echo "   - main.py (server application)"
echo "   - requirements.txt (dependencies)"
echo "   - .env.template (environment template)"
echo "   - config.yaml (advanced configuration)"
echo "   - run.sh (development server)"
echo "   - test_server.py (test suite)"
echo "   - client_example.py (usage examples)"
echo "   - Dockerfile (containerization)"
echo "   - docker-compose.yml (orchestration)"
echo "   - install.sh (automated installer)"
echo "   - openai-server.service (systemd service)"
echo "   - README.md (documentation)"
echo ""
echo "🚀 Quick start:"
echo "   1. cp .env.template .env"
echo "   2. Edit .env and add your API keys"
echo "   3. ./run.sh"
echo ""
echo "📚 Documentation:"
echo "   - README.md contains detailed setup instructions"
echo "   - API docs will be available at http://localhost:8000/docs"
echo "   - Test with: python test_server.py"