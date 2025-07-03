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
