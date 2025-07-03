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
