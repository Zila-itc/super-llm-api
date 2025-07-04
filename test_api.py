#!/usr/bin/env python3
"""
Test script for the enhanced G4F API
"""

import asyncio
import json
import httpx
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def test_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Test chat completion endpoint"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "data": response.json()
                    }
                else:
                    return {
                        "status": "error",
                        "code": response.status_code,
                        "message": response.text
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
    
    async def test_streaming(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Test streaming chat completion"""
        payload["stream"] = True
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=60.0
                ) as response:
                    
                    if response.status_code != 200:
                        return {
                            "status": "error",
                            "code": response.status_code,
                            "message": await response.aread()
                        }
                    
                    chunks = []
                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            chunks.append(chunk)
                    
                    return {
                        "status": "success",
                        "chunks": len(chunks),
                        "sample_chunks": chunks[:3]  # First 3 chunks for inspection
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
    
    async def test_models_endpoint(self) -> Dict[str, Any]:
        """Test models listing endpoint"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/v1/models")
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "data": response.json()
                    }
                else:
                    return {
                        "status": "error",
                        "code": response.status_code,
                        "message": response.text
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }

async def main():
    """Run comprehensive API tests"""
    tester = APITester()

    # Wait for the server to be ready
    print("Waiting for server to be ready...")
    for i in range(30):
        try:
            await tester.test_models_endpoint() # A simple request to check readiness
            print("✅ Server is ready!")
            break
        except Exception as e:
            if i == 29:
                print(f"❌ Server did not become ready in time: {e}")
                return
            await asyncio.sleep(1)

    print("🚀 Testing Enhanced G4F API")
    print("=" * 50)
    
    # Test 1: Models endpoint
    print("\n📋 Testing /v1/models endpoint...")
    models_result = await tester.test_models_endpoint()
    print(f"Status: {models_result['status']}")
    if models_result['status'] == 'success':
        models = models_result['data'].get('data', [])
        print(f"Available models: {len(models)}")
        g4f_models = [m for m in models if m['id'].startswith('g4f-')]
        print(f"G4F models: {len(g4f_models)}")
    else:
        print(f"Error: {models_result.get('message', 'Unknown error')}")
    
    # Test 2: Basic G4F chat completion
    print("\n💬 Testing G4F chat completion...")
    basic_payload = {
        "model": "g4f-gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello! Can you tell me a short joke?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    basic_result = await tester.test_chat_completion(basic_payload)
    print(f"Status: {basic_result['status']}")
    if basic_result['status'] == 'success':
        response = basic_result['data']
        print(f"Model: {response.get('model')}")
        print(f"Provider: {response.get('provider')}")
        content = response['choices'][0]['message']['content']
        print(f"Response: {content[:100]}..." if len(content) > 100 else f"Response: {content}")
    else:
        print(f"Error: {basic_result.get('message', 'Unknown error')}")
    
    # Test 3: G4F with web search
    print("\n🔍 Testing G4F with web search...")
    web_search_payload = {
        "model": "g4f-gpt-4",
        "messages": [
            {"role": "user", "content": "What's the latest news about AI developments?"}
        ],
        "web_search": True,
        "temperature": 0.5,
        "max_tokens": 150
    }
    
    web_result = await tester.test_chat_completion(web_search_payload)
    print(f"Status: {web_result['status']}")
    if web_result['status'] == 'success':
        response = web_result['data']
        print(f"Model: {response.get('model')}")
        print(f"Provider: {response.get('provider')}")
        content = response['choices'][0]['message']['content']
        print(f"Response: {content[:150]}..." if len(content) > 150 else f"Response: {content}")
    else:
        print(f"Error: {web_result.get('message', 'Unknown error')}")
    
    # Test 4: Streaming response
    print("\n📡 Testing G4F streaming...")
    streaming_payload = {
        "model": "g4f-gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Write a short poem about technology."}
        ],
        "temperature": 0.8,
        "max_tokens": 100
    }
    
    stream_result = await tester.test_streaming(streaming_payload)
    print(f"Status: {stream_result['status']}")
    if stream_result['status'] == 'success':
        print(f"Received {stream_result['chunks']} chunks")
        print("Sample chunks:")
        for i, chunk in enumerate(stream_result.get('sample_chunks', [])):
            print(f"  Chunk {i+1}: {chunk[:100]}..." if len(chunk) > 100 else f"  Chunk {i+1}: {chunk}")
    else:
        print(f"Error: {stream_result.get('message', 'Unknown error')}")
    
    # Test 5: Provider priority
    print("\n⚡ Testing provider priority...")
    priority_payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello from priority test!"}
        ],
        "provider_priority": ["g4f", "openai", "anthropic"],
        "retry_attempts": 2,
        "temperature": 0.7
    }
    
    priority_result = await tester.test_chat_completion(priority_payload)
    print(f"Status: {priority_result['status']}")
    if priority_result['status'] == 'success':
        response = priority_result['data']
        print(f"Used provider: {response.get('provider')}")
        print(f"Model: {response.get('model')}")
    else:
        print(f"Error: {priority_result.get('message', 'Unknown error')}")

    # Test 6: Cohere chat completion
    print("\n🔵 Testing Cohere chat completion...")
    cohere_payload = {
        "model": "command-r-plus",
        "messages": [
            {"role": "user", "content": "Hello from Cohere!"}
        ]
    }

    cohere_result = await tester.test_chat_completion(cohere_payload)
    print(f"Status: {cohere_result['status']}")
    if cohere_result['status'] == 'success':
        response = cohere_result['data']
        print(f"Model: {response.get('model')}")
        print(f"Provider: {response.get('provider')}")
        content = response['choices'][0]['message']['content']
        print(f"Response: {content[:100]}..." if len(content) > 100 else f"Response: {content}")
    else:
        print(f"Error: {cohere_result.get('message', 'Unknown error')}")
    
    print("\n✅ Testing completed!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())