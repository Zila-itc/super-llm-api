#!/usr/bin/env python3
"""
OpenAI-Compatible Server with gpt4free fallback
Provides a unified API endpoint for multiple providers with automatic fallback
"""

import asyncio
import json
import os
import random
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx
import g4f
from g4f.Provider import RetryProvider

# Configuration
class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    LOG_LEVEL: str = "info"

    # Rate Limiting (requests per minute)
    OPENAI_RATE_LIMIT: int = 60
    ANTHROPIC_RATE_LIMIT: int = 50
    GOOGLE_RATE_LIMIT: int = 60

    # G4F Configuration
    G4F_ENABLED: bool = True
    G4F_RETRY_ATTEMPTS: int = 3
    G4F_TIMEOUT: int = 30

@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    models: List[str]
    priority: int = 0
    rate_limit: int = 60  # requests per minute
    enabled: bool = True

class RequestModel(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    # Custom fields
    force_provider: Optional[str] = None
    force_api_key: Optional[str] = None
    use_fallback: Optional[bool] = True

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    provider: Optional[str] = None

class App:
    def __init__(self):
        self.config = Config()
        self.app = FastAPI(
            title="OpenAI-Compatible Server with gpt4free",
            description="Unified API endpoint with multiple providers and fallback",
            version="1.0.0"
        )
        
        # Load configuration
        self.providers = self.load_providers()
        self.rate_limits = {}

        # Setup g4f providers
        self.g4f_providers = [
            g4f.Provider.Bing,
            g4f.Provider.ChatgptX,
            g4f.Provider.GPTalk,
            g4f.Provider.Liaobots,
            g4f.Provider.Phind,
            g4f.Provider.Yqcloud,
        ]
        
        # Setup g4f providers
        self.g4f_providers = [
            g4f.Provider.Bing,
            g4f.Provider.ChatgptX,
            g4f.Provider.GPTalk,
            g4f.Provider.Liaobots,
            g4f.Provider.Phind,
            g4f.Provider.Yqcloud,
        ]
        
        # Setup routes
        self.setup_routes()
        
    def load_providers(self) -> List[ProviderConfig]:
        """Load provider configurations from environment or config file"""
        providers = []
        
        # OpenAI
        if self.config.OPENAI_API_KEY:
            providers.append(ProviderConfig(
                name="openai",
                api_key=self.config.OPENAI_API_KEY,
                base_url="https://api.openai.com/v1",
                models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                priority=1,
                rate_limit=self.config.OPENAI_RATE_LIMIT
            ))
        
        # Anthropic
        if self.config.ANTHROPIC_API_KEY:
            providers.append(ProviderConfig(
                name="anthropic",
                api_key=self.config.ANTHROPIC_API_KEY,
                base_url="https://api.anthropic.com/v1",
                models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                priority=2,
                rate_limit=self.config.ANTHROPIC_RATE_LIMIT
            ))
        
        # Add more providers as needed
        # Google AI
        if self.config.GOOGLE_API_KEY:
            providers.append(ProviderConfig(
                name="google",
                api_key=self.config.GOOGLE_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta",
                models=["gemini-pro", "gemini-pro-vision"],
                priority=3,
                rate_limit=self.config.GOOGLE_RATE_LIMIT
            ))
        
        return sorted(providers, key=lambda x: x.priority)
    
    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: RequestModel):
            return await self.handle_chat_completion(request)
        
        @self.app.get("/v1/models")
        async def list_models():
            models = []
            for provider in self.providers:
                for model in provider.models:
                    models.append({
                        "id": model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": provider.name
                    })
            
            # Add g4f models
            g4f_models = ["gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro"]
            for model in g4f_models:
                models.append({
                    "id": f"g4f-{model}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "g4f"
                })
            
            return {"object": "list", "data": models}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "providers": len(self.providers)}
    
    async def handle_chat_completion(self, request: RequestModel) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Main handler for chat completions with fallback logic"""
        
        # Check if specific provider is forced
        if request.force_provider:
            provider = next((p for p in self.providers if p.name == request.force_provider), None)
            if provider:
                try:
                    return await self.call_provider(provider, request)
                except Exception as e:
                    if not request.use_fallback:
                        raise HTTPException(status_code=500, detail=f"Provider {request.force_provider} failed: {str(e)}")
        
        # Check if specific API key is forced
        if request.force_api_key:
            provider = next((p for p in self.providers if p.api_key == request.force_api_key), None)
            if provider:
                try:
                    return await self.call_provider(provider, request)
                except Exception as e:
                    if not request.use_fallback:
                        raise HTTPException(status_code=500, detail=f"Forced API key failed: {str(e)}")
        
        # Try providers in priority order
        for provider in self.providers:
            if not provider.enabled:
                continue
                
            if not self.check_rate_limit(provider):
                continue
                
            try:
                response = await self.call_provider(provider, request)
                self.update_rate_limit(provider)
                return response
            except Exception as e:
                print(f"Provider {provider.name} failed: {str(e)}")
                continue
        
        # Fallback to g4f if enabled
        if request.use_fallback:
            try:
                return await self.call_g4f(request)
            except Exception as e:
                print(f"g4f fallback failed: {str(e)}")
        
        raise HTTPException(status_code=503, detail="All providers failed")
    
    async def call_provider(self, provider: ProviderConfig, request: RequestModel) -> ChatCompletionResponse:
        """Call a specific provider"""
        
        if provider.name == "openai":
            return await self.call_openai(provider, request)
        elif provider.name == "anthropic":
            return await self.call_anthropic(provider, request)
        elif provider.name == "google":
            return await self.call_google(provider, request)
        else:
            raise Exception(f"Unknown provider: {provider.name}")
    
    async def call_openai(self, provider: ProviderConfig, request: RequestModel) -> ChatCompletionResponse:
        """Call OpenAI API"""
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
            }
            
            if request.stop:
                payload["stop"] = request.stop
            
            response = await client.post(
                f"{provider.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            data = response.json()
            data["provider"] = provider.name
            return ChatCompletionResponse(**data)
    
    async def call_anthropic(self, provider: ProviderConfig, request: RequestModel) -> ChatCompletionResponse:
        """Call Anthropic API (Claude)"""
        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": provider.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # Convert messages format
            system_message = ""
            messages = []
            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)
            
            payload = {
                "model": request.model,
                "messages": messages,
                "max_tokens": request.max_tokens or 1024,
                "temperature": request.temperature,
            }
            
            if system_message:
                payload["system"] = system_message
            
            response = await client.post(
                f"{provider.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            # Convert to OpenAI format
            openai_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data["content"][0]["text"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data["usage"]["input_tokens"],
                    "completion_tokens": data["usage"]["output_tokens"],
                    "total_tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"]
                },
                "provider": provider.name
            }
            
            return ChatCompletionResponse(**openai_response)
    
    async def call_google(self, provider: ProviderConfig, request: RequestModel) -> ChatCompletionResponse:
        """Call Google Gemini API"""
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Convert messages format
            contents = []
            for msg in request.messages:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
            
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": request.temperature,
                    "maxOutputTokens": request.max_tokens or 1024,
                    "topP": request.top_p,
                }
            }
            
            response = await client.post(
                f"{provider.base_url}/models/{request.model}:generateContent?key={provider.api_key}",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Google API error: {response.status_code} - {response.text}")
            
            data = response.json()
            
            # Convert to OpenAI format
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            openai_response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data.get("usageMetadata", {}).get("promptTokenCount", 0),
                    "completion_tokens": data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
                    "total_tokens": data.get("usageMetadata", {}).get("totalTokenCount", 0)
                },
                "provider": provider.name
            }
            
            return ChatCompletionResponse(**openai_response)
    
    async def call_g4f(self, request: RequestModel) -> ChatCompletionResponse:
        """Call g4f as fallback"""
        try:
            # Use RetryProvider for better reliability
            provider = RetryProvider(self.g4f_providers, shuffle=True)
            
            # Convert messages format
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Get response from g4f
            response = await g4f.ChatCompletion.acreate(
                model=request.model.replace("g4f-", ""),
                messages=messages,
                provider=provider,
                temperature=request.temperature,
            )
            
            # Convert to OpenAI format
            openai_response = {
                "id": f"chatcmpl-g4f-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # g4f doesn't provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "provider": "g4f"
            }
            
            return ChatCompletionResponse(**openai_response)
            
        except Exception as e:
            raise Exception(f"g4f error: {str(e)}")
    
    def check_rate_limit(self, provider: ProviderConfig) -> bool:
        """Check if provider is within rate limits"""
        now = time.time()
        key = provider.name
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old entries
        self.rate_limits[key] = [
            timestamp for timestamp in self.rate_limits[key]
            if now - timestamp < 60  # 1 minute window
        ]
        
        return len(self.rate_limits[key]) < provider.rate_limit
    
    def update_rate_limit(self, provider: ProviderConfig):
        """Update rate limit counter"""
        now = time.time()
        key = provider.name
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        self.rate_limits[key].append(now)

# Create app instance
app_instance = App()
app = app_instance.app

if __name__ == "__main__":
    # Load environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run server
    uvicorn.run(
        app,
        host=app_instance.config.HOST,
        port=app_instance.config.PORT,
        reload=app_instance.config.RELOAD,
        log_level=app_instance.config.LOG_LEVEL
    )