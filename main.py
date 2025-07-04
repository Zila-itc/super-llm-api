#!/usr/bin/env python3
"""
Advanced OpenAI-Compatible Server with Enhanced G4F Integration
Provides a unified API endpoint with intelligent provider selection and robust fallback
"""

import asyncio
import json
import os
import random
import time
import uuid
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import logging
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import httpx
import g4f

# Configuration
class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    G4F_API_KEY: Optional[str] = None

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    LOG_LEVEL: str = "info"
    CORS_ORIGINS: List[str] = ["*"]

    # Rate Limiting (requests per minute)
    OPENAI_RATE_LIMIT: int = 60
    ANTHROPIC_RATE_LIMIT: int = 50
    GOOGLE_RATE_LIMIT: int = 60
    G4F_RATE_LIMIT: int = 100

    # G4F Configuration
    G4F_ENABLED: bool = True
    G4F_RETRY_ATTEMPTS: int = 3
    G4F_TIMEOUT: int = 60
    G4F_BASE_URL: Optional[str] = None
    G4F_WEB_SEARCH: bool = False
    G4F_IMAGE_PROVIDER: Optional[str] = None
    
    # Advanced Configuration
    CONFIG_FILE: str = "config.yaml"
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 120

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
    
    # G4F specific fields
    web_search: Optional[bool] = None
    proxy: Optional[str] = None
    timeout: Optional[int] = None
    image_provider: Optional[str] = None
    
    # Advanced options
    retry_attempts: Optional[int] = None
    provider_priority: Optional[List[str]] = None

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
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("App initialization started.")

        self.app = FastAPI(
            title="Advanced OpenAI-Compatible Server with G4F",
            description="Unified API endpoint with intelligent provider selection and robust fallback",
            version="2.0.0"
        )
        
        # Add CORS middleware
        @self.app.get("/")
        async def read_root():
            return {"message": "Welcome to the Super LLM API!"}

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.templates = Jinja2Templates(directory="templates")

        self.logger.info("Loading YAML configuration...")
        self.yaml_config = self.load_yaml_config()
        self.logger.info("YAML configuration loaded.")
        
        self.logger.info("Loading providers...")
        self.providers = self.load_providers()
        self.logger.info(f"Loaded {len(self.providers)} providers.")
        if not self.providers:
            self.logger.warning("No providers loaded. This might cause the application to shut down if no fallback is available.")
        self.rate_limits = {}

        self.logger.info("Initializing G4F clients...")
        self.g4f_client = None
        self.g4f_async_client = None
        self.logger.info("G4F clients initialized.")
        
        self.logger.info("Setting up G4F providers...")
        self.g4f_providers = self.get_g4f_providers()
        self.logger.info(f"G4F providers setup. Count: {len(self.g4f_providers)}")
        
        self.logger.info("Setting up routes...")
        self.setup_routes()
        self.logger.info("Routes setup complete.")
        self.logger.info("Application setup complete. Ready to serve requests.")
        
        self.logger.info(f"Server initialized with {len(self.providers)} providers and G4F enabled: {self.config.G4F_ENABLED}")
    
    def load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file if it exists"""
        try:
            if os.path.exists(self.config.CONFIG_FILE):
                with open(self.config.CONFIG_FILE, 'r') as f:
                    config = yaml.safe_load(f)
                    self.logger.info(f"Loaded configuration from {self.config.CONFIG_FILE}")
                    return config
        except Exception as e:
            self.logger.warning(f"Failed to load YAML config: {e}")
        return {}
    

    
    def get_g4f_providers(self) -> List[Any]:
        """Get available G4F providers based on configuration"""
        if not self.config.G4F_ENABLED:
            return []
        
        # Default providers with better reliability
        default_providers = []
        try:
            if hasattr(g4f, 'Provider'):
                provider_names = ['Bing', 'ChatgptX', 'GPTalk', 'Liaobots', 'Phind', 'Yqcloud', 'Blackboxai', 'DDG']
                for name in provider_names:
                    if hasattr(g4f.Provider, name):
                        default_providers.append(getattr(g4f.Provider, name))
        except Exception as e:
            self.logger.warning(f"Failed to load G4F providers: {e}")
        
        # Load from YAML config if available
        if 'g4f' in self.yaml_config and 'providers' in self.yaml_config['g4f']:
            yaml_providers = []
            for provider_name in self.yaml_config['g4f']['providers']:
                try:
                    provider = getattr(g4f.Provider, provider_name)
                    yaml_providers.append(provider)
                except AttributeError:
                    self.logger.warning(f"G4F provider {provider_name} not found")
            
            if yaml_providers:
                self.logger.info(f"Using G4F providers from config: {[p.__name__ for p in yaml_providers]}")
                return yaml_providers
        
        self.logger.info(f"Using default G4F providers: {[p.__name__ for p in default_providers]}")
        return default_providers
        
    def load_providers(self) -> List[ProviderConfig]:
        """Load provider configurations from environment or config file"""
        providers = []
        if 'providers' in self.yaml_config:
            for name, config in self.yaml_config['providers'].items():
                if config.get('enabled', False):
                    api_key = None
                    if name.lower() == 'openai':
                        api_key = self.config.OPENAI_API_KEY
                    elif name.lower() == 'anthropic':
                        api_key = self.config.ANTHROPIC_API_KEY
                    elif name.lower() == 'google':
                        api_key = self.config.GOOGLE_API_KEY
                    else:
                        api_key_env_name = config.get('api_key_env')
                        if isinstance(api_key_env_name, str):
                            api_key = os.getenv(api_key_env_name)
                        else:
                            self.logger.warning(f"Provider '{name}' has an invalid or missing 'api_key_env' configuration.")

                    if api_key:
                        providers.append(ProviderConfig(
                            name=name.lower(),
                            api_key=api_key,

                            base_url=config.get('base_url'),
                            models=config.get('models', []),
                            priority=config.get('priority', 99),
                            rate_limit=config.get('rate_limit', '10/minute'),

                            enabled=config.get('enabled', True)
                        ))
                    else:
                        self.logger.warning(f"API key for provider {name} not found in environment variable {config.get('api_key_env')}")

        return sorted(providers, key=lambda x: x.priority)
    
    def setup_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: RequestModel):
            return await self.handle_chat_completion(request)
        
        @self.app.get("/v1/models")
        async def list_models():
            models = []
            for provider_config in self.providers:
                models.append({
                    "id": provider_config.name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "user",
                    "permission": [
                        {
                            "id": f"model-perm-{provider_config.name}",
                            "object": "model_permission",
                            "created": int(time.time()),
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": False,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False
                        }
                    ]
                })
            return {"object": "list", "data": models}
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "providers": len(self.providers)}

        @self.app.get("/preview", response_class=HTMLResponse)
        async def preview(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})

    async def handle_chat_completion(self, request: RequestModel) -> Union[ChatCompletionResponse, StreamingResponse]:
        self.logger.info(f"Received chat completion request for model: {request.model}")
        """Main handler for chat completions with fallback logic and streaming support"""
        
        # Check for G4F model request
        if request.model.startswith("g4f-"):
            self.logger.info(f"G4F model requested: {request.model}")
            if self.config.G4F_ENABLED:
                return await self.call_g4f(request)
            else:
                raise HTTPException(status_code=400, detail="G4F is disabled")
        
        # Check if specific provider is forced
        if request.force_provider:
            provider = next((p for p in self.providers if p.name == request.force_provider), None)
            if provider:
                self.logger.info(f"Force provider: {provider.name}")
                try:
                    return await self.call_provider(provider, request)
                except Exception as e:
                    self.logger.error(f"Forced provider {request.force_provider} failed: {str(e)}")
                    if not request.use_fallback:
                        raise HTTPException(status_code=500, detail=f"Provider {request.force_provider} failed: {str(e)}")
            else:
                self.logger.warning(f"Forced provider {request.force_provider} not found.")
                if not request.use_fallback:
                    raise HTTPException(status_code=400, detail=f"Provider {request.force_provider} not found")
        
        # Check if specific API key is forced
        if request.force_api_key:
            provider = next((p for p in self.providers if p.api_key == request.force_api_key), None)
            if provider:
                self.logger.info(f"Force API key for provider: {provider.name}")
                try:
                    return await self.call_provider(provider, request)
                except Exception as e:
                    self.logger.error(f"Forced API key for provider {provider.name} failed: {str(e)}")
                    if not request.use_fallback:
                        raise HTTPException(status_code=500, detail=f"Forced API key failed: {str(e)}")
            else:
                self.logger.warning("Forced API key not found in any configured provider.")
                if not request.use_fallback:
                    raise HTTPException(status_code=400, detail="Forced API key not found in any configured provider")
        
        # Apply retry logic if specified
        max_retries = request.retry_attempts or self.config.MAX_RETRIES
        
        # Try providers in priority order (with custom priority if specified)
        providers_to_try = self.providers
        if request.provider_priority:
            # Reorder providers based on request priority
            priority_map = {name: i for i, name in enumerate(request.provider_priority)}
            providers_to_try = sorted(
                self.providers,
                key=lambda x: (priority_map.get(x.name.lower(), 999), x.priority)
            )
        else:
            providers_to_try = sorted(self.providers, key=lambda x: x.priority)
        
        for retry in range(max_retries):
            for provider in providers_to_try:
                if not provider.enabled:
                    self.logger.info(f"Skipping disabled provider: {provider.name}")
                    continue
                    
                if not self.check_rate_limit(provider):
                    self.logger.warning(f"Rate limit exceeded for provider: {provider.name}")
                    continue
                    
                self.logger.info(f"Attempting to use provider: {provider.name} (retry {retry + 1}/{max_retries})")
                try:
                    response = await self.call_provider(provider, request)
                    self.update_rate_limit(provider)
                    self.logger.info(f"Successfully received response from {provider.name}")
                    return response
                except Exception as e:
                    self.logger.error(f"Provider {provider.name} failed (retry {retry + 1}/{max_retries}): {str(e)}")
                    continue
            
            if retry < max_retries - 1:
                self.logger.info(f"All providers failed on retry {retry + 1}, retrying...")
                await asyncio.sleep(1)  # Brief delay before retry
        
        # Fallback to g4f if enabled
        if self.config.G4F_ENABLED and request.use_fallback:
            self.logger.info("All providers failed after retries, falling back to g4f")
            try:
                response = await self.call_g4f(request)
                self.logger.info("Successfully received response from g4f")
                return response
            except Exception as e:
                self.logger.error(f"g4f fallback failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"g4f fallback failed: {str(e)}")
        
        self.logger.error("No suitable provider found or all providers failed")
        raise HTTPException(status_code=503, detail="No suitable provider found or all providers failed")
    
    async def call_provider(self, provider: ProviderConfig, request: RequestModel) -> ChatCompletionResponse:
        """Call a specific provider"""
        
        if provider.name == "openai":
            return await self.call_openai(provider, request)
        elif provider.name == "anthropic":
            return await self.call_anthropic(provider, request)
        elif provider.name == "google":
            return await self.call_google(provider, request)
        elif provider.name == "cohere":
            return await self.call_cohere(provider, request)
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
    
    async def call_cohere(self, provider: ProviderConfig, request: RequestModel) -> ChatCompletionResponse:
        """Call Cohere API"""
        try:
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json"
            }

            # Convert messages to Cohere format
            chat_history = []
            message = ""
            system_message = ""  # preamble

            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                elif msg["role"] == "user":
                    chat_history.append({"role": "USER", "message": msg["content"]})
                elif msg["role"] == "assistant":
                    chat_history.append({"role": "CHATBOT", "message": msg["content"]})

            # Extract the last user message as the main message
            if chat_history and chat_history[-1]["role"] == "USER":
                message = chat_history.pop()["message"]

            if not message and request.messages:
                message = request.messages[-1]["content"]

            payload = {
                "model": request.model,
                "message": message,
                "preamble": system_message,
                "chat_history": chat_history,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "p": request.top_p,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{provider.base_url or 'https://api.cohere.ai/v1'}/chat",
                    headers=headers,
                    json=payload,
                    timeout=provider.timeout
                )
                response.raise_for_status()
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
                        "content": data["text"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": data.get("meta", {}).get("tokens", {}).get("input_tokens", 0),
                    "completion_tokens": data.get("meta", {}).get("tokens", {}).get("output_tokens", 0),
                    "total_tokens": data.get("meta", {}).get("tokens", {}).get("input_tokens", 0) + data.get("meta", {}).get("tokens", {}).get("output_tokens", 0)
                },
                "provider": provider.name
            }
            return ChatCompletionResponse(**openai_response)

        except httpx.HTTPStatusError as e:
            logger.error(f"Cohere API request failed with status {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred in call_cohere: {e}")
            raise

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
    
    async def call_g4f(self, request: RequestModel) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Enhanced G4F call with streaming support and better error handling"""
        try:
            # Prepare request parameters
            model = request.model.replace("g4f-", "")
            
            # Use request-specific settings or fallback to config
            web_search = request.web_search if request.web_search is not None else self.config.G4F_WEB_SEARCH
            timeout = request.timeout or self.config.G4F_TIMEOUT
            
            # Convert messages format
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Handle streaming vs non-streaming
            if request.stream:
                return await self._stream_g4f_response(request, model, messages, web_search, timeout)
            else:
                return await self._get_g4f_response(request, model, messages, web_search, timeout)
                
        except Exception as e:
            self.logger.error(f"G4F error: {str(e)}")
            raise Exception(f"G4F error: {str(e)}")
    
    async def _get_g4f_response(self, request: RequestModel, model: str, messages: List[Dict], web_search: bool, timeout: int) -> ChatCompletionResponse:
        """Get non-streaming G4F response"""
        return await self._fallback_g4f_response(request, model, messages)
    
    async def _fallback_g4f_response(self, request: RequestModel, model: str, messages: List[Dict]) -> ChatCompletionResponse:
        """Fallback to legacy G4F method"""
        try:
            response = await g4f.ChatCompletion.create_async(
                model=model,
                messages=messages,
                proxy=request.proxy,
                timeout=request.timeout or self.config.G4F_TIMEOUT
            )
            
            openai_response = {
                "id": f"chatcmpl-g4f-{uuid.uuid4().hex[:8]}",
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
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "provider": "g4f"
            }
            
            return ChatCompletionResponse(**openai_response)
        except Exception as e:
            self.logger.error(f"G4F fallback error: {e}")
            raise Exception(f"G4F fallback error: {e}")
    
    async def _stream_g4f_response(self, request: RequestModel, model: str, messages: List[Dict], web_search: bool, timeout: int) -> StreamingResponse:
        """Handle streaming G4F response"""
        async def generate_stream():
            try:
                # Bard provider does not support stream=True, so we handle it separately
                if model == "bard":
                    self.logger.warning("Bard provider does not support streaming. Generating full response and streaming it manually.")
                    full_response = await g4f.ChatCompletion.create_async(
                        model=model,
                        messages=messages,
                        proxy=request.proxy,
                        timeout=request.timeout or self.config.G4F_TIMEOUT
                    )

                    async def response_generator(response):
                        yield response
                    
                    response_stream = response_generator(full_response)

                else:
                    response_stream = await g4f.ChatCompletion.create_async(
                        model=model,
                        messages=messages,
                        stream=True,
                        proxy=request.proxy,
                        timeout=request.timeout or self.config.G4F_TIMEOUT
                    )

                chunk_id = f"chatcmpl-g4f-{uuid.uuid4().hex[:8]}"
                full_content = ""
                async for chunk in response_stream:
                    if isinstance(chunk, str):
                        content = chunk
                    elif isinstance(chunk, dict) and 'content' in chunk:
                        content = chunk['content']
                    else:
                        content = str(chunk)

                    full_content += content
                    json_data = json.dumps({
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "logprobs": None,
                            "finish_reason": None
                        }]
                    })
                    yield f'data: {json_data}\n\n'

                # Send final chunk with finish_reason
                yield f"""data: {json.dumps({
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                })}\n\n"""
                yield "data: [DONE]\n\n"

            except Exception as e:
                self.logger.error(f"G4F streaming error: {e}")
                yield f"""data: {json.dumps({"error": str(e)})}\n\n"""
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")
                self.logger.warning(f"Forced provider {request.force_provider} not found.")
                if not request.use_fallback:
                    raise HTTPException(status_code=400, detail=f"Provider {request.force_provider} not found")
        
        # Check if specific API key is forced
        if request.force_api_key:
            provider = next((p for p in self.providers if p.api_key == request.force_api_key), None)
            if provider:
                self.logger.info(f"Force API key for provider: {provider.name}")
                try:
                    return await self.call_provider(provider, request)
                except Exception as e:
                    self.logger.error(f"Forced API key for provider {provider.name} failed: {str(e)}")
                    if not request.use_fallback:
                        raise HTTPException(status_code=500, detail=f"Forced API key failed: {str(e)}")
            else:
                self.logger.warning("Forced API key not found in any configured provider.")
                if not request.use_fallback:
                    raise HTTPException(status_code=400, detail="Forced API key not found in any configured provider")
        
        # Apply retry logic if specified
        max_retries = request.retry_attempts or self.config.MAX_RETRIES
        
        # Try providers in priority order (with custom priority if specified)
        providers_to_try = self.providers
        if request.provider_priority:
            # Reorder providers based on request priority
            priority_map = {name: i for i, name in enumerate(request.provider_priority)}
            providers_to_try = sorted(
                self.providers,
                key=lambda x: (priority_map.get(x.name.lower(), 999), x.priority)
            )
        else:
            providers_to_try = sorted(self.providers, key=lambda x: x.priority)
        
        for retry in range(max_retries):
            for provider in providers_to_try:
                if not provider.enabled:
                    self.logger.info(f"Skipping disabled provider: {provider.name}")
                    continue
                    
                if not self.check_rate_limit(provider):
                    self.logger.warning(f"Rate limit exceeded for provider: {provider.name}")
                    continue
                    
                self.logger.info(f"Attempting to use provider: {provider.name} (retry {retry + 1}/{max_retries})")
                try:
                    response = await self.call_provider(provider, request)
                    self.update_rate_limit(provider)
                    self.logger.info(f"Successfully received response from {provider.name}")
                    return response
                except Exception as e:
                    self.logger.error(f"Provider {provider.name} failed (retry {retry + 1}/{max_retries}): {str(e)}")
                    continue
            
            if retry < max_retries - 1:
                self.logger.info(f"All providers failed on retry {retry + 1}, retrying...")
                await asyncio.sleep(1)  # Brief delay before retry
        
        # Fallback to g4f if enabled
        if self.config.G4F_ENABLED and request.use_fallback:
            self.logger.info("All providers failed after retries, falling back to g4f")
            try:
                response = await self.call_g4f(request)
                self.logger.info("Successfully received response from g4f")
                return response
            except Exception as e:
                self.logger.error(f"g4f fallback failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"g4f fallback failed: {str(e)}")
        
        self.logger.error("No suitable provider found or all providers failed")
        raise HTTPException(status_code=503, detail="No suitable provider found or all providers failed")
    
    def check_rate_limit(self, provider: ProviderConfig) -> bool:
        """Check if provider is within rate limits"""
        current_time = time.time()
        if provider.name not in self.rate_limits:
            self.rate_limits[provider.name] = {'timestamps': [], 'count': 0}

        # Filter out requests older than 1 minute
        self.rate_limits[provider.name]['timestamps'] = [t for t in self.rate_limits[provider.name]['timestamps'] if current_time - t < 60]
        self.rate_limits[provider.name]['count'] = len(self.rate_limits[provider.name]['timestamps'])

        # Parse rate limit from string (e.g., "10/minute")
        rate_limit_str = str(provider.rate_limit)
        try:
            limit_value = int(rate_limit_str.split('/')[0])
        except (ValueError, IndexError):
            limit_value = 60 # Default to 60 if parsing fails

        return self.rate_limits[provider.name]['count'] < limit_value

    def update_rate_limit(self, provider: ProviderConfig):
        """Update rate limit for provider"""
        self.rate_limits[provider.name]['timestamps'].append(time.time())
        self.rate_limits[provider.name]['count'] += 1


# Main entry point for running the FastAPI application
app = App().app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)