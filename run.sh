#!/bin/bash

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi


echo "📦 Installing dependencies..."
uv pip install -r requirements.txt

echo "🚀 Starting server..."
uvicorn main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --reload
