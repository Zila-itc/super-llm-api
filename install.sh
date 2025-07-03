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
