#!/bin/sh
set -eu

echo "Initializing Ollama models..."

# Give the service a moment to settle after health check passes.
sleep 5

if ollama list | grep -q "nomic-embed-text"; then
	echo "nomic-embed-text already present, skipping pull."
else
	echo "Pulling nomic-embed-text..."
	ollama pull nomic-embed-text
fi

if ollama list | grep -q "llama3.2:latest"; then
	echo "llama3.2:latest already present, skipping pull."
else
	echo "Pulling llama3.2:latest..."
	ollama pull llama3.2:latest
fi

if ollama list | grep -q "smollm2:360m"; then
	echo "smollm2:360m already present, skipping pull."
else
	echo "Pulling smollm2:360m..."
	ollama pull smollm2:360m
fi

echo "Ollama model initialization complete."
