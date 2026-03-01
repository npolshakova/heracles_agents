# Agent Gateway Failover Demo

Demonstrates model failover with Agent Gateway: requests go to OpenAI first, falling back to a local Llama model via Ollama.

## Prerequisites

### 1. Install Agent Gateway

```sh
curl -sL https://agentgateway.dev/install | bash
```

### 2. Install Ollama

```sh
# macOS
brew install ollama
```

### 3. Pull the Llama model

```sh
ollama pull llama3.2
```

### 4. Set your OpenAI API key

```sh
export OPENAI_API_KEY=sk-...
```

## Run

### Start Ollama (if not already running)

```sh
ollama serve
```

### Start Agent Gateway

```sh
agentgateway -f config.yaml
```

## Test

Send a request through the gateway:

```sh
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "any", "messages": [{"role": "user", "content": "Say hello in one sentence."}]}'
```

To test failover, unset `OPENAI_API_KEY` and restart the gateway — requests will route to the local Llama model instead.
