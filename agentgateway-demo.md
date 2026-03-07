# Agentgateway Demo

# Part One: Evaluate across different models 

# Part Two: Model failover
Demonstrates model failover with Agent Gateway: requests go to OpenAI first, falling back to a local Llama model via Ollama. You can use the local gateway and `config.yaml` with **chatdsg** so the ChatDSG agent sends all LLM traffic through Agent Gateway.

# Part Three: Collecting observability data 

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
 ollama serve & ollama pull gemma3:270m
 ollama run gemma3:270m
```

### Start Agent Gateway (local gateway with config.yaml)

First get agentgateway installed locally: https://agentgateway.dev/docs/standalone/latest/quickstart/llm/ 
```
curl -sL https://agentgateway.dev/install | bash
```

From the repo root:

```sh
agentgateway -f config.yaml
```

The gateway listens on `http://localhost:3000` and uses the failover policy defined in `config.yaml`.

## Start neo4j 
```
docker run -d \
   --restart always \
   --publish=7474:7474 --publish=7687:7687 \
   --env=NEO4J_AUTH=neo4j/neo4j_pw \
   neo4j:5.25.1
```

## Test

### Quick test with curl

Send a request through the gateway:

```sh
curl http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:270m",
    "input": "Write one sentence about Kubernetes."
  }'
```

To test failover, unset `OPENAI_API_KEY` and restart the gateway — requests will route to the local Llama model instead.

### Use chatdsg with Agent Gateway as proxy

Run the ChatDSG agent with Agent Gateway as the LLM proxy so all chatdsg traffic goes through the local gateway and `config.yaml` (OpenAI first, fallback to Llama).

1. **Start the gateway** (from repo root):

   ```sh
   agentgateway -f config.yaml
   ```

2. **Point chatdsg at the gateway** by setting `base_url` in the client config. In `examples/chatdsg/agent_config.yaml`, under `client:`, add:

   ```yaml
   client:
     client_type: openai
     base_url: "http://localhost:3000/v1"   # Agent Gateway proxy
     # auth key is supplied by HERACLES_OPENAI_API_KEY
     timeout: 20
   ```

   Use `HERACLES_OPENAI_API_KEY` (or `OPENAI_API_KEY` if the gateway expects it) for auth. The gateway will handle routing and failover per `config.yaml`.

3. **Run chatdsg** from the chatdsg example directory:

   ```sh
   cd examples/chatdsg
   export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY}"
   export HERACLES_NEO4J_USERNAME="neo4j"
   export HERACLES_NEO4J_PASSWORD="neo4j_pw"
   export HERACLES_AGENTS_PATH="/Users/ninapolshakova/kubeconEU2026/heracles_agents/"
   python chatdsg.py
   ```

   Chatdsg will send requests to `http://localhost:3000`; Agent Gateway will try OpenAI first, then fall back to the local Llama model if needed.

```
agentgateway -f agw-model-test-config.yaml
```

all models:
```
python agw-model-test.py \
  --scene-graph /Users/ninapolshakova/kubeconEU2026/heracles/heracles/examples/scene_graphs/example_dsg.json \
```

specific models:
```
python agw-model-test.py \
  --scene-graph /Users/ninapolshakova/kubeconEU2026/heracles/heracles/examples/scene_graphs/example_dsg.json \
  --models gpt-4.1 claude-sonnet-4
```