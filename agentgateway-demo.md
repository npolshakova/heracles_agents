# Agentgateway Demo

# Part One: Evaluate across different models 

# Part Two: Model failover
Demonstrates model failover with Agent Gateway: requests go to OpenAI first, falling back to a local Llama model via Ollama. You can use the local gateway and `config.yaml` with **chatdsg** so the ChatDSG agent sends all LLM traffic through Agent Gateway.

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

### Visualize the scene graph with Viser

Run the Viser visualizer to see the scene graph stored in Neo4j. From the [heracles repo](https://github.com/GoldenZephyr/heracles):

First, load a scene graph into Neo4j (if you haven't already via chatdsg):

```sh
cd /Users/ninapolshakova/kubeconEU2026/heracles
docker compose -f docker/docker-compose.yaml run --rm \
  --no-deps \
  --entrypoint python \
  cli \
  /heracles/examples/load_scene_graph.py \
  --scene_graph /heracles/examples/scene_graphs/example_dsg.json
```

Then start Viser:

```sh
docker compose -f docker/docker-compose.yaml run -d \
  --no-deps -p 8081:8081 \
  --entrypoint python \
  viser_visualization \
  /heracles/src/heracles/heracles_viser_publisher.py \
  --ip 0.0.0.0 --port 8081 --uri neo4j://host.docker.internal:7687
```

Then navigate to http://127.0.0.1:8081 in your browser.

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

# Part Three: Collecting observability data

Demonstrates how to collect OTel traces and access logs from Agent Gateway, running the observability stack (Jaeger + OTel Collector) in a Kind cluster.

## Prerequisites

- Docker
- [Kind](https://kind.sigs.k8s.io/) (`brew install kind`)
- kubectl
- Agent Gateway installed (see Part One)

## 1. Create the Kind cluster

The cluster config in `kind-config.yaml` maps NodePort services to localhost so the Jaeger UI and OTel Collector OTLP endpoint are reachable from the host.

```sh
kind create cluster --name agw-otel --config kind-config.yaml
```

## 2. Deploy Jaeger and the OTel Collector

The Kubernetes manifests in `k8s/otel-stack.yaml` create an `observability` namespace with:

- **Jaeger** (all-in-one) — trace storage and UI, exposed on `localhost:16686` via NodePort 30686
- **OTel Collector** — receives OTLP gRPC on `localhost:4317` via NodePort 30317, forwards to Jaeger

The OTel Collector config is embedded in a ConfigMap inside the manifest (using the Jaeger in-cluster service name `jaeger.observability.svc.cluster.local:4317`).

```sh
kubectl apply -f k8s/otel-stack.yaml
```

Wait for the pods to be ready:

```sh
kubectl -n observability rollout status deployment/jaeger deployment/otel-collector
```

## 3. Start Agent Gateway with OTel + LLM failover

Both OTel configs include the full LLM failover backend (OpenAI → Ollama) so all chatdsg tool and provider calls go through the gateway and are traced:

- **`agw-otel-tracing-config.yaml`** — tracing only (`frontendPolicies.tracing`).
- **`agw-otel-full-config.yaml`** — tracing + OTLP access logs (`frontendPolicies.accessLog.otlp`).

Both send OTLP data to `localhost:4317`, which the Kind NodePort mapping routes to the in-cluster OTel Collector.

```sh
agentgateway -f agw-otel-full-config.yaml
```

## 4. Run chatdsg through the gateway

Since `agent_config.yaml` already points at `http://localhost:3000/v1`, all LLM and tool calls from chatdsg are routed through Agent Gateway and traced.

```sh
cd examples/chatdsg
export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY}"
export HERACLES_NEO4J_USERNAME="neo4j"
export HERACLES_NEO4J_PASSWORD="neo4j_pw"
export HERACLES_AGENTS_PATH="/Users/ninapolshakova/kubeconEU2026/heracles_agents/"
python chatdsg.py
```

## 5. View traces

Open the Jaeger UI at [http://127.0.0.1:16686](http://127.0.0.1:16686) to see traces and access logs for every request chatdsg made through the gateway.

You can also send a quick test request directly:

```sh
curl http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "Write one sentence about Kubernetes."
  }'
```

## Teardown

```sh
kind delete cluster --name agw-otel
```