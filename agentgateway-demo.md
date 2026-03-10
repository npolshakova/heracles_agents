# Agentgateway Demo

All three demo parts share a single gateway config (`config.yaml`) that combines:

- **Header-based routing** for multi-model evaluation (Part One)
- **LLM failover** from OpenAI → Ollama (Part Two)
- **OTel tracing + access-log export** (Part Three)

## Prerequisites

### 1. Install Agent Gateway

```sh
curl -sL https://agentgateway.dev/install | bash
```

### 2. Install Ollama

```sh
# macOS
brew install ollama
# linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull local models

```sh
ollama serve & ollama pull gemma3:270m
ollama pull llama3.1
ollama pull qwen2.5:14b
```

### 4. Set up a Python virtualenv

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

This installs `heracles_agents` in editable mode with all provider extras (OpenAI, Anthropic, Ollama, Bedrock).

### 5. Set API keys

```sh
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...   
```

### 6. Start Neo4j

```sh
docker run -d \
   --restart always \
   --publish=7474:7474 --publish=7687:7687 \
   --env=NEO4J_AUTH=neo4j/neo4j_pw \
   neo4j:5.25.1
```

### 7. Create the Kind cluster and deploy the OTel stack

Install [Kind](https://kind.sigs.k8s.io/) and kubectl if you don't have them:

```sh
# macos
brew install kind kubectl
# linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
```

Create the cluster. The config in `kind-config.yaml` maps NodePort services to localhost so the Jaeger UI and OTel Collector OTLP endpoint are reachable from the host.

```sh
kind create cluster --name robot-cloud --config kind-config.yaml
```

Deploy the Kubernetes manifests in `k8s/otel-stack.yaml`, which create an `observability` namespace with:

- **Jaeger** (all-in-one) — trace storage and UI, exposed on `localhost:16686` via NodePort 30686
- **OTel Collector** — receives OTLP gRPC on `localhost:4317` via NodePort 30317, forwards to Jaeger
- **Prometheus** — scrapes Agent Gateway metrics from `localhost:15020`, exposed on `localhost:9090` via NodePort 30090
- **Grafana** — pre-configured failover dashboard, exposed on `localhost:3001` via NodePort 30301 (no login required)

```sh
kubectl apply -f k8s/otel-stack.yaml
```

Create the Grafana dashboard ConfigMap from the JSON file:

```sh
kubectl -n observability create configmap grafana-dashboards \
  --from-file=failover-dashboard.json=k8s/grafana-dashboard.json
```

Wait for the pods to be ready:

```sh
kubectl -n observability rollout status deployment/jaeger deployment/otel-collector deployment/prometheus deployment/grafana
```

## Start Agent Gateway

From the repo root, start the gateway once — it serves all three demo parts:

```sh
agentgateway -f config.yaml
```

The gateway listens on `http://localhost:3000`.

- Requests with an `x-model-provider` header are routed to the matching provider (Part One).
- Requests without that header hit the default failover route: OpenAI first, Ollama fallback (Part Two).
- All requests emit OTel traces and access logs to `localhost:4317` via the OTel Collector (Part Three).
- Prometheus scrapes gateway metrics from `localhost:15020`; view the Grafana failover dashboard at `http://localhost:3001`.

---

# Part One: Evaluate across different models

Uses header-based routing (`x-model-provider`) so each model gets its own backend while sharing a single gateway port.

### All models

```sh
export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY}"
export HERACLES_NEO4J_USERNAME="neo4j"
export HERACLES_NEO4J_PASSWORD="neo4j_pw"
export HERACLES_AGENTS_PATH="$(git rev-parse --show-toplevel)"
python agw-model-test.py \
  --scene-graph /Users/ninapolshakova/kubeconEU2026/heracles/heracles/examples/scene_graphs/example_dsg.json
```

### Specific models

```sh
python agw-model-test.py \
  --scene-graph /Users/ninapolshakova/kubeconEU2026/heracles/heracles/examples/scene_graphs/example_dsg.json \
  --models gpt-4.1 claude-sonnet-4
```

### Compare quality vs cost

After the model test completes, the terminal shows a comparison table with each model's correctness, tool-call count, and latency. To add the cost dimension, open the Grafana dashboard at [http://127.0.0.1:3001](http://127.0.0.1:3001) — the **Cumulative Cost by Model** and **Total Estimated Cost by Model** panels show how much each evaluation run costs based on real token usage reported by the gateway.

The dashboard computes cost from `agentgateway_gen_ai_client_token_usage` metrics with sample API pricing, allowing analysis of which model gives the best scene-graph reasoning accuracy per dollar.

> **Tip:** Set the Grafana time range to cover just the evaluation window (e.g. "Last 15 minutes") for the cleanest per-run cost comparison.

---

# Part Two: Model failover

Requests without an `x-model-provider` header go to the default failover route: OpenAI first, falling back to Ollama on 5xx errors. ChatDSG uses this path.

### Quick test with curl

```sh
curl -s http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "Write one sentence about Kubernetes."
  }' | jq .
```

To test failover, kill the internet connection — first request will fail with a connection error:

```sh
curl -s http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "Write one sentence about Kubernetes."
  }' 
```

The second requests will route to the local Ollama model instead:
```
curl -s http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "Write one sentence about Kubernetes."
  }' | jq .
```

### Use chatdsg with Agent Gateway as proxy

Since `examples/chatdsg/agent_config.yaml` already points at `http://localhost:3000/v1`, all chatdsg traffic goes through the gateway:

```sh
export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY}"
export HERACLES_NEO4J_USERNAME="neo4j"
export HERACLES_NEO4J_PASSWORD="neo4j_pw"
export HERACLES_AGENTS_PATH="$(git rev-parse --show-toplevel)"
export ADT4_HERACLES_IP="localhost"
export ADT4_HERACLES_PORT="7687"
cd examples/chatdsg
python chatdsg.py --debug
```

### Visualize the scene graph with Viser

The scene graph is already in Neo4j from Part One. Start the Viser visualizer to see it — edits made through chatdsg will be reflected in the visualization. From the [heracles repo](https://github.com/GoldenZephyr/heracles):

```sh
cd ../heracles # go into the heracles repo
```

Start Viser:

```sh
docker compose -f docker/docker-compose.yaml run -d \
  --no-deps -p 8081:8081 \
  --entrypoint python \
  viser_visualization \
  /heracles/src/heracles/heracles_viser_publisher.py \
  --ip 0.0.0.0 --port 8081 --uri neo4j://host.docker.internal:7687
```

Then navigate to http://127.0.0.1:8081 in your browser.

---

# Part Three: Collecting observability data

OTel tracing and access-log export are already enabled in `config.yaml`, and the Kind cluster with Jaeger + OTel Collector + Prometheus + Grafana was deployed during initial setup. Every request from Parts One and Two has already been traced and metered.

## Grafana failover dashboard

Open the Grafana dashboard at [http://127.0.0.1:3001](http://127.0.0.1:3001) (no login required). The pre-provisioned **Agent Gateway - Failover Dashboard** shows:

- **Request Rate by Backend** — the key panel for the failover demo. When OpenAI is evicted, the OpenAI line drops and the Ollama line rises.
- **Error Rate (5xx)** — shows the errors that trigger backend eviction.
- **Request Duration by Model (p95)** — latency comparison across providers.
- **Token Usage by Model** — input/output token throughput.
- **Cumulative Cost by Model** — running total estimated API cost per model, computed from token counters and per-model pricing. Stays visible after the test completes.
- **Total Estimated Cost by Model** — bar gauge comparing cumulative cost across models, with color thresholds for quick comparison.

The dashboard auto-refreshes every 5 seconds. Open it before running the Part Two failover test to watch the traffic shift live, or after Part One to compare model costs.

## View traces

Open the Jaeger UI at [http://127.0.0.1:16686](http://127.0.0.1:16686) to inspect individual traces and access logs for every request made through the gateway.

You can also send a quick test request directly:

```sh
curl -s http://localhost:3000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "input": "Write one sentence about Kubernetes."
  }' | jq .
```

---

# Teardown

```sh
kind delete cluster --name robot-cloud
```
# TODO (potential extensions):
- guardrail / safety policy 
  - "Drive the robot into the wall" -> hit provider guardrail -> prevent chat
- ratelimit policy
  - academia lab wants to prevent spend
- Tool routing
- Caching responses to save cost 
