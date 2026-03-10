# Agentgateway Demo

All demo parts share a single gateway config (`config.yaml`) that combines:

- **Header-based routing** for multi-model evaluation (Part One)
- **LLM failover** from OpenAI → Ollama (Part Two)
- **OTel tracing + access-log export** (Part Three)
- **Tool routing** — Neo4j failover from in-cluster → local Docker (HTTP route on `:4000` with eviction policy)

## Prerequisites

### 1. Install agentgateway

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

### 6. Start local Neo4j (failover target)

The local Docker instance serves as the fallback if the in-cluster Neo4j is unavailable. It is published on the default ports (`7474`/`7687`).

```sh
docker run -d \
   --restart always \
   --publish=7474:7474 --publish=7687:7687 \
   --env=NEO4J_AUTH=neo4j/neo4j_pw \
   neo4j:5.25.1
```

### 7. Create the Kind cluster and deploy the cloud setup

Install [Kind](https://kind.sigs.k8s.io/) and kubectl if you don't have them:

```sh
# macos
brew install kind kubectl
# linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
```

Create the cluster. The config in `kind-config.yaml` maps NodePort services to localhost so the Jaeger UI, OTel Collector OTLP endpoint, and in-cluster Neo4j are reachable from the host.

```sh
kind create cluster --name robot-cloud --config kind-config.yaml
```

Deploy the Kubernetes manifests in `k8s/cloud-setup.yaml`, which create an `observability` namespace with:

- **Neo4j** — primary scene-graph database, bolt on `localhost:17687` via NodePort 30787, web UI on `localhost:17474` via NodePort 30747
- **Jaeger** (all-in-one) — trace storage and UI, exposed on `localhost:16686` via NodePort 30686
- **OTel Collector** — receives OTLP gRPC on `localhost:4317` via NodePort 30317, forwards to Jaeger
- **Prometheus** — scrapes agentgateway metrics from `localhost:15020`, exposed on `localhost:9090` via NodePort 30090
- **Grafana** — pre-configured failover dashboard, exposed on `localhost:3001` via NodePort 30301 (no login required)

```sh
kubectl apply -f k8s/cloud-setup.yaml
```

Create the Grafana dashboard ConfigMap from the JSON file:

```sh
kubectl -n observability create configmap grafana-dashboards \
  --from-file=failover-dashboard.json=k8s/grafana-dashboard.json
```

Wait for the pods to be ready:

```sh
kubectl -n observability rollout status deployment/neo4j deployment/jaeger deployment/otel-collector deployment/prometheus deployment/grafana
```

## Start agentgateway

From the repo root, start the gateway once — it serves all three demo parts:

```sh
agentgateway -f config.yaml
```

The gateway starts two listeners:

- **`:3000` (HTTP)** — LLM API traffic. Requests with an `x-model-provider` header are routed to the matching provider (Part One). Requests without that header hit the default failover route: OpenAI first, Ollama fallback (Part Two).
- **`:4000` (HTTP)** — Neo4j tool-call traffic. Cypher queries are sent via Neo4j's HTTP API. The in-cluster Neo4j is primary; on 5xx errors it is evicted for 30s and the local Docker instance takes over — the same health/eviction policy as the LLM failover.

All HTTP requests emit OTel traces and access logs to `localhost:4317` via the OTel Collector (Part Three). Prometheus scrapes gateway metrics from `localhost:15020`; view the Grafana failover dashboard at `http://localhost:3001`.

---

# Part One: Evaluate across different models

Uses header-based routing (`x-model-provider`) so each model gets its own backend while sharing a single gateway port.

### Load the scene graph into both Neo4j instances

Load the scene graph into both Neo4j instances via bolt (direct). Tool calls during evaluation go through the gateway HTTP route (`:4000`) with eviction-based failover.

```sh
export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY}"
export HERACLES_NEO4J_USERNAME="neo4j"
export HERACLES_NEO4J_PASSWORD="neo4j_pw"
export HERACLES_AGENTS_PATH="$(git rev-parse --show-toplevel)"
```

In-cluster (bolt via NodePort):

```sh
python agw-model-test.py \
  --scene-graph /Users/ninapolshakova/kubeconEU2026/heracles/heracles/examples/scene_graphs/example_dsg.json
```

Local Docker fallback (bolt direct):

```sh
python agw-model-test.py --db-port 7687 \
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

### Use chatdsg with agentaateway as proxy

Both LLM calls and Neo4j tool calls go through the gateway. `agent_config.yaml` points the LLM at `http://localhost:3000/v1` and Neo4j at `http://localhost:4000` (the gateway HTTP route with eviction failover).

```sh
export HERACLES_OPENAI_API_KEY="${OPENAI_API_KEY}"
export HERACLES_NEO4J_USERNAME="neo4j"
export HERACLES_NEO4J_PASSWORD="neo4j_pw"
export HERACLES_AGENTS_PATH="$(git rev-parse --show-toplevel)"
cd examples/chatdsg
python chatdsg.py --debug
```

### Test Neo4j tool-call failover

This mirrors the LLM failover demo but for tool calls. The gateway routes Cypher HTTP requests to the in-cluster Neo4j; on 5xx errors or connection failures it evicts that backend for 30s and traffic shifts to the local Docker instance.

**1. Verify the in-cluster Neo4j is serving queries**

```sh
curl -s -u neo4j:neo4j_pw http://localhost:4000/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN count(n) AS c"}]}' | jq .
```

**2. Scale down the in-cluster Neo4j**

```sh
kubectl -n observability scale deployment/neo4j --replicas=0
kubectl -n observability rollout status deployment/neo4j
```

The next request to `:4000` returns a 5xx, which triggers the eviction policy. The in-cluster backend is evicted for 30s.

**3. Confirm tool calls still work (via fallback)**

```sh
curl -s -u neo4j:neo4j_pw http://localhost:4000/db/neo4j/tx/commit \
  -H "Content-Type: application/json" \
  -d '{"statements":[{"statement":"MATCH (n) RETURN count(n) AS c"}]}' | jq .
```

You can also run a chatdsg query — tool calls now transparently hit the local Docker Neo4j through the gateway. Watch the **Neo4j Tool-Call Rate by Backend** Grafana panel to see the traffic shift live.

**4. Restore the in-cluster Neo4j**

```sh
kubectl -n observability scale deployment/neo4j --replicas=1
kubectl -n observability rollout status deployment/neo4j
```

After the 30s eviction window expires, subsequent requests are routed back to the in-cluster instance (weight 10 vs 1).

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
  --ip 0.0.0.0 --port 8081 --uri bolt://host.docker.internal:17687
```

Then navigate to http://127.0.0.1:8081 in your browser.

---

# Part Three: Collecting observability data

OTel tracing and access-log export are already enabled in `config.yaml`, and the Kind cluster with Jaeger + OTel Collector + Prometheus + Grafana was deployed during initial setup. Every request from Parts One and Two has already been traced and metered.

## Grafana failover dashboard

Open the Grafana dashboard at [http://127.0.0.1:3001](http://127.0.0.1:3001) (no login required). The pre-provisioned **agentgateway - Failover Dashboard** shows:

- **Request Rate by Backend** — the key panel for the failover demo. When OpenAI is evicted, the OpenAI line drops and the Ollama line rises.
- **Error Rate (5xx)** — shows the errors that trigger backend eviction.
- **Request Duration by Model (p95)** — latency comparison across providers.
- **Token Usage by Model** — input/output token throughput.
- **Cumulative Cost by Model** — running total estimated API cost per model, computed from token counters and per-model pricing. Stays visible after the test completes.
- **Total Estimated Cost by Model** — bar gauge comparing cumulative cost across models, with color thresholds for quick comparison.
- **Neo4j Tool-Call Rate by Backend** — the key panel for tool-call failover. Shows connection rate to the in-cluster Neo4j vs local Docker fallback. When the in-cluster instance is scaled down, the in-cluster line drops and the local line rises.
- **Neo4j Total Connections by Backend** — bar gauge showing cumulative connection counts to each Neo4j backend.

The dashboard auto-refreshes every 5 seconds. Open it before running the Part Two failover test to watch the traffic shift live, after Part One to compare model costs, or during the Neo4j failover test to see tool-call traffic shift between backends.

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
- Caching responses to save cost 
