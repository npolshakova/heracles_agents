#!/usr/bin/env python3
"""
Test one hard question across multiple models via Agent Gateway.

Runs the full heracles agent pipeline (system prompt, Cypher tools, agent loop)
for each model through agentgateway using the Chat Completions API, then prints
a comparison table.

All providers sit behind a single gateway port.  The new openai_completions
client type uses /v1/chat/completions, which agentgateway supports for every
provider (OpenAI native, Anthropic translation, Gemini native, Ollama native).

Prerequisites:
  - Agent Gateway running: agentgateway -f config.yaml
  - Neo4j running with the scene graph loaded
  - Environment variables:
      OPENAI_API_KEY           — for agentgateway backend auth
      HERACLES_OPENAI_API_KEY  — for the heracles client (can equal OPENAI_API_KEY)
      HERACLES_AGENTS_PATH     — repo root (for prompt YAML files)
  - pip install -e .   (editable install of this repo)
"""

import argparse
import logging
import os
import textwrap
import time

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import load_dsg_to_db

# Ensure the completions agent integration dispatchers are registered
import heracles_agents.provider_integrations.openai.openai_completions_agent_integration  # noqa: F401
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext, EvalQuestion
from heracles_agents.pipelines.agentic_pipeline import generate_prompt
from heracles_agents.pipelines.comparisons import evaluate_answer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_PATH = os.environ.get("HERACLES_AGENTS_PATH", _SCRIPT_DIR)

if not os.path.isdir(os.path.join(AGENTS_PATH, "examples", "chatdsg")):
    logging.warning(
        "HERACLES_AGENTS_PATH=%s does not contain examples/chatdsg; "
        "falling back to script directory %s",
        AGENTS_PATH,
        _SCRIPT_DIR,
    )
    AGENTS_PATH = _SCRIPT_DIR

AGW_GATEWAY = "http://localhost:3000/v1"

MODELS = [
    {"name": "gpt-4.1",          "model": "gpt-4.1",                                  "provider": "openai"},
    {"name": "claude-sonnet-4",  "model": "claude-sonnet-4-20250514",                  "provider": "anthropic"},
    {"name": "qwen2.5:14b",         "model": "qwen2.5:14b",                                  "provider": "ollama"},
    # {"name": "gemini-2.5-flash-lite", "model": "gemini-2.5-flash-lite",                          "provider": "gemini"},
    # {"name": "bedrock-claude-haiku",   "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0", "provider": "bedrock"},
]

DEFAULT_QUESTIONS = {
    "qa": "0A",
    "pddl": "0A",
}

OUTPUT_TYPE_FOR = {
    "SLDP": "SLDP",
    "PDDL": "PDDL",
}


def load_questions(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)["questions"]


def find_question(questions, uid):
    return next((q for q in questions if str(q["uid"]) == uid), None)


def load_scene_graph(dsg_path, neo4j_uri, object_labelspace, room_labelspace):
    """Load a scene graph file into Neo4j (mirrors chatdsg.py behaviour)."""
    print(f"  Loading scene graph from {dsg_path} ...")
    scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_path)
    summarize_dsg(scene_graph)
    neo4j_creds = (
        os.getenv("HERACLES_NEO4J_USERNAME", "neo4j"),
        os.getenv("HERACLES_NEO4J_PASSWORD", "neo4j"),
    )
    load_dsg_to_db(
        object_labelspace,
        room_labelspace,
        neo4j_uri,
        neo4j_creds,
        scene_graph,
    )
    print("  Scene graph loaded!\n")


def make_agent_config(model_name, provider, gateway_url, neo4j_uri):
    return {
        "client": {
            "client_type": "openai_completions",
            "base_url": gateway_url,
            "timeout": 120,
            "default_headers": {"x-model-provider": provider},
        },
        "model_info": {
            "model": model_name,
            "temperature": 0.2,
        },
        "agent_info": {
            "prompt_settings": {
                "base_prompt": os.path.join(
                    AGENTS_PATH, "examples", "chatdsg", "agent_prompt.yaml"
                ),
                "output_type": "SLDP",
                "sldp_answer_type_hint": True,
            },
            "tools": [
                {
                    "name": "run_cypher_query",
                    "bound_args": {
                        "dsgdb_conf": {
                            "dsg_interface_type": "heracles",
                            "uri": neo4j_uri,
                        }
                    },
                }
            ],
            "tool_interface": "openai_completions",
            "max_iterations": 6,
        },
    }


def run_agent_for_question(model_cfg, question_dict, gateway_url, neo4j_uri):
    """Run the full heracles agent pipeline and return results."""
    config = make_agent_config(model_cfg["model"], model_cfg["provider"], gateway_url, neo4j_uri)

    comparison_type = question_dict["correctness_comparator"]["comparison_type"]
    config["agent_info"]["prompt_settings"]["output_type"] = OUTPUT_TYPE_FOR.get(
        comparison_type, "SLDP"
    )

    try:
        agent = LlmAgent(**config)
        eval_q = EvalQuestion(**question_dict)
        prompt = generate_prompt(eval_q, agent)

        cxt = AgentContext(agent)
        cxt.initialize_agent(prompt)

        start = time.time()
        success, answer = cxt.run()
        elapsed = time.time() - start

        valid_format, correct = evaluate_answer(
            eval_q.correctness_comparator, answer, eval_q.solution
        )

        return (
            str(answer) if answer else "(no answer)",
            correct,
            cxt.n_tool_calls,
            elapsed,
            None,
        )

    except Exception as e:
        logging.exception("Agent error")
        return None, False, 0, 0.0, str(e)


def print_table(rows, col_widths, headers):
    def sep():
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def fmt_row(cells):
        parts = []
        for cell, w in zip(cells, col_widths):
            lines = textwrap.wrap(str(cell), width=w) or [""]
            parts.append(lines)
        max_lines = max(len(p) for p in parts)
        output = []
        for i in range(max_lines):
            row_parts = []
            for p, w in zip(parts, col_widths):
                line = p[i] if i < len(p) else ""
                row_parts.append(f" {line:<{w}} ")
            output.append("|" + "|".join(row_parts) + "|")
        return "\n".join(output)

    print(sep())
    print(fmt_row(headers))
    print(sep())
    for row in rows:
        print(fmt_row(row))
        print(sep())


def main():
    all_model_names = [m["name"] for m in MODELS]

    parser = argparse.ArgumentParser(
        description="Test a hard question across multiple models via Agent Gateway"
    )
    parser.add_argument(
        "--gateway-url",
        default=AGW_GATEWAY,
        help="Agent Gateway base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=all_model_names,
        metavar="NAME",
        help=f"Models to test (choices: {', '.join(all_model_names)})",
    )
    parser.add_argument(
        "--question-type",
        choices=["qa", "pddl", "both"],
        default="qa",
        help="Which question set to test (default: qa)",
    )
    parser.add_argument(
        "--scene-graph",
        default=None,
        help="DSG filepath to load into Neo4j before testing",
    )
    parser.add_argument(
        "--object-labelspace",
        default="ade20k_mit_label_space.yaml",
        help="Path to object labelspace (default: %(default)s)",
    )
    parser.add_argument(
        "--room-labelspace",
        default="b45_label_space.yaml",
        help="Path to room labelspace (default: %(default)s)",
    )
    parser.add_argument(
        "--db-ip",
        default=None,
        help="Neo4j host (default: $ADT4_HERACLES_IP or localhost)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=None,
        help="Neo4j port (default: $ADT4_HERACLES_PORT or 7687)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    db_ip = args.db_ip or os.getenv("ADT4_HERACLES_IP", "localhost")
    db_port = args.db_port or int(os.getenv("ADT4_HERACLES_PORT", "7687"))
    neo4j_uri = f"neo4j://{db_ip}:{db_port}"

    if args.scene_graph:
        load_scene_graph(
            args.scene_graph, neo4j_uri,
            args.object_labelspace, args.room_labelspace,
        )

    model_index = {m["name"]: m for m in MODELS}
    selected = []
    for name in args.models:
        if name not in model_index:
            parser.error(
                f"Unknown model '{name}'. Choose from: {', '.join(all_model_names)}"
            )
        selected.append(model_index[name])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    qa_path = os.path.join(script_dir, "examples", "questions", "qa_questions.yaml")
    pddl_path = os.path.join(
        script_dir, "examples", "questions", "pddl_questions.yaml"
    )

    qa_questions = load_questions(qa_path)
    pddl_questions = load_questions(pddl_path)

    test_cases = []
    if args.question_type in ("qa", "both"):
        q = find_question(qa_questions, DEFAULT_QUESTIONS["qa"])
        if q:
            test_cases.append(("QA", q))
    if args.question_type in ("pddl", "both"):
        q = find_question(pddl_questions, DEFAULT_QUESTIONS["pddl"])
        if q:
            test_cases.append(("PDDL", q))

    if not test_cases:
        print("No matching questions found.")
        return

    col_widths = [18, 40, 9, 7, 8]
    headers = ["Model", "Answer", "Correct?", "Tools", "Time"]

    for q_type, question in test_cases:
        print(f"\n{'=' * 90}")
        print(f"  [{q_type}] uid={question['uid']}  {question['name']}")
        print(f"  Q: {question['question']}")
        print(f"  Expected: {question['solution']}")
        print(f"{'=' * 90}\n")

        rows = []
        for model_cfg in selected:
            print(f"  Running {model_cfg['name']} ...", flush=True)
            answer, correct, n_tools, elapsed, error = run_agent_for_question(
                model_cfg, question, args.gateway_url, neo4j_uri
            )

            if error:
                print(f"    ERROR: {error[:80]}")
                rows.append(
                    [model_cfg["name"], f"ERROR: {error[:36]}", "--", "--", "--"]
                )
            else:
                mark = "YES" if correct else "no"
                truncated = answer if len(answer) <= 40 else answer[:37] + "..."
                print(
                    f"    Answer: {truncated}  Correct: {mark}  "
                    f"Tools: {n_tools}  Time: {elapsed:.1f}s"
                )
                rows.append([
                    model_cfg["name"],
                    truncated,
                    mark,
                    str(n_tools),
                    f"{elapsed:.1f}s",
                ])

        print()
        print_table(rows, col_widths, headers)
        print()


if __name__ == "__main__":
    main()
