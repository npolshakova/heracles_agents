#!/usr/bin/env python3
import argparse
import logging
import os
import threading
import time

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import load_dsg_to_db
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Label, RichLog, Rule, Static, TextArea

from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext

logger = logging.getLogger(__name__)


def new_user_message(text):
    return [{"role": "user", "content": text}]


def generate_initial_prompt(agent: LlmAgent):
    prompt = agent.agent_info.prompt_settings.base_prompt
    return prompt


class MyTextArea(TextArea):
    BINDINGS = [
        Binding("ctrl+b", "submit", "Submit text"),
    ]

    def action_submit(self) -> None:
        self.app.action_submit()


class InputDisplayApp(App):
    BINDINGS = [
        Binding("ctrl+x", "quit", "Quit"),
    ]

    def __init__(self, agent):
        self.agent = agent
        self.messages = generate_initial_prompt(agent).to_openai_json(
            "Now you will interact with the user:"
        )
        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield VerticalScroll(
            Label("Agent Chat:"),
            Rule(),
            RichLog(highlight=True, markup=True, wrap=True),
            Rule(line_style="thick"),
            Label("Enter text below:"),
            Rule(),
            MyTextArea("text", id="text_area"),
            Footer(id="footer"),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""

        input_widget = self.query_one("#input_box", Input)
        # Clear the input box
        input_widget.value = ""

        self.display_text = f"You entered: {event.value}"
        self.query_one("#display_panel", Static).update(self.display_text)
        text_log = self.query_one(RichLog)
        text_log.write(event.value)

    def action_submit(self) -> None:
        """Called when ctrl+b is pressed."""
        input_text_box = self.query_one("#text_area", MyTextArea)
        input_text = input_text_box.text.strip()
        if input_text.lower() in ("exit", "/exit", "quit", "/quit"):
            self.exit()
            return
        text_log = self.query_one(RichLog)
        formatted_text = f"[bold black on white]User:[/] {input_text}"
        text_log.write(formatted_text)
        text_log.write("")
        input_text_box.text = ""

        self.messages += new_user_message(input_text)
        initial_length = len(self.messages)

        cxt = AgentContext(self.agent)
        cxt.history = self.messages

        def run_agent():
            history_cursor = initial_length
            max_iter = self.agent.agent_info.max_iterations

            def on_tool_start(n, name, args):
                preview = (args[:80] + "...") if len(args) > 80 else args
                text_log.write(
                    f"[dim]  Tool #{n} [bold]{name}[/bold]({preview}) running...[/]"
                )

            def on_tool_end(n, name, elapsed_ms, result):
                preview = str(result)[:120]
                text_log.write(
                    f"[dim]  Tool #{n} {name} done in {elapsed_ms:.0f} ms → {preview}[/]"
                )

            for i in range(max_iter):
                text_log.write(
                    f"[dim]Step {i + 1}/{max_iter}: calling LLM...[/]"
                )
                t0 = time.perf_counter()
                try:
                    response = cxt.call_llm(cxt.history)
                except Exception as exc:
                    text_log.write(f"[bold red]LLM call failed:[/] {exc}")
                    logger.exception("LLM call failed")
                    return

                llm_ms = (time.perf_counter() - t0) * 1000
                text_log.write(f"[dim]  LLM responded in {llm_ms:.0f} ms[/]")

                try:
                    update = cxt.handle_response(
                        response,
                        on_tool_start=on_tool_start,
                        on_tool_end=on_tool_end,
                    )
                except Exception as exc:
                    text_log.write(f"[bold red]Tool call failed:[/] {exc}")
                    logger.exception("Tool call failed")
                    return

                cxt.update_history(response)
                cxt.update_history(update)

                done = cxt.check_if_done(cxt.history, response, update)

                responses = cxt.get_agent_responses()
                for r in responses[history_cursor:]:
                    if r.parsed_response:
                        text_log.write(r.parsed_response)
                        text_log.write("")
                history_cursor = len(cxt.history)

                if done:
                    text_log.write("[dim]Agent finished.[/]")
                    break
            else:
                text_log.write(
                    f"[dim yellow]Agent hit max iterations ({max_iter}).[/]"
                )

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ChatDSG agent")
    parser.add_argument(
        "--scene-graph",
        nargs="?",
        const=None,
        default=None,
        help="DSG Filepath to load",
    )
    parser.add_argument(
        "--object-labelspace",
        type=str,
        help="Path to object labelspace",
        default="ade20k_mit_label_space.yaml",
    )
    parser.add_argument(
        "--room-labelspace",
        type=str,
        help="Path to room labelspace",
        default="b45_label_space.yaml",
    )
    parser.add_argument("--db_ip", type=str, help="Heracles database ip")
    parser.add_argument("--db_port", type=int, help="Heracles database ip")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging to chatdsg_debug.log",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
            filename="chatdsg_debug.log",
            filemode="w",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
            filename="chatdsg_debug.log",
            filemode="w",
        )

    if args.db_ip is None:
        args.db_ip = os.getenv("ADT4_HERACLES_IP")

    if args.db_port is None:
        args.db_port = os.getenv("ADT4_HERACLES_PORT")

    if args.scene_graph:
        dsg_filepath = args.scene_graph
        logger.info(f"Loading DSG into database from filepath: {dsg_filepath}")

        scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
        summarize_dsg(scene_graph)
        neo4j_uri = f"neo4j://{args.db_ip}:{args.db_port}"
        neo4j_creds = (
            os.getenv("HERACLES_NEO4J_USERNAME"),
            os.getenv("HERACLES_NEO4J_PASSWORD"),
        )

        load_dsg_to_db(
            args.object_labelspace,
            args.room_labelspace,
            neo4j_uri,
            neo4j_creds,
            scene_graph,
        )
        logger.info("DSG loaded!")

    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = InputDisplayApp(agent)
    app.run()
