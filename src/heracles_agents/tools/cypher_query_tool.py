import json
import urllib.request
import urllib.error
from base64 import b64encode

from heracles.query_interface import Neo4jWrapper

from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool


def _query_http(cypher_string, uri, username, password):
    """Execute a Cypher query via the Neo4j HTTP Transaction API."""
    url = uri.rstrip("/") + "/db/neo4j/tx/commit"
    body = json.dumps({"statements": [{"statement": cypher_string}]}).encode()
    creds = b64encode(f"{username}:{password}".encode()).decode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {creds}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if data.get("errors"):
        return str(data["errors"])
    results = []
    for stmt_result in data.get("results", []):
        columns = stmt_result.get("columns", [])
        for row_data in stmt_result.get("data", []):
            results.append(dict(zip(columns, row_data["row"])))
    return str(results)


def query_db(cypher_string, dsgdb_conf: HeraclesDsgInterface = None):
    if dsgdb_conf is None:
        raise ValueError(
            "query_db called with dsgdb_conf=None. Did you forget to bind the config to the tool?"
        )

    if dsgdb_conf.uri.startswith("http://") or dsgdb_conf.uri.startswith("https://"):
        try:
            return _query_http(
                cypher_string,
                dsgdb_conf.uri,
                dsgdb_conf.username.get_secret_value(),
                dsgdb_conf.password.get_secret_value(),
            )
        except Exception as ex:
            print(ex)
            return str(ex)

    with Neo4jWrapper(
        dsgdb_conf.uri,
        (
            dsgdb_conf.username.get_secret_value(),
            dsgdb_conf.password.get_secret_value(),
        ),
        atomic_queries=True,
        print_profiles=False,
    ) as db:
        if dsgdb_conf.n_object_verification is not None:
            v = db.query("MATCH (n: Object) RETURN COUNT(*) as count")
            count = v[0]["count"]
            assert count == dsgdb_conf.n_object_verification, (
                f"Connected database has {count} objects ({dsgdb_conf.n_object_verification} expected)"
            )
        try:
            query_result = str(db.query(cypher_string))
            return query_result
        except Exception as ex:
            print(ex)
            query_result = str(ex)
            return query_result


# TODO: we need to warp the query_db in another function that takes only the cypher string, and not the dsgdb_conf
# Probably need to have the experiment runner automatically insert the experiment description into the tool call?
cypher_tool = ToolDescription(
    name="run_cypher_query",
    description="An interface for running Cypher queries on a Neo4j database containing a 3D Scene Graph.",
    parameters=[
        FunctionParameter("cypher_string", str, "Your Cypher query"),
    ],
    function=query_db,
)

register_tool(cypher_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
