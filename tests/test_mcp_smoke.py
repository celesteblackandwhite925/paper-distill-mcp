"""Smoke tests for Paper Distill MCP Server.

Run: python tests/test_mcp_smoke.py
  or: .venv/bin/python tests/test_mcp_smoke.py

Tests run without network access or API keys — only local data operations.
"""
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_server.server import mcp

EXPECTED_TOOLS = {
    "search_papers",
    "rank_papers",
    "filter_duplicates",
    "generate_digest",
    "send_push",
    "collect_to_zotero",
    "manage_topics",
    "ingest_research_context",
    "init_session",
    "load_session_context",
    "setup",
    "add_topic",
    "configure",
    "pool_refresh",
    "prepare_review",
    "finalize_review",
    "collect",
    "pool_status",
    "prepare_summarize",
}

EXPECTED_RESOURCES = {
    "paper-distill://topics",
    "paper-distill://history",
    "paper-distill://config",
}


async def test_tools_registered():
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    missing = EXPECTED_TOOLS - tool_names
    assert not missing, f"Missing tools: {missing}"
    print(f"  [PASS] {len(tools)} tools registered")


async def test_resources_registered():
    resources = await mcp.list_resources()
    resource_uris = {str(r.uri) for r in resources}
    missing = EXPECTED_RESOURCES - resource_uris
    assert not missing, f"Missing resources: {missing}"
    print(f"  [PASS] {len(resources)} resources registered")


async def test_manage_topics_list():
    result = await mcp.call_tool("manage_topics", {"action": "list"})
    # Result has .content list with TextContent
    text = result.content[0].text
    data = json.loads(text)
    assert "topics" in data, "No 'topics' key in response"
    assert len(data["topics"]) > 0, "No topics found"
    print(f"  [PASS] manage_topics(list) returned {len(data['topics'])} topics")


async def test_filter_duplicates():
    fake_papers = [
        {"doi": "10.1234/fake-never-pushed", "title": "Fake Paper"},
        {"doi": "", "title": "No DOI Paper"},
    ]
    result = await mcp.call_tool("filter_duplicates", {"papers": fake_papers})
    text = result.content[0].text
    filtered = json.loads(text)
    # Both should pass since fake DOI won't be in history
    assert len(filtered) == 2, f"Expected 2 papers, got {len(filtered)}"
    print(f"  [PASS] filter_duplicates returned {len(filtered)} papers")


async def test_rank_papers():
    fake_papers = [
        {
            "title": "Example keyword1 keyword2 research paper",
            "abstract": "A study about keyword1 keyword2 keyword3",
            "doi": "10.1234/test1",
            "year": 2025,
            "citation_count": 10,
            "published_date": "2025-01-01",
            "topic_tags": ["example-topic"],
        },
        {
            "title": "Unrelated paper about something else",
            "abstract": "No matching keywords here",
            "doi": "10.1234/test2",
            "year": 2024,
            "citation_count": 5,
            "published_date": "2024-06-01",
            "topic_tags": ["other"],
        },
    ]
    result = await mcp.call_tool("rank_papers", {"papers": fake_papers, "top_n": 2})
    text = result.content[0].text
    ranked = json.loads(text)
    assert len(ranked) == 2, f"Expected 2, got {len(ranked)}"
    assert "_scores" in ranked[0], "No _scores in ranked paper"
    # Paper matching topic keywords should rank higher
    assert ranked[0]["doi"] == "10.1234/test1", "Matching paper should rank first"
    print(f"  [PASS] rank_papers: top score={ranked[0]['_scores']['total']:.4f}")


def _read_resource_text(result) -> str:
    """Extract text from resource result (handles FastMCP 3.x format)."""
    if hasattr(result, 'contents'):
        return result.contents[0].content
    if hasattr(result, 'content'):
        c = result.content[0]
        return c.text if hasattr(c, 'text') else c.content
    return str(result)


async def test_resource_topics():
    result = await mcp.read_resource("paper-distill://topics")
    text = _read_resource_text(result)
    assert "topics" in text, "topics resource missing 'topics' key"
    print(f"  [PASS] paper-distill://topics readable")


async def test_resource_config():
    result = await mcp.read_resource("paper-distill://config")
    text = _read_resource_text(result)
    data = json.loads(text)
    # API keys should NOT appear
    assert "telegram_bot_token" not in text, "API key leaked in config resource!"
    assert "telegram_configured" in data, "Missing telegram_configured field"
    print(f"  [PASS] paper-distill://config readable, keys redacted")


async def test_init_session():
    result = await mcp.call_tool("init_session", {"load_context": "no"})
    text = result.content[0].text
    data = json.loads(text)
    assert "session_id" in data, "Missing session_id"
    assert data["session_id"].startswith("session-"), "Bad session_id format"
    assert "available_platforms" in data, "Missing available_platforms"
    assert "context" in data, "Missing context info"
    print(f"  [PASS] init_session: session={data['session_id']}, platforms={list(data['available_platforms'].keys())}")


async def test_load_session_context():
    result = await mcp.call_tool("load_session_context", {})
    text = result.content[0].text
    data = json.loads(text)
    assert "keywords" in data, "Missing keywords"
    assert "entry_count" in data, "Missing entry_count"
    print(f"  [PASS] load_session_context: {data['entry_count']} entries")


async def main():
    tests = [
        ("Tools registered", test_tools_registered),
        ("Resources registered", test_resources_registered),
        ("manage_topics(list)", test_manage_topics_list),
        ("filter_duplicates", test_filter_duplicates),
        ("rank_papers", test_rank_papers),
        ("init_session", test_init_session),
        ("load_session_context", test_load_session_context),
        ("Resource: topics", test_resource_topics),
        ("Resource: config", test_resource_config),
    ]

    print(f"Running {len(tests)} smoke tests...\n")
    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed:
        sys.exit(1)
    print("All smoke tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
