#!/usr/bin/env python3
"""Unpaywall DOI lookup for paper-distill.

Given a list of papers with DOIs, queries https://api.unpaywall.org/v2/{doi}
to find legal open-access PDF links.

This is NOT a search source — it's a post-merge enrichment step that fills
in open_access_url for papers that have a DOI but no free link yet.

No API key needed, just an email address (OPENALEX_EMAIL or UNPAYWALL_EMAIL).

CLI usage:
    python search/unpaywall_lookup.py --doi "10.1234/example"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

LOG = logging.getLogger("search.unpaywall")

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"


async def _lookup_one(
    client: httpx.AsyncClient,
    doi: str,
    email: str,
) -> str:
    """Look up a single DOI and return the best OA PDF URL, or ''."""
    try:
        resp = await client.get(
            f"{UNPAYWALL_BASE}/{doi}",
            params={"email": email},
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        best = data.get("best_oa_location") or {}
        return best.get("url_for_pdf") or best.get("url") or ""
    except (httpx.RequestError, Exception) as exc:
        LOG.debug("Unpaywall lookup failed for %s: %s", doi, exc)
        return ""


async def enrich_open_access(
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For papers with a DOI but no open_access_url, try Unpaywall."""
    email = os.getenv("UNPAYWALL_EMAIL") or os.getenv("OPENALEX_EMAIL", "")
    if not email:
        LOG.warning("No email configured for Unpaywall (set UNPAYWALL_EMAIL or OPENALEX_EMAIL)")
        return papers

    # Find papers that need enrichment
    to_enrich = [
        (i, p["doi"])
        for i, p in enumerate(papers)
        if p.get("doi") and not p.get("open_access_url")
    ]

    if not to_enrich:
        return papers

    LOG.info("Unpaywall: looking up %d DOIs", len(to_enrich))

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Batch with concurrency limit to respect rate limits
        sem = asyncio.Semaphore(10)

        async def _limited_lookup(doi: str) -> str:
            async with sem:
                return await _lookup_one(client, doi, email)

        tasks = [_limited_lookup(doi) for _, doi in to_enrich]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    found = 0
    for (idx, doi), result in zip(to_enrich, results):
        if isinstance(result, str) and result:
            papers[idx]["open_access_url"] = result
            found += 1

    LOG.info("Unpaywall: found OA links for %d / %d papers", found, len(to_enrich))
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpaywall DOI lookup")
    parser.add_argument("--doi", required=True, help="DOI to look up")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    email = os.getenv("UNPAYWALL_EMAIL") or os.getenv("OPENALEX_EMAIL", "user@example.com")

    async def _run():
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = await _lookup_one(client, args.doi, email)
            print(json.dumps({"doi": args.doi, "open_access_url": url}, indent=2))

    asyncio.run(_run())


if __name__ == "__main__":
    main()
