#!/usr/bin/env python3
"""PubMed E-utilities client for paper-distill.

Uses Biopython's Entrez module (esearch + efetch) to query PubMed for
recent papers matching user interests.

CLI usage:
    python search/pubmed.py \
        --interests-file data/interests.jsonl \
        --output data/tmp_pubmed.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.pubmed")

# ---------------------------------------------------------------------------
# Interest helpers
# ---------------------------------------------------------------------------

def _load_latest_interests(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"interests file is empty: {path}")
    return json.loads(lines[-1])


def _extract_keywords(interests: dict[str, Any]) -> list[str]:
    keywords: list[str] = list(interests.get("keywords", []))
    topics = interests.get("topics", {})
    if isinstance(topics, dict):
        for topic_data in topics.values():
            if isinstance(topic_data, dict):
                keywords.extend(topic_data.get("keywords", []))
            elif isinstance(topic_data, list):
                keywords.extend(topic_data)
    return list(dict.fromkeys(keywords))


# ---------------------------------------------------------------------------
# PubMed search
# ---------------------------------------------------------------------------

def _build_query(keywords: list[str], year_start: int = 2021) -> str:
    """Build a PubMed search query ORing keyword phrases."""
    terms = " OR ".join(f'"{kw}"' for kw in keywords)
    return f"({terms}) AND {year_start}:{2026}[dp]"


def _parse_article(article: Any) -> dict[str, Any] | None:
    """Parse a single PubmedArticle XML element into the unified schema."""
    try:
        medline = article.find("MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        art = medline.find("Article")
        if art is None:
            return None

        # Title
        title_el = art.find("ArticleTitle")
        title = title_el.text if title_el is not None else ""

        # Abstract
        abstract_parts: list[str] = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for text_el in abstract_el.findall("AbstractText"):
                label = text_el.get("Label", "")
                text = text_el.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors: list[str] = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author_el in author_list.findall("Author"):
                last = author_el.findtext("LastName", "")
                fore = author_el.findtext("ForeName", "")
                if last:
                    authors.append(f"{fore} {last}".strip())

        # Journal
        journal_el = art.find("Journal")
        journal = ""
        if journal_el is not None:
            journal = journal_el.findtext("Title", "")

        # Year
        year = None
        pub_date = None
        if journal_el is not None:
            jissue = journal_el.find("JournalIssue")
            if jissue is not None:
                pub_date = jissue.find("PubDate")
        if pub_date is not None:
            year_el = pub_date.find("Year")
            if year_el is not None and year_el.text:
                year = int(year_el.text)

        # DOI
        doi = ""
        article_ids = art.findall("ELocationID")
        for eid in article_ids:
            if eid.get("EIdType") == "doi":
                doi = eid.text or ""
                break
        # Also check PubmedData/ArticleIdList
        if not doi:
            pubmed_data = article.find("PubmedData")
            if pubmed_data is not None:
                for aid in pubmed_data.findall("ArticleIdList/ArticleId"):
                    if aid.get("IdType") == "doi":
                        doi = aid.text or ""
                        break

        # MeSH terms
        mesh_terms: list[str] = []
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for heading in mesh_list.findall("MeshHeading"):
                descriptor = heading.find("DescriptorName")
                if descriptor is not None and descriptor.text:
                    mesh_terms.append(descriptor.text)

        return {
            "source": "pubmed",
            "doi": doi,
            "pmid": pmid,
            "arxiv_id": "",
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "citation_count": 0,
            "tldr": "",
            "open_access_url": "",
            "mesh_terms": mesh_terms,
            "categories": [],
            "topic_tags": [],
        }
    except Exception:
        LOG.exception("Error parsing PubMed article")
        return None


def search_pubmed(
    keywords: list[str],
    *,
    year_start: int = 2021,
    max_results: int = 20,
    email: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Run esearch + efetch and return normalised paper dicts."""

    Entrez.email = email or os.getenv("NCBI_EMAIL", os.getenv("OPENALEX_EMAIL", ""))
    ncbi_key = api_key or os.getenv("NCBI_API_KEY", "")
    if ncbi_key:
        Entrez.api_key = ncbi_key

    query = _build_query(keywords, year_start)
    LOG.info("PubMed query: %s", query)

    # esearch
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            usehistory="y",
        )
        search_results = Entrez.read(handle)
        handle.close()
    except Exception:
        LOG.exception("PubMed esearch failed")
        return []

    id_list = search_results.get("IdList", [])
    if not id_list:
        LOG.info("PubMed returned 0 results")
        return []

    LOG.info("PubMed esearch returned %d IDs", len(id_list))

    # efetch in batches
    papers: list[dict[str, Any]] = []
    batch_size = 50
    for start in range(0, len(id_list), batch_size):
        batch = id_list[start : start + batch_size]
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="xml",
                retmode="xml",
            )
            xml_data = handle.read()
            handle.close()
        except Exception:
            LOG.exception("PubMed efetch failed for batch starting at %d", start)
            continue

        # Parse XML
        try:
            root = ElementTree.fromstring(xml_data)
        except ElementTree.ParseError:
            LOG.exception("Failed to parse PubMed XML")
            continue

        for article in root.findall("PubmedArticle"):
            paper = _parse_article(article)
            if paper:
                papers.append(paper)

        # Be polite: small delay between batches
        if start + batch_size < len(id_list):
            time.sleep(0.4)

    LOG.info("Parsed %d papers from PubMed", len(papers))
    return papers[:max_results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search PubMed for papers",
    )
    parser.add_argument(
        "--interests-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "interests.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tmp_pubmed.json",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    interests = _load_latest_interests(args.interests_file)
    keywords = _extract_keywords(interests)
    if not keywords:
        LOG.error("No keywords found in %s", args.interests_file)
        sys.exit(1)

    LOG.info("Keywords (%d): %s", len(keywords), keywords[:10])

    papers = search_pubmed(keywords, max_results=args.max_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote %d papers to %s", len(papers), args.output)


if __name__ == "__main__":
    main()
