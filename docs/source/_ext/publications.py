import json
import os
import re
from pathlib import Path

import yaml
from docutils import nodes
from docutils.parsers.rst import Directive
from pyalex import Works
from sphinx.errors import ExtensionError
from sphinx.util import logging

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent.parent / "_data" / "publications_cache.json"
DATA_PATH = Path(__file__).parent.parent / "_data" / "publications.yml"

# Mapping from OpenAlex type to our display types
OPENALEX_TYPE_MAP = {
    "article": "article",
    "journal-article": "article",
    "book": "book",
    "book-chapter": "book",
    "dissertation": "thesis",
    "thesis": "thesis",
    "preprint": "preprint",
    "posted-content": "preprint",
    "proceedings-article": "conference",
    "conference-paper": "conference",
    "report": "report",
    "review": "article",
    "editorial": "article",
    "letter": "article",
    "erratum": "article",
    "paratext": "article",
    "other": "article",
}


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def truncate_authors(authors, max_count=3):
    if len(authors) <= max_count:
        return ", ".join(authors)
    return ", ".join(authors[:max_count]) + " et al."


def resolve_type(work):
    """Determine publication type from OpenAlex metadata."""
    raw_type = work.get("type", "") or ""
    raw_type = raw_type.lower().strip()

    # Check the mapping
    if raw_type in OPENALEX_TYPE_MAP:
        return OPENALEX_TYPE_MAP[raw_type]

    # Fallback: check type_crossref
    crossref_type = work.get("type_crossref", "") or ""
    crossref_type = crossref_type.lower().strip()

    if crossref_type in OPENALEX_TYPE_MAP:
        return OPENALEX_TYPE_MAP[crossref_type]

    return "article"


def extract_record_from_work(work, entry_type=None):
    """Extract a standardised record from an OpenAlex work object."""
    # Authors
    authors = []
    for authorship in work.get("authorships", []):
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    # Source / journal
    source = work.get("primary_location", {}) or {}
    source_obj = source.get("source", {}) or {}
    journal = source_obj.get("display_name", "")

    # Auto-detect type if not overridden
    detected_type = resolve_type(work)
    final_type = entry_type if entry_type else detected_type

    # For theses, build a better journal line
    if final_type == "thesis":
        institutions = set()
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("display_name", "")
                if name:
                    institutions.add(name)
        if institutions:
            journal = f"PhD thesis, {', '.join(institutions)}"
        elif journal:
            journal = f"PhD thesis, {journal}"
        else:
            journal = "PhD thesis"

    # Open access
    oa = work.get("open_access", {}) or {}
    oa_url = oa.get("oa_url", "")

    # Topics
    topics = []
    for topic in work.get("topics", [])[:3]:
        name = topic.get("display_name", "")
        if name:
            topics.append(name)

    # DOI
    doi = work.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]

    year = work.get("publication_year", 0)
    title = work.get("title", "Unknown title")
    cited_by = work.get("cited_by_count", 0)

    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "cited_by": cited_by,
        "oa_url": oa_url,
        "topics": topics,
        "type": final_type,
    }


def fetch_work(doi, cache, entry_type=None):
    """Fetch a single work from OpenAlex by DOI, using cache if available."""
    if doi in cache:
        logger.info(f"[publications] Cache hit: {doi}")
        record = cache[doi]
        # Allow type override even from cache
        if entry_type:
            record["type"] = entry_type
        return record

    logger.info(f"[publications] Fetching from OpenAlex: {doi}")

    url = f"https://doi.org/{doi}"
    work = Works()[url]

    if work is None:
        raise ExtensionError(
            f"[publications] OpenAlex returned no result for DOI: {doi}\n"
            f"If this DOI is correct but not indexed, add it as a manual entry "
            f"with 'authors' field in publications.yml"
        )

    record = extract_record_from_work(work, entry_type)
    record["doi"] = doi  # ensure the DOI from YAML is used
    cache[doi] = record
    return record


def fetch_work_by_title(title, cache, entry_type=None):
    """Fetch a work from OpenAlex by title search, using cache if available."""
    cache_key = f"title:{title}"

    if cache_key in cache:
        logger.info(f"[publications] Cache hit: {title[:60]}...")
        record = cache[cache_key]
        if entry_type:
            record["type"] = entry_type
        return record

    logger.info(f"[publications] Searching OpenAlex for: {title[:60]}...")

    works = Works().search(title).get()

    if not works:
        raise ExtensionError(
            f"[publications] OpenAlex returned no results for title: {title}\n"
            f"If this work exists but is not indexed, add it as a manual entry "
            f"with 'authors' field in publications.yml"
        )

    # Take the first (most relevant) result
    work = works[0]

    record = extract_record_from_work(work, entry_type)
    cache[cache_key] = record
    return record


def fetch_work_by_id(openalex_id, cache, entry_type=None):
    """Fetch a work from OpenAlex by its work ID, using cache if available."""
    cache_key = f"openalex:{openalex_id}"

    if cache_key in cache:
        logger.info(f"[publications] Cache hit: {openalex_id}")
        record = cache[cache_key]
        if entry_type:
            record["type"] = entry_type
        return record

    logger.info(f"[publications] Fetching from OpenAlex by ID: {openalex_id}")

    try:
        work = Works()[f"https://openalex.org/works/{openalex_id}"]

        if work is None:
            raise ExtensionError(
                f"[publications] OpenAlex returned no result for ID: {openalex_id}\n"
                f"If this work exists but cannot be fetched, add it as a manual entry "
                f"with 'authors' field in publications.yml"
            )

        record = extract_record_from_work(work, entry_type)
        cache[cache_key] = record
        return record

    except ExtensionError:
        raise
    except Exception as e:
        raise ExtensionError(
            f"[publications] Error fetching OpenAlex ID {openalex_id}: {e}"
        )


def parse_manual_entry(entry):
    """Parse a manually defined entry (thesis, report, etc.)."""
    entry_type = entry.get("type", "article")

    # Build journal line
    if entry_type == "thesis":
        institution = entry.get("institution", "")
        degree = entry.get("degree", "PhD thesis")
        journal = f"{degree}, {institution}" if institution else degree
    else:
        journal = entry.get("journal", "")

    return {
        "title": entry.get("title", "Unknown title"),
        "authors": entry.get("authors", []),
        "year": entry.get("year", 0),
        "type": entry_type,
        "doi": entry.get("doi", ""),
        "cited_by": entry.get("cited_by", 0),
        "oa_url": entry.get("url", ""),
        "topics": entry.get("topics", []),
        "journal": journal,
    }


def build_card_html(record):
    """Build the HTML for a single publication card."""
    title = record.get("title", "Unknown title")
    authors = truncate_authors(record.get("authors", []))
    journal = record.get("journal", "")
    year = record.get("year", "")
    doi = record.get("doi", "")
    cited_by = record.get("cited_by", 0)
    oa_url = record.get("oa_url", "")
    topics = record.get("topics", [])
    entry_type = record.get("type", "article")

    # Type icon
    type_icons = {
        "article": "fa-solid fa-file-lines",
        "thesis": "fa-solid fa-graduation-cap",
        "report": "fa-solid fa-file-contract",
        "preprint": "fa-solid fa-file-pen",
        "book": "fa-solid fa-book",
        "conference": "fa-solid fa-users",
    }
    icon_class = type_icons.get(entry_type, "fa-solid fa-file-lines")

    # Type label
    type_labels = {
        "article": "Article",
        "thesis": "Thesis",
        "report": "Report",
        "preprint": "Preprint",
        "book": "Book",
        "conference": "Conference",
    }
    type_label = type_labels.get(entry_type, "Article")

    # Type badge
    type_badge = (
        f'<span class="festim-pub-type festim-pub-type-{entry_type}">'
        f'<i class="{icon_class}"></i> {type_label}'
        f"</span>"
    )

    # DOI badge
    doi_badge = ""
    if doi:
        doi_badge = (
            f'<a class="festim-pub-doi" href="https://doi.org/{doi}" target="_blank">'
            f'<span class="festim-pub-doi-label">DOI</span>'
            f'<span class="festim-pub-doi-value">{doi}</span>'
            f"</a>"
        )

    # Link (for works without DOI)
    link_html = ""
    if not doi and oa_url:
        link_html = (
            f'<a class="festim-pub-pdf" href="{oa_url}" target="_blank">'
            f'<i class="fa-solid fa-arrow-up-right-from-square"></i> View'
            f"</a>"
        )

    # OA PDF link (only show separately if there is also a DOI)
    pdf_html = ""
    if doi and oa_url:
        pdf_html = (
            f'<a class="festim-pub-pdf" href="{oa_url}" target="_blank">'
            f'<i class="fa-solid fa-file-pdf"></i> PDF'
            f"</a>"
        )

    # Citation count
    cite_html = ""
    if cited_by > 0:
        cite_html = (
            f'<span class="festim-pub-cited">'
            f'<i class="fa-solid fa-quote-left"></i> Cited {cited_by} times'
            f"</span>"
        )

    # Topic tags
    tags_html = ""
    if topics:
        tags = "".join(f'<span class="festim-pub-tag">{t}</span>' for t in topics)
        tags_html = f'<div class="festim-pub-tags">{tags}</div>'

    # Journal + year line
    journal_line = ""
    if journal:
        journal_line = f"{journal}, {year}" if year else journal
    elif year:
        journal_line = str(year)

    return f"""
    <div class="festim-pub-card festim-pub-card-{entry_type}">
      <div class="festim-pub-content">
        <div class="festim-pub-header">
          {type_badge}
        </div>
        <h3 class="festim-pub-title">{title}</h3>
        <p class="festim-pub-authors">{authors}</p>
        <p class="festim-pub-journal">{journal_line}</p>
        <div class="festim-pub-links">
          {doi_badge}
          {link_html}
          {pdf_html}
          {cite_html}
        </div>
        {tags_html}
      </div>
    </div>"""


class PublicationsDirective(Directive):
    has_content = False
    optional_arguments = 0

    def run(self):
        if not DATA_PATH.exists():
            raise ExtensionError(f"[publications] Data file not found: {DATA_PATH}")

        with open(DATA_PATH, "r") as f:
            entries = yaml.safe_load(f) or []

        cache = load_cache()
        records = []

        for entry in entries:
            if "authors" in entry:
                # Manual entry
                record = parse_manual_entry(entry)
                records.append(record)

            elif "openalex_id" in entry:
                # Fetch by OpenAlex work ID
                entry_type = entry.get("type", None)
                record = fetch_work_by_id(entry["openalex_id"], cache, entry_type)
                records.append(record)

            elif "doi" in entry:
                # Auto-fetch by DOI
                entry_type = entry.get("type", None)
                record = fetch_work(entry["doi"], cache, entry_type)
                records.append(record)

            elif "title" in entry:
                # Auto-fetch by title
                entry_type = entry.get("type", None)
                record = fetch_work_by_title(entry["title"], cache, entry_type)
                records.append(record)

            else:
                raise ExtensionError(
                    f"[publications] Invalid entry in publications.yml: {entry}\n"
                    f"Each entry must have at least one of: 'doi', 'title', "
                    f"'openalex_id', or 'authors' (for manual entries)."
                )

        save_cache(cache)

        # Sort by year descending, then by title
        records.sort(key=lambda r: (-r.get("year", 0), r.get("title", "")))

        # Group by year
        years = {}
        for r in records:
            y = r.get("year", 0)
            years.setdefault(y, []).append(r)

        # Build HTML
        count = len(records)
        html = f"""
            <div class="festim-pub-container">
            <p class="festim-pub-count">
                <strong>{count}</strong> publication{"s" if count != 1 else ""} and counting
            </p>
            """

        for year in sorted(years.keys(), reverse=True):
            cards = "".join(build_card_html(r) for r in years[year])
            html += f"""
            <div class="festim-pub-year-group">
                <div class="festim-pub-year-marker">
                <span>{year}</span>
                </div>
                <div class="festim-pub-year-cards">
                {cards}
                </div>
            </div>"""

        html += "</div>"

        raw_node = nodes.raw("", html, format="html")
        return [raw_node]


def setup(app):
    app.add_directive("publications", PublicationsDirective)
    return {"version": "0.2", "parallel_read_safe": True}
