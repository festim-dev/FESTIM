import json
import os
import re
import time
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

CACHE_MAX_AGE_DAYS = 30

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

TYPE_ICONS = {
    "article": "fa-solid fa-file-lines",
    "thesis": "fa-solid fa-graduation-cap",
    "report": "fa-solid fa-file-contract",
    "preprint": "fa-solid fa-file-pen",
    "book": "fa-solid fa-book",
    "conference": "fa-solid fa-users",
}

TYPE_LABELS = {
    "article": "Article",
    "thesis": "Thesis",
    "report": "Report",
    "preprint": "Preprint",
    "book": "Book",
    "conference": "Conference",
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


def is_cache_stale(record):
    cached_at = record.get("_cached_at", 0)
    age_days = (time.time() - cached_at) / 86400
    return age_days > CACHE_MAX_AGE_DAYS


def truncate_authors(authors, max_count=3):
    if len(authors) <= max_count:
        return ", ".join(authors)
    return ", ".join(authors[:max_count]) + " et al."


def resolve_type(work):
    raw_type = (work.get("type", "") or "").lower().strip()
    if raw_type in OPENALEX_TYPE_MAP:
        return OPENALEX_TYPE_MAP[raw_type]
    crossref_type = (work.get("type_crossref", "") or "").lower().strip()
    if crossref_type in OPENALEX_TYPE_MAP:
        return OPENALEX_TYPE_MAP[crossref_type]
    return "article"


def extract_record_from_work(work, entry_type=None, degree=None):
    authors = []
    for authorship in work.get("authorships", []):
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    source = work.get("primary_location", {}) or {}
    source_obj = source.get("source", {}) or {}
    journal = source_obj.get("display_name", "")

    detected_type = resolve_type(work)
    final_type = entry_type if entry_type else detected_type

    if final_type == "thesis":
        # Use provided degree, or default to generic "Thesis"
        degree_label = degree if degree else "Thesis"

        institutions = set()
        for authorship in work.get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("display_name", "")
                if name:
                    institutions.add(name)
        if institutions:
            journal = f"{degree_label}, {', '.join(institutions)}"
        elif journal:
            journal = f"{degree_label}, {journal}"
        else:
            journal = degree_label

    oa = work.get("open_access", {}) or {}
    oa_url = oa.get("oa_url", "")

    topics = []
    for topic in work.get("topics", [])[:3]:
        name = topic.get("display_name", "")
        if name:
            topics.append(name)

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


def fetch_work(doi, cache, entry_type=None, degree=None):
    if doi in cache and not is_cache_stale(cache[doi]):
        logger.info(f"[publications] Cache hit: {doi}")
        record = cache[doi]
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

    record = extract_record_from_work(work, entry_type, degree)
    record["doi"] = doi
    record["_cached_at"] = time.time()
    cache[doi] = record
    return record


def fetch_work_by_title(title, cache, entry_type=None, degree=None):
    cache_key = f"title:{title}"

    if cache_key in cache and not is_cache_stale(cache[cache_key]):
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

    work = works[0]
    record = extract_record_from_work(work, entry_type, degree)
    record["_cached_at"] = time.time()
    cache[cache_key] = record
    return record


def fetch_work_by_id(openalex_id, cache, entry_type=None, degree=None):
    cache_key = f"openalex:{openalex_id}"

    if cache_key in cache and not is_cache_stale(cache[cache_key]):
        logger.info(f"[publications] Cache hit: {openalex_id}")
        record = cache[cache_key]
        if entry_type:
            record["type"] = entry_type
        return record

    logger.info(f"[publications] Fetching from OpenAlex by ID: {openalex_id}")

    try:
        work = Works()[openalex_id]

        if work is None:
            raise ExtensionError(
                f"[publications] OpenAlex returned no result for ID: {openalex_id}\n"
                f"If this work exists but cannot be fetched, add it as a manual entry "
                f"with 'authors' field in publications.yml"
            )

        record = extract_record_from_work(work, entry_type, degree)
        record["_cached_at"] = time.time()
        cache[cache_key] = record
        return record

    except ExtensionError:
        raise
    except Exception as e:
        raise ExtensionError(
            f"[publications] Error fetching OpenAlex ID {openalex_id}: {e}"
        )


def parse_manual_entry(entry):
    entry_type = entry.get("type", "article")

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


def build_stats_html(records):
    """Build the stats bar at the top of the page."""
    total = len(records)
    total_citations = sum(r.get("cited_by", 0) for r in records)

    # Count per type
    type_counts = {}
    for r in records:
        t = r.get("type", "article")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Year range
    years = [r.get("year", 0) for r in records if r.get("year", 0) > 0]
    year_range = f"{min(years)}-{max(years)}" if years else ""

    # Build type pills
    type_pills = ""
    for t in ["article", "thesis", "conference", "preprint", "book", "report"]:
        count = type_counts.get(t, 0)
        if count > 0:
            icon = TYPE_ICONS.get(t, "fa-solid fa-file-lines")
            label = TYPE_LABELS.get(t, "Article")
            type_pills += (
                f'<button class="festim-pub-filter-btn" '
                f'data-filter="{t}" title="{label}">'
                f'<i class="{icon}"></i> '
                f"{count} {label}{'s' if count != 1 else ''}"
                f"</button>"
            )

    return f"""
    <div class="festim-pub-stats">
      <div class="festim-pub-stats-numbers">
        <div class="festim-pub-stat">
          <span class="festim-pub-stat-value">{total}</span>
          <span class="festim-pub-stat-label">Publications</span>
        </div>
        <div class="festim-pub-stat">
          <span class="festim-pub-stat-value">{total_citations:,}</span>
          <span class="festim-pub-stat-label">Total citations</span>
        </div>
        <div class="festim-pub-stat">
          <span class="festim-pub-stat-value">{year_range}</span>
          <span class="festim-pub-stat-label">Year range</span>
        </div>
      </div>

      <div class="festim-pub-toolbar">
        <div class="festim-pub-search-wrapper">
          <i class="fa-solid fa-magnifying-glass"></i>
          <input type="text" class="festim-pub-search"
                 id="festimPubSearch"
                 placeholder="Search by title, author, journal...">
        </div>
        <div class="festim-pub-filters">
          <button class="festim-pub-filter-btn active" data-filter="all">
            <i class="fa-solid fa-list"></i> All
          </button>
          {type_pills}
        </div>
      </div>
    </div>"""


def build_card_html(record):
    title = record.get("title", "Unknown title")
    all_authors = record.get("authors", [])
    authors = truncate_authors(all_authors)
    journal = record.get("journal", "")
    year = record.get("year", "")
    doi = record.get("doi", "")
    cited_by = record.get("cited_by", 0)
    oa_url = record.get("oa_url", "")
    topics = record.get("topics", [])
    entry_type = record.get("type", "article")

    icon_class = TYPE_ICONS.get(entry_type, "fa-solid fa-file-lines")
    type_label = TYPE_LABELS.get(entry_type, "Article")

    type_badge = (
        f'<span class="festim-pub-type festim-pub-type-{entry_type}">'
        f'<i class="{icon_class}"></i> {type_label}'
        f"</span>"
    )

    doi_badge = ""
    if doi:
        doi_badge = (
            f'<a class="festim-pub-doi" href="https://doi.org/{doi}" target="_blank">'
            f'<span class="festim-pub-doi-label">DOI</span>'
            f'<span class="festim-pub-doi-value">{doi}</span>'
            f"</a>"
        )

    link_html = ""
    if not doi and oa_url:
        link_html = (
            f'<a class="festim-pub-pdf" href="{oa_url}" target="_blank">'
            f'<i class="fa-solid fa-arrow-up-right-from-square"></i> View'
            f"</a>"
        )

    pdf_html = ""
    if doi and oa_url:
        pdf_html = (
            f'<a class="festim-pub-pdf" href="{oa_url}" target="_blank">'
            f'<i class="fa-solid fa-file-pdf"></i> PDF'
            f"</a>"
        )

    cite_html = ""
    if cited_by > 0:
        cite_html = (
            f'<span class="festim-pub-cited">'
            f'<i class="fa-solid fa-quote-left"></i> Cited {cited_by} times'
            f"</span>"
        )

    tags_html = ""
    if topics:
        tags = "".join(f'<span class="festim-pub-tag">{t}</span>' for t in topics)
        tags_html = f'<div class="festim-pub-tags">{tags}</div>'

    journal_line = ""
    if journal:
        journal_line = f"{journal}, {year}" if year else journal
    elif year:
        journal_line = str(year)

    # Build searchable data attributes
    search_text = " ".join(
        [
            title,
            " ".join(all_authors),
            journal,
            str(year),
            " ".join(topics),
        ]
    ).lower()

    return f"""
    <div class="festim-pub-card festim-pub-card-{entry_type}"
         data-type="{entry_type}"
         data-year="{year}"
         data-search="{search_text}">
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


def build_filter_script():
    return """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
      var searchInput = document.getElementById("festimPubSearch");
      var filterBtns = document.querySelectorAll(".festim-pub-filter-btn");
      var cards = document.querySelectorAll(".festim-pub-card");
      var yearGroups = document.querySelectorAll(".festim-pub-year-group");
      var noResults = document.getElementById("festimPubNoResults");

      var activeFilter = "all";

      function applyFilters() {
        var query = searchInput.value.toLowerCase().trim();
        var visibleCount = 0;

        cards.forEach(function(card) {
          var type = card.getAttribute("data-type");
          var searchText = card.getAttribute("data-search");

          var matchesFilter = (activeFilter === "all" || type === activeFilter);
          var matchesSearch = (!query || searchText.indexOf(query) !== -1);
          var visible = matchesFilter && matchesSearch;

          card.style.display = visible ? "" : "none";
          if (visible) visibleCount++;
        });

        yearGroups.forEach(function(group) {
          var visibleCards = group.querySelectorAll(
            ".festim-pub-card:not([style*='display: none'])"
          );
          group.style.display = visibleCards.length > 0 ? "" : "none";
        });

        if (noResults) {
          noResults.style.display = visibleCount === 0 ? "" : "none";
        }
      }

      filterBtns.forEach(function(btn) {
        btn.addEventListener("click", function() {
          filterBtns.forEach(function(b) { b.classList.remove("active"); });
          btn.classList.add("active");
          activeFilter = btn.getAttribute("data-filter");
          applyFilters();
        });
      });

      searchInput.addEventListener("input", applyFilters);
    });
    </script>"""


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
                record = parse_manual_entry(entry)
                records.append(record)

            elif "openalex_id" in entry:
                entry_type = entry.get("type", None)
                degree = entry.get("degree", None)
                record = fetch_work_by_id(
                    entry["openalex_id"], cache, entry_type, degree
                )
                records.append(record)

            elif "doi" in entry:
                entry_type = entry.get("type", None)
                degree = entry.get("degree", None)
                record = fetch_work(entry["doi"], cache, entry_type, degree)
                records.append(record)

            elif "title" in entry:
                entry_type = entry.get("type", None)
                degree = entry.get("degree", None)
                record = fetch_work_by_title(entry["title"], cache, entry_type, degree)
                records.append(record)

            else:
                raise ExtensionError(
                    f"[publications] Invalid entry in publications.yml: {entry}\n"
                    f"Each entry must have at least one of: 'doi', 'title', "
                    f"'openalex_id', or 'authors' (for manual entries)."
                )

        save_cache(cache)

        records.sort(key=lambda r: (-r.get("year", 0), r.get("title", "")))

        years = {}
        for r in records:
            y = r.get("year", 0)
            years.setdefault(y, []).append(r)

        # Build HTML
        html = '<div class="festim-pub-container">'
        html += build_stats_html(records)

        # No results message (hidden by default)
        html += """
        <div class="festim-pub-no-results" id="festimPubNoResults"
             style="display: none;">
          <i class="fa-solid fa-magnifying-glass"></i>
          <p>No publications match your search.</p>
        </div>
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
        html += build_filter_script()

        raw_node = nodes.raw("", html, format="html")
        return [raw_node]


def setup(app):
    app.add_directive("publications", PublicationsDirective)
    return {"version": "0.3", "parallel_read_safe": True}
