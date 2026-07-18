#!/usr/bin/env python3
"""
Build a text training corpus from an INSPIRE-HEP author's first-author papers.

Pipeline:
  1. Fetch the author record to get their BAI (e.g. D.L.Wiltshire.1) and name.
  2. Search INSPIRE literature for everything claimed by that author.
  3. Keep only records where they are FIRST author (matched by author recid).
  4. For each record with an arXiv ID, download the LaTeX source from arXiv
     (falls back to PDF text extraction if the submission is PDF-only).
  5. Concatenate everything into corpus.txt (+ a manifest.json of what went in).

Usage:
    pip install requests pylatexenc pymupdf
    python build_corpus.py --author 983425 --out corpus_dir

Options:
    --plain-text   convert LaTeX to plain text via pylatexenc (default: keep raw LaTeX)
    --include-nonfirst  keep all claimed papers, not just first-author ones

Rate limits respected: INSPIRE allows 15 req/5s; arXiv asks ~1 req/3s.
"""

import argparse
import gzip
import io
import json
import re
import tarfile
import time
from pathlib import Path

import requests

INSPIRE_API = "https://inspirehep.net/api"
HEADERS = {"User-Agent": "corpus-builder/0.1 (personal research use)"}

# document types worth keeping for a prose corpus
KEEP_TYPES = {"article", "conference paper", "review", "book chapter", "note"}


def get_author(recid: str) -> dict:
    r = requests.get(f"{INSPIRE_API}/authors/{recid}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    meta = r.json()["metadata"]
    bai = next(
        (i["value"] for i in meta.get("ids", []) if i.get("schema") == "INSPIRE BAI"),
        None,
    )
    name = meta.get("name", {}).get("value", "unknown")
    if bai is None:
        raise SystemExit(f"No BAI found on author record {recid}")
    return {"recid": recid, "bai": bai, "name": name}


def search_literature(author: dict) -> list[dict]:
    """All literature claimed by the author, paginated."""
    fields = ",".join(
        [
            "titles",
            "authors.full_name",
            "authors.record",
            "arxiv_eprints",
            "document_type",
            "earliest_date",
            "abstracts",
        ]
    )
    url = f"{INSPIRE_API}/literature"
    params = {"q": f"a {author['bai']}", "fields": fields, "size": 100, "page": 1}
    records = []
    while True:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        hits = data["hits"]["hits"]
        records.extend(hits)
        next_link = data.get("links", {}).get("next")
        if not next_link or not hits:
            break
        url, params = next_link, None  # 'next' link already encodes everything
        time.sleep(0.5)  # stay well under 15 req / 5 s
    return records


def is_first_author(record: dict, author_recid: str) -> bool:
    authors = record["metadata"].get("authors", [])
    if not authors:
        return False
    ref = authors[0].get("record", {}).get("$ref", "")
    return ref.rstrip("/").endswith(f"/{author_recid}")


def wanted_type(record: dict) -> bool:
    types = set(record["metadata"].get("document_type", []))
    return bool(types & KEEP_TYPES) or not types


def arxiv_id(record: dict) -> str | None:
    eprints = record["metadata"].get("arxiv_eprints", [])
    return eprints[0]["value"] if eprints else None


# ---------------------------------------------------------------- arXiv fetch


def fetch_arxiv_source(aid: str, cache: Path) -> tuple[str, str] | None:
    """
    Returns (kind, text) where kind is 'latex' or 'pdf', or None on failure.
    Caches raw downloads so re-runs don't hammer arXiv.
    """
    cache.mkdir(parents=True, exist_ok=True)
    blob_path = cache / f"{aid.replace('/', '_')}.blob"

    if blob_path.exists():
        blob = blob_path.read_bytes()
    else:
        r = requests.get(
            f"https://arxiv.org/e-print/{aid}", headers=HEADERS, timeout=60
        )
        if r.status_code != 200:
            return None
        blob = r.content
        blob_path.write_bytes(blob)
        time.sleep(3)  # arXiv rate-limit etiquette

    # PDF-only submission?
    if blob[:4] == b"%PDF":
        text = extract_pdf_text(blob)
        return ("pdf", text) if text else None

    # tar.gz of source files?
    buf = io.BytesIO(blob)
    try:
        with tarfile.open(fileobj=buf, mode="r:*") as tar:
            texts = []
            for member in tar.getmembers():
                if member.name.endswith(".tex"):
                    f = tar.extractfile(member)
                    if f:
                        texts.append(f.read().decode("utf-8", errors="replace"))
            if texts:
                # put the file containing \documentclass first
                texts.sort(key=lambda t: 0 if "\\documentclass" in t else 1)
                return ("latex", "\n\n".join(texts))
    except tarfile.TarError:
        pass

    # single gzipped .tex file?
    try:
        text = gzip.decompress(blob).decode("utf-8", errors="replace")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return ("latex", text)
    except (OSError, gzip.BadGzipFile):
        pass

    return None


def fetch_arxiv_pdf(aid: str, cache: Path) -> str | None:
    pdf_path = cache / f"{aid.replace('/', '_')}.pdf"
    if pdf_path.exists():
        blob = pdf_path.read_bytes()
    else:
        r = requests.get(f"https://arxiv.org/pdf/{aid}", headers=HEADERS, timeout=60)
        if r.status_code != 200:
            return None
        blob = r.content
        pdf_path.write_bytes(blob)
        time.sleep(3)
    return extract_pdf_text(blob)


def extract_pdf_text(blob: bytes) -> str | None:
    try:
        import fitz  # pymupdf
    except ImportError:
        print("    pymupdf not installed — skipping PDF-only submission")
        return None
    with fitz.open(stream=blob, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


# ------------------------------------------------------------- text cleaning


def latex_body(text: str) -> str:
    """Keep only what's between \\begin{document} and \\end{document} if present."""
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
    return m.group(1) if m else text


def latex_to_plain(text: str) -> str:
    from pylatexenc.latex2text import LatexNodes2Text

    return LatexNodes2Text(math_mode="text").latex_to_text(latex_body(text))


def strip_comments(text: str) -> str:
    """Remove LaTeX comment lines (but keep escaped \\%)."""
    return re.sub(r"(?<!\\)%.*", "", text)


# ----------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="983425", help="INSPIRE author recid")
    ap.add_argument("--out", default="corpus_out", help="output directory")
    ap.add_argument("--plain-text", action="store_true", help="de-TeX the sources")
    ap.add_argument("--include-nonfirst", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)

    author = get_author(args.author)
    print(f"Author: {author['name']}  (BAI {author['bai']})")

    records = search_literature(author)
    print(f"Claimed literature records: {len(records)}")

    selected = [
        r
        for r in records
        if (args.include_nonfirst or is_first_author(r, author["recid"]))
        and wanted_type(r)
    ]
    selected.sort(key=lambda r: r["metadata"].get("earliest_date", ""))
    print(f"Selected (first-author, article-like): {len(selected)}")

    corpus_parts, manifest, skipped = [], [], []

    for rec in selected:
        meta = rec["metadata"]
        title = meta["titles"][0]["title"]
        date = meta.get("earliest_date", "????")
        aid = arxiv_id(rec)

        if aid is None:
            # Pre-arXiv paper: fall back to the abstract so it isn't lost entirely
            abstract = (meta.get("abstracts") or [{}])[0].get("value")
            if abstract:
                corpus_parts.append(
                    f"\n\n===== {date} | {title} (abstract only, no arXiv) =====\n\n"
                    + abstract
                )
                manifest.append({"title": title, "date": date, "source": "abstract"})
            else:
                skipped.append({"title": title, "date": date, "reason": "no arXiv id"})
            continue

        print(f"  fetching {aid}  {title[:60]}")
        result = fetch_arxiv_source(aid, cache)
        if result is None:
            text = fetch_arxiv_pdf(aid, cache)
            result = ("pdf", text) if text else None
        if result is None:
            skipped.append({"title": title, "date": date, "reason": "download failed"})
            continue

        kind, text = result
        if kind == "latex":
            text = strip_comments(text)
            if args.plain_text:
                try:
                    text = latex_to_plain(text)
                except Exception as e:
                    print(f"    de-TeX failed ({e}), keeping raw LaTeX")
                    text = latex_body(text)
            else:
                text = latex_body(text)

        corpus_parts.append(f"\n\n===== {date} | {title} =====\n\n" + text.strip())
        manifest.append(
            {"title": title, "date": date, "arxiv": aid, "source": kind}
        )

    corpus = "".join(corpus_parts).strip() + "\n"
    (out / "corpus.txt").write_text(corpus, encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps({"included": manifest, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {out/'corpus.txt'}  ({len(corpus)/1e6:.2f} MB, "
          f"{len(manifest)} documents, {len(skipped)} skipped)")
    if skipped:
        print("Skipped items are listed in manifest.json")


if __name__ == "__main__":
    main()
