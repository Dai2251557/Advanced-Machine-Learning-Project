"""Import a small, explicit paper list from the official arXiv Atom API."""

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ATOM = "{http://www.w3.org/2005/Atom}"
ID_PATTERN = r"(?P<base>\d{4}\.\d{4,5})(?:v(?P<version>[1-9]\d*))?"


def parse_identifier(value):
    text = value.strip()
    if "://" in text:
        url = urllib.parse.urlsplit(text)
        if (url.scheme not in ("https", "http") or url.netloc not in ("arxiv.org", "www.arxiv.org")
                or url.query or url.fragment):
            raise ValueError("Expected an official arXiv abs/PDF URL or a modern arXiv ID.")
        match = re.fullmatch(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?", url.path)
        if not match:
            raise ValueError("Expected an arXiv abs/PDF URL.")
        text = match.group(1)
    match = re.fullmatch(ID_PATTERN, text)
    if not match or not 1 <= int(match.group("base")[2:4]) <= 12:
        raise ValueError("Invalid modern arXiv ID.")
    return match.group("base"), int(match.group("version")) if match.group("version") else None


def parse_feed(raw, requested):
    """Preserve original wording; normalize whitespace introduced by Atom line wrapping."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        raise ValueError("arXiv returned invalid Atom XML.") from None
    if root.tag != ATOM + "feed":
        raise ValueError("arXiv response is not an Atom feed.")
    found = {}
    for entry in root.findall(ATOM + "entry"):
        def field(name):
            node = entry.find(ATOM + name)
            return " ".join("".join(node.itertext()).split()) if node is not None else ""

        base, version = parse_identifier(field("id"))
        if base in found or version is None:
            raise ValueError("arXiv returned duplicate or unversioned records.")
        title, abstract = field("title"), field("summary")
        authors = [" ".join(author.findtext(ATOM + "name", "").split())
                   for author in entry.findall(ATOM + "author")]
        if not title or not abstract or not authors or not all(authors):
            raise ValueError("arXiv record is missing its title, abstract or authors.")
        if not field("published") or not field("updated"):
            raise ValueError("arXiv record is missing publication or update metadata.")
        found[base] = {
            "id": base, "version": version, "title": title, "abstract": abstract,
            "authors": authors, "published": field("published"), "updated": field("updated"),
            "categories": [node.attrib["term"] for node in entry.findall(ATOM + "category")],
            "source_url": f"https://arxiv.org/abs/{base}v{version}",
            "pdf_url": f"https://arxiv.org/pdf/{base}v{version}",
            "abstract_sha256": hashlib.sha256(abstract.encode("utf-8")).hexdigest(),
        }
    ids = [parse_identifier(item)[0] for item in requested]
    if len(ids) != len(set(ids)):
        raise ValueError("Requested paper IDs must be unique, including across versions.")
    if set(ids) != set(found):
        raise ValueError("arXiv response does not exactly match the requested paper list.")
    for item in requested:
        base, version = parse_identifier(item)
        if version is not None and found[base]["version"] != version:
            raise ValueError("arXiv returned a different version than requested.")
        found[base]["requested_source"] = item
    return [found[base] for base in ids]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers", nargs="+", help="Up to 50 modern arXiv IDs or official abs/PDF URLs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not 1 <= len(args.papers) <= 50:
            raise ValueError("Import 1 to 50 papers in a single batch.")
        ids = []
        for item in args.papers:
            base, version = parse_identifier(item)
            if any(base == parse_identifier(other)[0] for other in ids):
                raise ValueError("Duplicate paper ID in request.")
            ids.append(base + (f"v{version}" if version else ""))
        raw_path = args.output.with_suffix(".atom.xml")
        if args.output.exists() or raw_path.exists():
            raise ValueError("Output already exists; choose a new file to preserve the snapshot.")
        query = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
        url = "https://export.arxiv.org/api/query?" + query
        request = urllib.request.Request(url, headers={"User-Agent": "DQN-LiteratureReview/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        papers = parse_feed(raw, args.papers)
        corpus = {
            "dataset_kind": "arxiv_abstracts", "purpose": "pilot_only",
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "metadata_api_url": url,
            "abstract_processing": "Original wording with whitespace normalized; no translation or rewriting.",
            "raw_metadata_file": raw_path.name,
            "raw_metadata_sha256": hashlib.sha256(raw).hexdigest(), "papers": papers,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("xb") as target:
            target.write(raw)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(corpus, target, ensure_ascii=False, indent=2)
            target.write("\n")
        print(f"Imported {len(papers)} papers to {args.output.resolve()}")
        return 0
    except (ValueError, OSError, urllib.error.URLError, KeyError) as exc:
        print(f"arXiv import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
