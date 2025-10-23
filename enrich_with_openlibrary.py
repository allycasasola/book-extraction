from pydantic import BaseModel
import os
import argparse
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from tqdm import tqdm
from urllib.parse import urlencode
import requests
from datetime import date
from dateutil import parser as date_parser
import re
import json
import zipfile
import pandas as pd
from io import BytesIO

load_dotenv()

# TODO: Add status field

DEFAULT_INPUT_FILE = f"{os.getenv('DATA_DIR')}/books3_extracted_info.jsonl"
DEFAULT_OUTPUT_FILE = f"{os.getenv('DATA_DIR')}/enriched_info.jsonl"

COPYRIGHT_RENEWALS_DATA_FILE_URL = "https://web.stanford.edu/dept/SUL/collections/copyrightrenewals/files/20170427-copyright-renewals-records.csv.zip"



WORK_FIELDS = "title,author_key,author_name,author_alternative_name,first_publish_year,publish_date,publish_year,key"

StatusType = Literal[
    "all rights reserved",
    "public domain",
    "cc-by",
    "cc-by-nc-sa",
    "cc-by-nd",
    "cc0",
    "orphan work",
    "government work",
    "fair use",
]


class BookMetadata(BaseModel):
    translator: list[str] | None = None
    isbn_13: list[str] | None = None
    isbn_10: list[str] | None = None
    publication_date: date | None = None
    status: StatusType | None = None
    publisher: list[str] | None = None


class OverallBookMetadata(BaseModel):
    filename: str | None = None
    title: str | None = None
    subtitle: str | None = None
    series: str | None = None
    author: list[str] | None = None
    books3_version_metadata: BookMetadata | None = None
    original_version_metadata: BookMetadata | None = None


def _fetch_all_author_names(author_keys: list[str], debug: bool = False) -> list[str]:
    """Fetch all author names (including alternative names) for given author keys."""
    author_names = []
    for author_key in author_keys:
        url = f"https://openlibrary.org/authors/{author_key}.json"
        if debug:
            print(f"Fetching author names for {author_key} from {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                author_data = response.json()
                # Add primary name
                if "name" in author_data:
                    author_names.append(author_data["name"])
                # Add alternative names
                if "alternate_names" in author_data:
                    author_names.extend(author_data["alternate_names"])
        except Exception as e:
            print(f"Error fetching author names for {author_key}: {e}")
    return author_names


def _fetch_title_and_author_by_isbn(
    isbn: str, debug: bool = False
) -> tuple[str, str] | None:
    """Fetch work key by ISBN from OpenLibrary API."""
    url = f"https://openlibrary.org/isbn/{isbn}.json"
    if debug:
        print(f"Fetching title and author by ISBN '{isbn}' from {url}")
    response = requests.get(url)
    title = response.json().get("title")
    # author_key should be the first author's key in the format "/authors/{author_key}"
    author_key = response.json().get("authors")[0].get("key").replace("/authors/", "")
    author_name = _fetch_author_name_by_key(author_key)
    return title, author_name


def _fetch_author_name_by_key(author_key: str, debug: bool = False) -> str | None:
    url = f"https://openlibrary.org/authors/{author_key}.json"
    if debug:
        print(f"Fetching author name by key '{author_key}' from {url}")
    response = requests.get(url)
    return response.json().get("name")


def _fetch_works(title: str, author: str, debug: bool = False) -> dict | None:
    """Fetch works from OpenLibrary API."""
    params = {
        "title": title,
        "author": author,
    }
    # Limit to only 100 results max
    url = f"https://openlibrary.org/search.json?{urlencode(params)}&fields={WORK_FIELDS}&limit=100"
    if debug:
        print(f"Fetching works from {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching works: {e}")
        return None


def _parse_edtf_date(edtf_date: str, debug: bool = False) -> date | None:
    """Parse various date formats from OpenLibrary."""
    try:
        # First check if it's just a 4-digit year
        year_only_match = re.match(r"^\s*(\d{4})\s*$", edtf_date)
        if year_only_match:
            year = int(year_only_match.group(1))
            parsed_date = date(year, 1, 1)
            if debug:
                print(f"Parsed EDTF date '{edtf_date}' as '{parsed_date}'")
            return parsed_date

        # Try to parse using dateutil which handles many formats
        parsed_date = date_parser.parse(edtf_date, fuzzy=True)
        if debug:
            print(f"Parsed EDTF date '{edtf_date}' as '{parsed_date}'")
        return parsed_date.date()
    except Exception as e:
        # Fallback: try to extract just the year
        try:
            # Look for a 4-digit year
            year_match = re.search(r"\b(1\d{3}|20\d{2})\b", edtf_date)
            if year_match:
                year = int(year_match.group(1))
                return date(year, 1, 1)
        except Exception:
            pass

        print(f"Error parsing date '{edtf_date}': {e}")
        return None


def fetch_and_filter_works(
    title: str, author: str, debug: bool = False
) -> list[dict] | None:
    """Fetch and filter works that match the title and author."""
    works = _fetch_works(title, author, debug)
    works_filtered = []
    if works is None or works.get("numFound", 0) == 0:
        return None
    # Filter works whose title matches exactly (case insensitive)
    # AND author_name/alternative_name matches the author (case insensitive)
    for work in works["docs"]:
        if "author_key" not in work:
            continue
        author_names = _fetch_all_author_names(work["author_key"])
        if work["title"].lower() == title.lower() and any(
            author_name.lower() == author.lower() for author_name in author_names
        ):
            works_filtered.append(work)
    return works_filtered if works_filtered else None


def find_work_keys_with_earliest_publication_year(
    works: list[dict], debug: bool = False
) -> list[str]:
    """Find all work keys with the earliest publication year.

    Returns:
        List of work keys (without "/works/" prefix) that share the earliest publication year
    """
    if not works:
        return []

    # Find the earliest publication year
    earliest_publication_year = float("inf")
    for work in works:
        if "first_publish_year" in work:
            if work["first_publish_year"] < earliest_publication_year:
                earliest_publication_year = work["first_publish_year"]

    if earliest_publication_year == float("inf"):
        return []

    # Collect all works with that earliest year
    earliest_works = []
    for work in works:
        if (
            "first_publish_year" in work
            and work["first_publish_year"] == earliest_publication_year
        ):
            earliest_works.append(work)

    if debug:
        print(
            f"Found {len(earliest_works)} work(s) with earliest publication year {earliest_publication_year}"
        )
        for work in earliest_works:
            print(f"  - {work.get('key', 'unknown')}")

    # Remove the "/works/" prefix from all work keys
    work_keys = [work["key"].replace("/works/", "") for work in earliest_works]
    return work_keys


def fetch_all_editions(
    work_key: str, max_limit: int = 2000, debug: bool = False
) -> list[dict]:
    """Fetch all editions for a given work.

    This function makes two API calls:
    1. First call to get the total size
    2. Second call with limit=size to fetch all editions

    Args:
        work_key: The OpenLibrary work key
        max_limit: Maximum limit to use even if size is larger (default 2000 for safety)
        debug: Enable debug output

    Returns:
        List of edition dictionaries
    """
    base_url = f"https://openlibrary.org/works/{work_key}/editions.json"

    try:
        # First call: Get the total size
        if debug:
            print(f"Fetching edition count from {base_url}")
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        total_size = data.get("size", 0)

        if total_size == 0:
            if debug:
                print("No editions found")
            return []

        # Limit the request size for safety
        limit = min(total_size, max_limit)

        if debug:
            print(f"Total editions available: {total_size}")
            if total_size > max_limit:
                print(f"Limiting request to {max_limit} editions for safety")

        # Second call: Fetch all editions with the appropriate limit
        url_with_limit = f"{base_url}?limit={limit}"
        if debug:
            print(f"Fetching {limit} editions from {url_with_limit}")

        response = requests.get(url_with_limit)
        response.raise_for_status()
        data = response.json()
        entries = data.get("entries", [])

        if debug:
            print(f"Successfully fetched {len(entries)} editions")

        return entries

    except Exception as e:
        print(f"Error fetching editions: {e}")
        return []


def find_edition_with_earliest_publication(
    editions: list[dict], debug: bool = False
) -> dict | None:
    """Find the edition with the earliest publication date."""
    print(
        f"Finding edition with earliest publication date from {len(editions)} editions"
    )
    earliest_publication_date = date(9999, 12, 31)
    earliest_editions = []
    for edition in editions:
        if debug:
            print(f"Checking edition: ")
            print(edition)
        if "publish_date" not in edition:
            continue
        publication_date = _parse_edtf_date(edition["publish_date"], debug)
        if publication_date is None:
            continue
        # If the publication date shares the same year as the earliest publication date, add it to the list
        if publication_date.year == earliest_publication_date.year:
            earliest_editions.append(edition)
        # If the publication date is before the earliest publication date, replace the list with the new edition
        elif publication_date < earliest_publication_date:
            earliest_publication_date = publication_date
            earliest_editions = [edition]

    # If no editions with valid dates found, return None
    if len(earliest_editions) == 0:
        return None

    # Return the first edition that has a publisher and ISBN
    for edition in earliest_editions:
        if (
            edition.get("publishers") is not None
            and len(edition.get("publishers")) > 0
            and (
                edition.get("isbn_13") is not None or edition.get("isbn_10") is not None
            )
        ):
            return edition

    # If there is no first edition that has a publisher and ISBN, return the first edition that has a publisher
    for edition in earliest_editions:
        if edition.get("publishers") is not None and len(edition.get("publishers")) > 0:
            return edition

    # If there is no edition that has a publisher and ISBN, return the first edition
    return earliest_editions[0]


def get_processed_filenames(output_file: str) -> set[str]:
    """Read the output file and return a set of already processed filenames."""
    if not os.path.exists(output_file):
        return set()

    processed = set()
    try:
        with open(output_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "filename" in data and data["filename"]:
                        processed.add(data["filename"])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Could not read existing output file: {e}")

    return processed




def enrich_with_openlibrary(
    metadata: OverallBookMetadata, debug: bool = False
) -> OverallBookMetadata:
    """
    Enrich metadata with OpenLibrary data.
    Fetches the original version metadata and adds it to the metadata object.
    """
    if not metadata.title or not metadata.author or len(metadata.author) == 0:
        print(f"Skipping {metadata.filename} - missing title or author")
        return metadata

    print(f"Fetching original version metadata for {metadata.filename}...")
    try:
        # Step 1: Try to fetch and filter works using extracted title and each author
        works = None
        for author_name in metadata.author:
            works = fetch_and_filter_works(metadata.title, author_name, debug)
            if works and len(works) > 0:
                if debug:
                    print(
                        f"Works found using extracted title '{metadata.title}' and author '{author_name}': {len(works)}"
                    )
                break

        if debug and (not works or len(works) == 0):
            print(f"No works found using extracted title and authors")

        # Step 2: If no works found, try to fetch title and author and then fetch works from ISBN
        if not works or len(works) == 0:
            # Collect all ISBNs (both ISBN-13 and ISBN-10)
            isbns = []
            if metadata.books3_version_metadata.isbn_13:
                isbns.extend(metadata.books3_version_metadata.isbn_13)
            if metadata.books3_version_metadata.isbn_10:
                isbns.extend(metadata.books3_version_metadata.isbn_10)

            # Try each ISBN until we find a match
            for isbn in isbns:
                try:
                    title, author = _fetch_title_and_author_by_isbn(isbn, debug)
                    if title and author:
                        works = fetch_and_filter_works(title, author, debug)
                        if works and len(works) > 0:
                            if debug:
                                print(
                                    f"Works found using title and author from ISBN '{isbn}': {len(works)}"
                                )
                            break
                        else:
                            if debug:
                                print(
                                    f"No works found using title and author from ISBN: {isbn}. Trying next ISBN."
                                )
                except Exception as e:
                    if debug:
                        print(f"Error fetching from ISBN {isbn}: {e}")

        # Step 3: Get all work keys with earliest publication year
        if works and len(works) > 0:
            work_keys = find_work_keys_with_earliest_publication_year(works, debug)
            if debug:
                print(f"Work keys with earliest publication year: {work_keys}")

            # Step 4: Fetch all editions from all work keys
            all_editions = []
            if work_keys:
                for work_key in work_keys:
                    if debug:
                        print(f"Fetching editions for work key: {work_key}")
                    editions = fetch_all_editions(work_key, debug=debug)
                    if editions:
                        all_editions.extend(editions)
                        if debug:
                            print(f"  Added {len(editions)} editions from this work")

                if debug:
                    print(f"Total editions found across all works: {len(all_editions)}")

                if all_editions:
                    # Step 5: Get edition with earliest publication date
                    earliest_edition = find_edition_with_earliest_publication(
                        all_editions, debug
                    )
                    if debug:
                        print(
                            f"Earliest edition: {earliest_edition.get('title') if earliest_edition else None}"
                        )

                    if earliest_edition:
                        # Fill out original_version_metadata from the earliest edition
                        metadata.original_version_metadata = BookMetadata(
                            translator=None,  # TO-DO: Translator info not typically in OpenLibrary editions
                            isbn_13=(
                                earliest_edition.get("isbn_13", [])
                                if "isbn_13" in earliest_edition
                                and earliest_edition["isbn_13"]
                                else None
                            ),
                            isbn_10=(
                                earliest_edition.get("isbn_10", [])
                                if "isbn_10" in earliest_edition
                                and earliest_edition["isbn_10"]
                                else None
                            ),
                            publication_date=(
                                _parse_edtf_date(
                                    earliest_edition["publish_date"], debug
                                )
                                if "publish_date" in earliest_edition
                                and earliest_edition["publish_date"] is not None
                                else None
                            ),
                            status=None,  # Status info not in OpenLibrary
                            publisher=(
                                earliest_edition.get("publishers", [])
                                if "publishers" in earliest_edition
                                else None
                            ),
                        )
                        print(
                            f"✓ Found original edition metadata for {metadata.filename}"
                        )
                    else:
                        print(
                            f"✗ No edition with valid publication date found for {metadata.filename}"
                        )
                else:
                    print(f"✗ No editions found for works {work_keys}")
            else:
                print(f"✗ No work keys found for {metadata.filename}")
        else:
            print(f"✗ No matching works found on OpenLibrary for {metadata.filename}")
    except Exception as e:
        print(f"Error fetching from OpenLibrary for {metadata.filename}: {e}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Enrich Books3 metadata with OpenLibrary information"
    )
    parser.add_argument("--input_file", "-i", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output_file", "-o", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Process only first N entries"
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    debug = args.debug
    limit = args.limit

    # Download copyright renewals data
    download_copyright_renewals_data()
    copyright_renewals_data = load_copyright_renewals_data()

    # Read input file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return

    print(f"Reading metadata from {input_file}...")
    with open(input_file, "r") as f:
        lines = f.readlines()

    # Check for already processed files
    processed_filenames = get_processed_filenames(output_file)
    if processed_filenames:
        print(
            f"Found {len(processed_filenames)} already processed files. Skipping those."
        )

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Filter and process entries
    skipped_entries = 0
    processed_entries = 0
    error_entries = 0

    # Parse all entries first to count them
    all_entries = []
    for line in lines:
        try:
            data = json.loads(line)
            metadata = OverallBookMetadata(**data)
            all_entries.append(metadata)
        except Exception as e:
            print(f"Error parsing line: {e}")
            error_entries += 1
            continue

    # Filter out already processed entries
    entries_to_process = [
        entry for entry in all_entries if entry.filename not in processed_filenames
    ]

    skipped_entries = len(all_entries) - len(entries_to_process)

    if not entries_to_process:
        print("All entries have already been processed!")
        return

    print(
        f"{len(entries_to_process)} entries remaining to process (out of {len(all_entries)} total)"
    )

    if limit:
        entries_to_process = entries_to_process[:limit]
        print(f"Processing first {limit} unprocessed entries...")

    print(
        f"Processing {len(entries_to_process)} entries (skipping {skipped_entries} already processed)"
    )

    # Process each entry
    for metadata in tqdm(
        entries_to_process, desc="Enriching metadata with OpenLibrary"
    ):
        try:
            # Enrich with OpenLibrary data
            enriched_metadata = enrich_with_openlibrary(metadata, debug)

            # Write to output file
            with open(output_file, "a") as f:
                f.write(enriched_metadata.model_dump_json() + "\n")

            processed_entries += 1

        except Exception as e:
            print(f"Error processing {metadata.filename}: {e}")
            error_entries += 1
            continue

    print(f"\nDone! Enriched metadata saved to {output_file}")
    print(
        f"Summary: {processed_entries} processed, {skipped_entries} skipped, {error_entries} errors"
    )


if __name__ == "__main__":
    main()
