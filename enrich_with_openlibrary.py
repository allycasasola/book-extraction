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
from fuzzywuzzy import fuzz
from utils import (
    StatusType,
    find_edition_with_earliest_publication,
    _parse_edtf_date,
    select_work_keys_for_expansion,
)

load_dotenv()

# TODO: Add status field
# TODO: Create a separate file for all classes and functions

DEFAULT_INPUT_FILE = f"{os.getenv('DATA_DIR')}/run_books_extracted_info.jsonl"
DEFAULT_OUTPUT_FILE = f"{os.getenv('DATA_DIR')}/run_books_enriched_info.jsonl"

COPYRIGHT_RENEWAL_RECORDS_FILE = (
    f"{os.getenv('DATA_DIR')}/copyright_renewal_records.csv"
)

WORK_FIELDS = "title,author_key,author_name,author_alternative_name,first_publish_year,publish_date,publish_year,key"


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


def _fetch_all_author_names(author_keys: list[str]) -> list[str]:
    """Fetch all author names (including alternative names) for given author keys."""
    author_names = []
    for author_key in author_keys:
        url = f"https://openlibrary.org/authors/{author_key}.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            author_data = response.json()

            # Add primary name
            if "name" in author_data:
                author_names.append(author_data["name"])

            # Add alternative names
            if "alternate_names" in author_data:
                author_names.extend(author_data["alternate_names"])

        except Exception as e:
            continue

    return author_names


def _fetch_title_and_author_by_isbn(isbn: str) -> tuple[str, str] | None:
    """Fetch work key by ISBN from OpenLibrary API."""
    url = f"https://openlibrary.org/isbn/{isbn}.json"
    print(f"Fetching title and author by ISBN: {url}")
    response = requests.get(url)
    title = response.json().get("title")
    # author_key should be the first author's key in the format "/authors/{author_key}"
    author_key = response.json().get("authors")[0].get("key").replace("/authors/", "")
    author_name = _fetch_author_name_by_key(author_key)
    return title, author_name


def _fetch_author_name_by_key(author_key: str) -> str | None:
    url = f"https://openlibrary.org/authors/{author_key}.json"
    response = requests.get(url)
    return response.json().get("name")


def fetch_works(title_variations: list[str], authors: list[str] | None) -> list[dict]:
    """Fetch works from OpenLibrary API. Try multiple title variations and authors. Return all results."""
    works = []
    for i, title_variant in enumerate(title_variations):
        if authors is not None and len(authors) > 0:
            for author in authors:
                params = {
                    "title": title_variant,
                    "author": author,
                }
                # Limit to only 100 results max
                url = f"https://openlibrary.org/search.json?{urlencode(params)}&fields={WORK_FIELDS}&limit=100"
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    data = response.json()

                    # If we found results, return them
                    if data.get("numFound", 0) > 0:
                        # data["docs"] is a list of dictionaries
                        works.extend(data.get("docs", []))
                except Exception as e:
                    continue

        params = {
            "title": title_variant,
        }
        # Limit to only 100 results max
        url = f"https://openlibrary.org/search.json?{urlencode(params)}&fields={WORK_FIELDS}&limit=100"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

        except Exception as e:
            continue
    return works


def filter_works(
    works: list[dict],
    title_variations: list[str],
    authors: list[str],
) -> list[dict]:
    """Fetch and filter works that match the title and author."""
    works_filtered = []
    # Filter works whose title matches exactly (case insensitive)
    # AND author_name/alternative_name matches the author (case insensitive)
    for work in works:
        openlibrary_author_names = []
        if (
            "author_key" not in work
            and "author_name" not in work
            and "author_alternative_name" not in work
        ):
            continue
        if "author_key" in work:
            openlibrary_author_names.extend(_fetch_all_author_names(work["author_key"]))
        if "author_name" in work:
            openlibrary_author_names.extend(work["author_name"])
        if "author_alternative_name" in work:
            openlibrary_author_names.extend(work["author_alternative_name"])
        openlibrary_author_names = list(set(openlibrary_author_names))

        # Hack to handle titles like, "Sunburn: A Novel"
        openlibrary_titles = (
            [work["title"], work["title"].split(":")[0]]
            if ":" in work["title"]
            else [work["title"]]
        )
        # Try any combination of author_name and author
        # Use token_set_ratio to handle titles with extra words like "and Other Essays"
        title_match = any(
            fuzz.token_set_ratio(ol_title.lower(), title_variation.lower()) > 85
            for ol_title in openlibrary_titles
            for title_variation in title_variations
        )

        # For author matching, check for exact match (case-insensitive)
        author_match = any(
            ol_author_name.lower() == author.lower()
            for ol_author_name in openlibrary_author_names
            for author in authors
        )

        if title_match and author_match:
            works_filtered.append(work)
    return works_filtered


def find_work_keys_with_earliest_publication_year(works: list[dict]) -> list[str]:
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

    # Remove the "/works/" prefix from all work keys
    work_keys = [work["key"].replace("/works/", "") for work in earliest_works]
    return work_keys


def fetch_all_editions(work_key: str, max_limit: int = 2000) -> list[dict]:
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
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        total_size = data.get("size", 0)

        if total_size == 0:
            return []

        # Limit the request size for safety
        limit = min(total_size, max_limit)

        # Second call: Fetch all editions with the appropriate limit
        url_with_limit = f"{base_url}?limit={limit}"

        response = requests.get(url_with_limit)
        response.raise_for_status()
        data = response.json()
        entries = data.get("entries", [])

        return entries

    except Exception as e:
        print(f"Error fetching editions: {e}")
        return []


# def find_edition_with_earliest_publication(editions: list[dict]) -> dict | None:
#     """Find the edition with the earliest publication date."""
#     print(
#         f"Finding edition with earliest publication date from {len(editions)} editions"
#     )
#     earliest_publication_date = date(9999, 12, 31)
#     earliest_editions = []
#     for i, edition in enumerate(editions):
#         print(f"Finding edition with earliest publication date from {len(editions)} editions: {i} {edition['title']}")
#         if "publish_date" not in edition:
#             continue
#         publication_date = _parse_edtf_date(edition["publish_date"])
#         if publication_date is None:
#             continue
#         # If the publication date shares the same year as the earliest publication date, add it to the list
#         if publication_date.year == earliest_publication_date.year:
#             earliest_editions.append(edition)
#         # If the publication date is before the earliest publication date, replace the list with the new edition
#         elif publication_date < earliest_publication_date:
#             breakpoint()
#             earliest_publication_date = publication_date
#             earliest_editions = [edition]

#     # If no editions with valid dates found, return None
#     if len(earliest_editions) == 0:
#         return None

#     # Return the first edition that has a publisher and ISBN
#     for edition in earliest_editions:
#         if (
#             edition.get("publishers") is not None
#             and len(edition.get("publishers")) > 0
#             and (
#                 edition.get("isbn_13") is not None or edition.get("isbn_10") is not None
#             )
#         ):
#             return edition

#     # If there is no first edition that has a publisher and ISBN, return the first edition that has a publisher
#     for edition in earliest_editions:
#         if edition.get("publishers") is not None and len(edition.get("publishers")) > 0:
#             return edition

#     # If there is no edition that has a publisher and ISBN, return the first edition
#     return earliest_editions[0]


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


def _get_title_variations(
    title: str, subtitle: str | None, series: str | None
) -> list[str]:
    title_variations = [title]
    if subtitle:
        title_variations.append(f"{title}: {subtitle}")
    if series:
        title_variations.append(f"{title} ({series})")
        title_variations.append(f"{series}: {title}")
    if subtitle and series:
        title_variations.append(f"{title}: {subtitle} ({series})")
    # If the title starts with, "The ", remove it
    if title.lower().startswith("the "):
        title_variations.append(title[4:])
    # If the title starts with, "A ", remove it
    if title.lower().startswith("a "):
        title_variations.append(title[2:])
    # If the title starts with, "An ", remove it
    if title.lower().startswith("an "):
        title_variations.append(title[3:])
    return title_variations


def _fuzzy_title_match(title_variations: list[str], target_title: str) -> bool:
    return any(
        fuzz.ratio(title_variation.lower(), target_title.lower()) >= 70
        for title_variation in title_variations
    )


def _fuzzy_author_match(authors: list[str], target_author: str) -> bool:
    for author in authors:
        author_surname = author.split(" ")[-1].lower()
        if author_surname in target_author.lower():
            return True
    return False


def _is_copyright_renewed(
    title_variations: list[str],
    authors: list[str],
    copyright_renewal_records: pd.DataFrame,
) -> bool:
    for index, row in copyright_renewal_records.iterrows():
        if not isinstance(row["TITLE"], str):
            row["TITLE"] = str(row["TITLE"])
        if not isinstance(row["AUTHOR"], str):
            row["AUTHOR"] = str(row["AUTHOR"])
        if _fuzzy_title_match(title_variations, row["TITLE"]) and _fuzzy_author_match(
            authors, row["AUTHOR"]
        ):
            return True
    return False


def _get_cc_license_code(title_variations: list[str]) -> str | None:
    api_url = "https://wiki.creativecommons.org/api.php"
    for title_variation in title_variations:
        params = {
            "action": "parse",
            "format": "json",
            "page": title_variation.replace(" ", "_"),
            "prop": "text",
        }
        try:
            response = requests.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            html_text = data["parse"]["text"]["*"]
            match = re.search(
                r"https?://creativecommons\.org/licenses/([^/]+)/", html_text
            )
            if match:
                license_code = match.group(1)
                print(f"CC license code for {title_variation}: {license_code}")
                return license_code
            else:
                continue
        except Exception as e:
            print(f"Error getting CC license code for {title_variation}: {e}")
            continue
    return None


def impute_copyright_status(
    title_variations: list[str],
    authors: list[str],
    first_publication_date: date,
    copyright_renewal_records: pd.DataFrame,
) -> str:
    cc_license_code = _get_cc_license_code(title_variations)
    if cc_license_code is not None:
        return cc_license_code

    public_domain_cutoff_year = date.today().year - 96  # Should be 1929

    if first_publication_date is None:
        return "unknown"
    elif first_publication_date.year <= public_domain_cutoff_year:
        return "public domain"
    elif first_publication_date.year >= 1930 and first_publication_date.year <= 1963:
        if _is_copyright_renewed(title_variations, authors, copyright_renewal_records):
            return "all rights reserved"
        else:
            return "public domain"
    else:
        return "all rights reserved"


def deduplicate_works(works: list[dict]) -> list[dict]:
    """Deduplicate works by key."""
    seen_keys = set()
    deduplicated_works = []
    for work in works:
        if work["key"] in seen_keys:
            continue
        seen_keys.add(work["key"])
        deduplicated_works.append(work)
    return deduplicated_works


def enrich_with_openlibrary(
    metadata: OverallBookMetadata,
    copyright_renewal_records: pd.DataFrame = None,
) -> OverallBookMetadata:
    """
    Enrich metadata with OpenLibrary data.
    Fetches the original version metadata and adds it to the metadata object.
    """
    try:
        works = []
        # Step 1: Try to fetch and filter works using title variations and each author; if author is missing, then just use the title variations
        if metadata.title is not None:
            title_variations = _get_title_variations(
                metadata.title, metadata.subtitle, metadata.series
            )

            works = fetch_works(title_variations, metadata.author)
            print(f"Works fetched: {len(works)}")
            works = deduplicate_works(works)
            print(f"Works deduplicated: {len(works)}")
            print(works)
            works = filter_works(works, title_variations, metadata.author)
            print(f"Works filtered: {len(works)}")
            print(works)

        # Step 2: If no works found, try to fetch title and author from ISBN and then fetch works
        if len(works) == 0:
            # Collect all ISBNs (both ISBN-13 and ISBN-10)
            isbns = []
            if metadata.books3_version_metadata.isbn_13:
                isbns.extend(metadata.books3_version_metadata.isbn_13)
            if metadata.books3_version_metadata.isbn_10:
                isbns.extend(metadata.books3_version_metadata.isbn_10)

            # Try each ISBN until we find a match
            for isbn in isbns:
                try:
                    title, author = _fetch_title_and_author_by_isbn(isbn)
                    title_variations = _get_title_variations(
                        title, metadata.subtitle, metadata.series
                    )
                    if title and author:
                        works = fetch_works(title_variations, [author])
                        print(f"Works fetched: {len(works)}")
                        works = deduplicate_works(works)
                        print(f"Works deduplicated: {len(works)}")
                        works = filter_works(works, title_variations, [author])
                        print(f"Works filtered: {len(works)}")
                        if len(works) > 0:
                            break
                except Exception as e:
                    continue

        # Step 3: If no works still found, try to fetch title from filename and then fetch works
        # breakpoint()
        # if len(works) == 0:
        #     title = metadata.filename.replace(".txt", "").replace("_", " ")
        #     title_variations = _get_title_variations(
        #         title, metadata.subtitle, metadata.series
        #     )
        #     works = fetch_works(title_variations, metadata.author, debug)
        #     print(f"Works fetched: {len(works)}")
        #     works = deduplicate_works(works)
        #     print(f"Works deduplicated: {len(works)}")
        #     works = filter_works(works, title_variations, metadata.author, debug)
        #     print(f"Works filtered: {len(works)}")
        #     if len(works) > 0:
        #         print(f"Works found using title from filename '{title}': {len(works)}")

        # Step 4: Get all work keys with earliest publication year
        if len(works) > 0:
            # Step 5: Fetch all editions from all work keys
            all_editions = []
            work_keys = select_work_keys_for_expansion(works)
            if work_keys:
                for work_key in work_keys:
                    editions = fetch_all_editions(work_key)

                    if editions:
                        all_editions.extend(editions)

                if all_editions:
                    # Step 6: Get edition with earliest publication date
                    earliest_edition = find_edition_with_earliest_publication(
                        all_editions
                    )

                    if earliest_edition:
                        # Extract publication date first
                        publication_date = (
                            _parse_edtf_date(earliest_edition["publish_date"])
                            if "publish_date" in earliest_edition
                            and earliest_edition["publish_date"] is not None
                            else None
                        )

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
                            publication_date=publication_date,
                            status=impute_copyright_status(
                                title_variations,
                                metadata.author,
                                publication_date,
                                copyright_renewal_records,
                            ),
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
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="Process only first N entries"
    )
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    limit = args.limit

    # Load copyright renewal records
    copyright_renewal_records = pd.read_csv(COPYRIGHT_RENEWAL_RECORDS_FILE)

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
            enriched_metadata = enrich_with_openlibrary(
                metadata, copyright_renewal_records
            )

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
