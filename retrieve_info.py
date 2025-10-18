from openai import OpenAI
from pydantic import BaseModel
import os
import argparse
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from tqdm import tqdm
from urllib.parse import urlencode
import random
import requests
from datetime import date
from dateutil import parser as date_parser
import re

load_dotenv()

# TODO: Implement batch queue
# TODO: Add debug mode

MAX_FIRST_WORD_COUNT = 6000
MAX_LAST_WORD_COUNT = 2500
DEFAULT_MODEL = "gpt-5"
DEFAULT_DATA_DIR = f"{os.getenv('DATA_DIR')}/run-books"
DEFAULT_OUTPUT_FILE = f"{os.getenv('DATA_DIR')}/extracted_info.jsonl"
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant for bibliographic metadata extraction. You will be provided roughly the first 6,000 words and the last 2,500 words of a book. This first 6,000 words may include the title page, copyright page, preface, or introduction. The last 2,500 words may include the conclusion, appendix, afterword, credits, bibliography, or other relevant information. From this text, extract the following information as accurately as possible:
- Main title: the book's main title (excluding the series name, edition notes, or subtitle, unless part of the main title)
- Subtitle: the book's subtitle (if any)
- Series: the book's series (if any)
- Author: the main author(s)
- Translator: the translator(s) (if any)
- ISBN-13: the ISBN-13 of the book
- ISBN-10: the ISBN-10 of the book
- Publication date: the date that the book was published
- Status: the copyright or licensing status of the book (e.g. "all rights reserved", "public domain", "cc-by", "cc-by-nc-sa", "cc-by-nd", "cc0", "orphan work", "government work", "fair use")
- Publisher: the name of the publisher of the book

Rules and disambiguation guidelines:
- If you can determine the ISBN-13 from the text, set the ISBN-10 to null. Otherwise, if you cannot determine the ISBN-13 but can determine the ISBN-10 from the text, set the ISBN-13 to null. If you cannot determine either from the text, set both to null.
- For any of the ISBNs, do not include the dashes.
- If there are multiple ISBN-13s (or ISBN-10s) found in the text, separate them with commas.
- If there are multiple authors found in the text, separate them with commas.
- If there are multiple translators found in the text, separate them with commas.
- If there are multiple publishers found in the text, separate them with semicolons.
- Never make up information. Only use the information provided in the text.
- If you can only determine the year of publication from the text, set the publication date to the first day of that year.
"""

WORK_FIELDS = "title,author_key,author_name,author_alternative_name,first_publish_year,publish_date,publish_year,key"


def _fetch_all_author_names(author_keys: list[str]) -> list[str]:
    """Fetch all author names (including alternative names) for given author keys."""

    author_names = []
    for author_key in author_keys:
        url = f"https://openlibrary.org/authors/{author_key}.json"
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


# TODO: Add a retry mechanism
def _fetch_works(title: str, author: str) -> dict | None:
    # Convert title and author to url encoded
    params = {
        "title": title,
        "author": author,
    }
    # Limit to only 100 results max
    url = f"https://openlibrary.org/search.json?{urlencode(params)}&fields={WORK_FIELDS}&limit=100"
    response = requests.get(url)
    return response.json()


def _parse_edtf_date(edtf_date: str) -> date | None:
    """Parse various date formats from OpenLibrary."""
    try:
        # Try to parse using dateutil which handles many formats
        parsed_date = date_parser.parse(edtf_date, fuzzy=True)
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


def fetch_and_filter_works(title: str, author: str) -> list[dict] | None:
    works = _fetch_works(title, author)
    works_filtered = []
    if works is None or works["numFound"] == 0:
        return None
    # Filter works whose title/alternative_title matches the title (case insensitive) AND author_name/alternative_name matches the author (case insensitive)
    for work in works["docs"]:
        author_names = _fetch_all_author_names(work["author_key"])
        if work["title"].lower() == title.lower() and any(
            author_name.lower() == author.lower() for author_name in author_names
        ):
            works_filtered.append(work)
    return works_filtered


def find_work_key_with_earliest_publication_year(works: list[dict]) -> str | None:
    earliest_publication_year = float("inf")
    earliest_work = None
    for work in works:
        if work["first_publish_year"] < earliest_publication_year:
            earliest_publication_year = work["first_publish_year"]
            earliest_work = work
    # Remove the "/works/" prefix from the work key
    return earliest_work["key"].replace("/works/", "")


def fetch_all_editions(work_key: str) -> list[dict]:
    url = f"https://openlibrary.org/works/{work_key}/editions.json"
    response = requests.get(url)
    return response.json()["entries"]


def find_edition_with_earliest_publication(editions: list[dict]) -> dict | None:
    earliest_publication_date = date(9999, 12, 31)
    earliest_edition = None
    for edition in editions:
        publication_date = _parse_edtf_date(edition["publish_date"])
        if (
            publication_date is not None
            and publication_date < earliest_publication_date
        ):
            earliest_publication_date = publication_date
            earliest_edition = edition
    return earliest_edition


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
    translator: str | None = None
    isbn_13: str | None = None
    isbn_10: str | None = None
    publication_date: date | None = None
    status: StatusType | None = None
    publisher: str | None = None


class ChatGPTExtraction(BaseModel):
    """Model for initial extraction from ChatGPT."""

    title: str | None = None
    subtitle: str | None = None
    series: str | None = None
    author: str | None = None
    translator: str | None = None
    isbn_13: str | None = None
    isbn_10: str | None = None
    publication_date: date | None = None
    status: StatusType | None = None
    publisher: str | None = None


class OverallBookMetadata(BaseModel):
    filename: str | None = None
    title: str | None = None
    subtitle: str | None = None
    series: str | None = None
    author: str | None = None
    books3_version_metadata: BookMetadata | None = None
    original_version_metadata: BookMetadata | None = None


def format_text(text, max_first_word_count, max_last_word_count):
    new_text = ""
    new_text += "START OF BEGINNING TEXT:\n"
    new_text += text[:max_first_word_count]
    new_text += "\nEND OF BEGINNING TEXT\n"
    new_text += "START OF FINAL TEXT:\n"
    new_text += text[-max_last_word_count:]
    new_text += "\nEND OF FINAL TEXT\n"
    return new_text


def retrieve_metadata(
    file_path, model, max_first_word_count, max_last_word_count, output_file, debug
):
    """
    Step 1: Create OverallBookMetadata instance
    Step 2: Extract title, subtitle, series, author, and books3_version_metadata using ChatGPT
    Step 3: Using title and author, get original_version_metadata from OpenLibrary API
    """
    # Read and format the text
    with open(file_path, "r") as f:
        text = f.read()
    text = format_text(text, max_first_word_count, max_last_word_count)

    if debug:
        print(text)

    if len(text) > MAX_FIRST_WORD_COUNT + MAX_LAST_WORD_COUNT + 100:
        print(f"Text is too long: {len(text)} words. Skipping...")
        return

    # Step 2: Extract metadata using ChatGPT
    print(f"Extracting metadata from {file_path} using ChatGPT...")
    response = OPENAI_CLIENT.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=ChatGPTExtraction,
    )

    if hasattr(response, "usage"):
        # TODO: Track usage
        input_tokens = getattr(response.usage, "input_tokens", 0)
        print(f"Input tokens: {input_tokens}")
        output_tokens = getattr(response.usage, "output_tokens", 0)
        print(f"Output tokens: {output_tokens}")

    extracted = response.output_parsed
    if debug:
        print(f"Extracted metadata from ChatGPT: {extracted}")
    # Step 1: Create OverallBookMetadata instance with extracted info
    overall_metadata = OverallBookMetadata(
        filename=os.path.basename(file_path),
        title=extracted.title,
        subtitle=extracted.subtitle,
        series=extracted.series,
        author=extracted.author,
        books3_version_metadata=BookMetadata(
            translator=extracted.translator,
            isbn_13=extracted.isbn_13,
            isbn_10=extracted.isbn_10,
            publication_date=extracted.publication_date,
            status=extracted.status,
            publisher=extracted.publisher,
        ),
        original_version_metadata=None,  # Will be filled in Step 3
    )

    # Step 3: Get original version metadata from OpenLibrary API
    if extracted.title and extracted.author:
        print(f"Fetching original version metadata from OpenLibrary...")
        try:
            # 3a: Fetch and filter works
            works = fetch_and_filter_works(extracted.title, extracted.author)
            if debug:
                print(f"Works found: {works}")
            if works and len(works) > 0:
                # 3b: Get work key with earliest publication year
                work_key = find_work_key_with_earliest_publication_year(works)
                if debug:
                    print(f"Work key: {work_key}")

                if work_key:
                    breakpoint()
                    # 3c: Get all editions for this work
                    editions = fetch_all_editions(work_key)
                    if debug:
                        print(f"Editions found: {editions}")

                    if editions and len(editions) > 0:
                        # 3d: Get edition with earliest publication date
                        breakpoint()
                        earliest_edition = find_edition_with_earliest_publication(
                            editions
                        )
                        if debug:
                            print(f"Earliest edition: {earliest_edition}")

                        if earliest_edition:
                            # Fill out original_version_metadata from the earliest edition
                            overall_metadata.original_version_metadata = BookMetadata(
                                translator=None,  # Translator info not typically in OpenLibrary editions
                                isbn_13=(
                                    earliest_edition.get("isbn_13", [None])[0]
                                    if "isbn_13" in earliest_edition
                                    and earliest_edition["isbn_13"]
                                    else None
                                ),
                                isbn_10=(
                                    earliest_edition.get("isbn_10", [None])[0]
                                    if "isbn_10" in earliest_edition
                                    and earliest_edition["isbn_10"]
                                    else None
                                ),
                                publication_date=(
                                    _parse_edtf_date(earliest_edition["publish_date"])
                                    if "publish_date" in earliest_edition
                                    and earliest_edition["publish_date"] is not None
                                    else None
                                ),
                                status=None,  # Status info not in OpenLibrary
                                publisher=(
                                    ", ".join(earliest_edition.get("publishers", []))
                                    if "publishers" in earliest_edition
                                    else None
                                ),
                            )
                            print(
                                f"Found original version metadata: {overall_metadata.original_version_metadata}"
                            )
                        else:
                            print(f"No edition with valid publication date found")
                    else:
                        print(f"No editions found for work {work_key}")
                else:
                    print(f"No work key found")
            else:
                print(f"No matching works found on OpenLibrary")
        except Exception as e:
            print(f"Error fetching from OpenLibrary: {e}")
    else:
        print(f"Skipping OpenLibrary lookup - missing title or author")

    # Save to output file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        f.write(overall_metadata.model_dump_json() + "\n")

    print(f"Saved metadata for {overall_metadata.filename}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max_first_word_count", "-f", type=int, default=MAX_FIRST_WORD_COUNT
    )
    parser.add_argument(
        "--max_last_word_count", "-l", type=int, default=MAX_LAST_WORD_COUNT
    )
    parser.add_argument("--output_file", "-o", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    data_dir = args.data_dir
    model = args.model
    max_first_word_count = args.max_first_word_count
    max_last_word_count = args.max_last_word_count
    output_file = args.output_file
    debug = args.debug

    print(f"Using API key: {os.getenv('OPENAI_API_KEY')}")
    if debug:
        # Get 5 random files from the data dir
        print("Debug mode enabled. Retrieving info from 5 random files...")
        files = random.sample(os.listdir(data_dir), 5)
        output_file = f"{os.getenv('DATA_DIR')}/debug_extracted_info.jsonl"
    else:
        files = os.listdir(data_dir)

    for file in tqdm(files, desc="Retrieving metadata from files", total=len(files)):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            retrieve_metadata(
                file_path,
                model,
                max_first_word_count,
                max_last_word_count,
                output_file,
                debug,
            )
        else:
            print(f"Skipping {file}...")

    print("Done retrieving metadata from all files!")


if __name__ == "__main__":
    main()
