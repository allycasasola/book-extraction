from openai import OpenAI
from pydantic import BaseModel
import os
import argparse
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from tqdm import tqdm
import random
from datetime import date
import json
from utils import StatusType
import requests
from urllib.parse import urlencode

# TODO: Search API using ISBN
# TODO: Add better search for multiple authors

load_dotenv()

MAX_FIRST_WORD_COUNT = 6000
MAX_LAST_WORD_COUNT = 2500
DEFAULT_MODEL = "gpt-5"
DEFAULT_DATA_DIR = f"{os.getenv('DATA_DIR')}/run-books"
DEFAULT_OUTPUT_FILE = f"{os.getenv('DATA_DIR')}/books3_extracted_info.jsonl"
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful assistant for bibliographic metadata extraction. You will be provided roughly the first words and the last words of a book. This first 6,000 words may include the title page, copyright page, preface, or introduction. The last 2,500 words may include the conclusion, appendix, afterword, credits, bibliography, or other relevant information. From this text, extract the following information as accurately as possible:
- Main title: the book's main title (excluding the series name, edition notes, or subtitle, unless part of the main title)
- Subtitle: the book's subtitle (if any)
- Series: the book's series (if any)
- Author: the main author(s) as a list
- Translator: the translator(s) (if any) as a list
- ISBN-13: all ISBN-13s found in the text as a list
- ISBN-10: all ISBN-10s found in the text as a list
- Publication date: the date that the book was published
- Status: the copyright or licensing status of the book (e.g. "all rights reserved", "public domain", "cc-by", "cc-by-nc-sa", "cc-by-nd", "cc0", "orphan work", "government work", "fair use")
- Publisher: the name(s) of the publisher(s) of the book as a list

Rules and disambiguation guidelines:
- Extract ALL ISBNs found in the text. If both ISBN-13 and ISBN-10 are present, extract both. If only one type is present, extract that type and set the other to null or an empty list.
- For any of the ISBNs, do not include the dashes.
- For fields that are lists (author, translator, publisher, isbn_13, isbn_10), add each item as a separate element in the list. Do not use commas or semicolons to separate items within a single string.
- Never make up information. Only use the information provided in the text.
- If you can only determine the year of publication from the text but not the month and day, set the publication date to the first day of that year. If you can determine the year and month but not the day, set the publication date to the first day of that month. If you can determine the year, month, and day, set the publication date to the exact date.
"""

TITLE_INFERENCE_PROMPT = """
The filename for this book is: {filename}

Based on this filename and the text provided, please infer the book's title. The filename often contains the title with underscores or other formatting. Use both the filename and any title information you can find in the text to determine the most accurate title. Extract all else normally.
"""

AUTHOR_MATCHING_PROMPT = """
We have identified the title as: {title}

The following authors have written works with this title according to OpenLibrary:
{author_list}

Please review the text and determine which of these authors (if any) wrote this specific book. Look for author names in the copyright page, title page, or any other mentions in the text. Return only the authors that you can confirm from the text. Extract all else normally.
"""


class BookMetadata(BaseModel):
    translator: list[str] | None = None
    isbn_13: list[str] | None = None
    isbn_10: list[str] | None = None
    publication_date: date | None = None
    status: StatusType | None = None
    publisher: list[str] | None = None


class ChatGPTExtraction(BaseModel):
    """Model for initial extraction from ChatGPT."""

    title: str | None = None
    subtitle: str | None = None
    series: str | None = None
    author: list[str] | None = None
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


def fetch_authors_from_openlibrary(title: str, debug: bool = False) -> list[str]:
    """Fetch all possible authors for works with the given title from OpenLibrary."""
    params = {"title": title}
    url = f"https://openlibrary.org/search.json?{urlencode(params)}&fields=author_name&limit=50"

    if debug:
        print(f"Fetching authors for title '{title}' from OpenLibrary...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Collect all unique author names
        authors = set()
        for work in data.get("docs", []):
            if "author_name" in work:
                authors.update(work["author_name"])

        author_list = sorted(list(authors))
        if debug:
            print(f"Found {len(author_list)} unique authors: {author_list}")

        return author_list
    except Exception as e:
        if debug:
            print(f"Error fetching authors from OpenLibrary: {e}")
        return []


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


def format_text(text, max_first_word_count, max_last_word_count):
    words = text.split()

    # Get first N words
    first_words = words[:max_first_word_count]
    first_text = " ".join(first_words)

    # Get last N words
    last_words = words[-max_last_word_count:]
    last_text = " ".join(last_words)

    new_text = ""
    new_text += "START OF BEGINNING TEXT:\n"
    new_text += first_text
    new_text += "\nEND OF BEGINNING TEXT\n"
    new_text += "START OF FINAL TEXT:\n"
    new_text += last_text
    new_text += "\nEND OF FINAL TEXT\n"
    return new_text


def extract_metadata(
    file_path, model, max_first_word_count, max_last_word_count, output_file, debug
):
    """
    Extract title, subtitle, series, author, and books3_version_metadata using ChatGPT.
    Uses multi-pass extraction with fallback strategies.
    """
    # Read and check the text
    with open(file_path, "r") as f:
        original_text = f.read()

    # Format the text with word limits
    text = format_text(original_text, max_first_word_count, max_last_word_count)

    if debug:
        print(f"Original text: {original_word_count} words")
        print(text)

    filename = os.path.basename(file_path)

    # PASS 1: Initial extraction
    print(f"Pass 1: Extracting metadata from {file_path} using ChatGPT...")
    response = OPENAI_CLIENT.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=ChatGPTExtraction,
    )

    if hasattr(response, "usage"):
        input_tokens = getattr(response.usage, "input_tokens", 0)
        print(f"  Input tokens: {input_tokens}")
        output_tokens = getattr(response.usage, "output_tokens", 0)
        print(f"  Output tokens: {output_tokens}")

    extracted = response.output_parsed
    if debug:
        print(f"  Extracted metadata: {extracted}")

    # PASS 2: If title is None, try again with filename hint
    if extracted.title is None:
        print(f"Pass 2: Title not found. Retrying with filename hint...")
        filename_without_ext = filename.replace(".txt", "").replace("_", " ")
        enhanced_prompt = (
            TITLE_INFERENCE_PROMPT.format(filename=filename_without_ext) + "\n\n" + text
        )

        response = OPENAI_CLIENT.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_prompt},
            ],
            text_format=ChatGPTExtraction,
        )

        extracted = response.output_parsed
        if debug:
            print(f"  Extracted metadata with filename hint: {extracted}")

        if extracted.title:
            print(f"  ✓ Title found: {extracted.title}")

    # PASS 3: If author is None/empty but title exists, fetch candidates from OpenLibrary
    if extracted.title and (not extracted.author or len(extracted.author) == 0):
        print(
            f"Pass 3: Author not found. Fetching candidate authors from OpenLibrary..."
        )
        candidate_authors = fetch_authors_from_openlibrary(extracted.title, debug)

        if candidate_authors:
            print(f"  Found {len(candidate_authors)} candidate authors")
            author_list_str = "\n".join(f"- {author}" for author in candidate_authors)
            enhanced_prompt = (
                AUTHOR_MATCHING_PROMPT.format(
                    title=extracted.title, author_list=author_list_str
                )
                + "\n\n"
                + text
            )

            response = OPENAI_CLIENT.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": enhanced_prompt},
                ],
                text_format=ChatGPTExtraction,
            )

            extracted = response.output_parsed
            if debug:
                print(f"  Extracted metadata with author candidates: {extracted}")

            if extracted.author and len(extracted.author) > 0:
                print(f"  ✓ Author(s) found: {extracted.author}")
        else:
            print(f"  ✗ No candidate authors found in OpenLibrary")

    # Create OverallBookMetadata instance with extracted info
    overall_metadata = OverallBookMetadata(
        filename=filename,
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
        original_version_metadata=None,  # Will be filled by enrich_with_openlibrary.py
    )

    # Save to output file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        f.write(overall_metadata.model_dump_json() + "\n")

    print(f"✓ Saved metadata for {overall_metadata.filename}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from Books3 text files using ChatGPT"
    )
    parser.add_argument("--data_dir", "-d", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max_first_word_count", "-f", type=int, default=MAX_FIRST_WORD_COUNT
    )
    parser.add_argument(
        "--max_last_word_count", "-l", type=int, default=MAX_LAST_WORD_COUNT
    )
    parser.add_argument("--output_file", "-o", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--sample", "-s", type=int, default=None, help="Process only N random files"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    model = args.model
    max_first_word_count = args.max_first_word_count
    max_last_word_count = args.max_last_word_count
    output_file = args.output_file
    debug = args.debug

    # Get list of files
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    # Check for already processed files
    processed_filenames = get_processed_filenames(output_file)
    if processed_filenames:
        print(
            f"Found {len(processed_filenames)} already processed files. Skipping those."
        )

    # Filter out already processed files
    unprocessed_files = [f for f in all_files if f not in processed_filenames]

    if not unprocessed_files:
        print("All files have already been processed!")
        return

    print(
        f"{len(unprocessed_files)} files remaining to process (out of {len(all_files)} total)"
    )

    if args.sample:
        print(f"Sampling {args.sample} random files from unprocessed files...")
        files = random.sample(
            unprocessed_files, min(args.sample, len(unprocessed_files))
        )
    elif debug:
        print("Debug mode enabled. Processing 5 random unprocessed files...")
        files = random.sample(unprocessed_files, min(5, len(unprocessed_files)))
    else:
        files = unprocessed_files

    skipped = len(all_files) - len(unprocessed_files)
    print(f"Processing {len(files)} files (skipping {skipped} already processed)")

    for file in tqdm(files, desc="Extracting metadata from files", total=len(files)):
        file_path = os.path.join(data_dir, file)
        extract_metadata(
            file_path,
            model,
            max_first_word_count,
            max_last_word_count,
            output_file,
            debug,
        )

    print(f"Done extracting metadata from all files! Output saved to {output_file}")


if __name__ == "__main__":
    main()
