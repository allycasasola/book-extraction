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
You are a helpful assistant for bibliographic metadata extraction. You will be provided roughly the first 6,000 words and the last 2,500 words of a book. This first 6,000 words may include the title page, copyright page, preface, or introduction. The last 2,500 words may include the conclusion, appendix, afterword, credits, bibliography, or other relevant information. From this text, extract the following information as accurately as possible:
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
    new_text = ""
    new_text += "START OF BEGINNING TEXT:\n"
    new_text += text[:max_first_word_count]
    new_text += "\nEND OF BEGINNING TEXT\n"
    new_text += "START OF FINAL TEXT:\n"
    new_text += text[-max_last_word_count:]
    new_text += "\nEND OF FINAL TEXT\n"
    return new_text


def extract_metadata(
    file_path, model, max_first_word_count, max_last_word_count, output_file, debug
):
    """
    Extract title, subtitle, series, author, and books3_version_metadata using ChatGPT.
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

    # Extract metadata using ChatGPT
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

    # Create OverallBookMetadata instance with extracted info
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
        original_version_metadata=None,  # Will be filled by enrich_with_openlibrary.py
    )

    # Save to output file
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        f.write(overall_metadata.model_dump_json() + "\n")

    print(f"Saved metadata for {overall_metadata.filename}\n")


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

    print(f"Using API key: {os.getenv('OPENAI_API_KEY')}")

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
