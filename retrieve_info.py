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
- Year of publication: the year that the book was published
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
"""

WORK_FIELDS = "title,author_key,author_name,author_alternative_name,first_publish_year,publish_date,publish_year"

# TODO: Add a retry mechanism
def _fetch_works(title: str, author: str) -> dict | None:
    # Convert title and author to url encoded
    params = {
        "title": title,
        "author": author,
    }
    title = urllib.parse.quote(title)
    author = urllib.parse.quote(author)
    # Limit to only 100 results max
    url= f"https://openlibrary.org/search/works.json?{urlencode(params)}&fields={WORK_FIELDS}&limit=100"
    response = requests.get(url)
    return response.json()


def _parse_edtf_date(edtf_date: str) -> date | None:
    try:
        parts = edtf_date.split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except Exception as e:
        print(f"Error parsing EDTF date: {e}")
        return None


def _fetch_all_author_names(author_key: str) -> list[str]:
    url= f"https://openlibrary.org/authors/{author_key}.json"
    response = requests.get(url)
    all_names = response.json()["name"]
    all_names.extend(response.json()["alternate_names"])
    all_names.extend(response.json()["fuller_name"])
    all_names.extend(response.json()["personal_name"])
    return all_names


def filter_works(title: str, author: str) -> list[dict] | None:
    works = _fetch_works(title, author)
    works_filtered = []
    if works is None or works["numFound"] == 0:
        return None
    # Filter works whose title/alternative_title matches the title (case insensitive) AND author_name/alternative_name matches the author (case insensitive)
    for work in works["docs"]:
        author_names = _fetch_all_author_names(work["author_key"])
        if work["title"].lower() == title.lower() and any(author_name.lower() == author.lower() for author_name in author_names):
            works_filtered.append(work)
    return works_filtered


def _find_work_key_with_earliest_publication_year(works: list[dict]) -> str | None:
    earliest_publication_year = float('inf')
    earliest_work = None
    for work in works:
        if work["first_publish_year"] < earliest_publication_year:
            earliest_publication_year = work["first_publish_year"]
            earliest_work = work
    return earliest_work["key"]


def fetch_all_editions(work_key: str) -> list[dict]:
    url= f"https://openlibrary.org/works/{work_key}/editions.json"
    response = requests.get(url)
    return response.json()["entries"]


def _find_edition_with_earliest_publication(editions: list[dict]) -> dict | None:
    earliest_publication_date = date(9999, 12, 31)
    earliest_edition = None
    for edition in editions:
        publication_date = _parse_edtf_date(edition["publish_date"])
        if publication_date is not None and publication_date < earliest_publication_date:
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


class OpenAIUsage():
    input_tokens: int
    output_tokens: int
    num_books: int


class OverallBookMetadata():
    version_specific_book_metadata: BookMetadata | None
    original_book_metadata: BookMetadata | None


class BookMetadata(BaseModel):
    filename: str | None
    title: str | None
    subtitle: str | None
    series: str | None
    author: str | None
    translator: str | None
    isbn_13: str | None
    isbn_10: str | None
    year_of_publication: int | None
    status: StatusType | None
    publisher: str | None


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
    with open(file_path, "r") as f:
        text = f.read()
    text = format_text(text, max_first_word_count, max_last_word_count)

    if debug:
        print(text)

    if len(text) > MAX_FIRST_WORD_COUNT + MAX_LAST_WORD_COUNT + 100:
        print(f"Text is too long: {len(text)} words. Skipping...")
        return

    response = OPENAI_CLIENT.responses.parse(
        model=model,
        tools=[
            {
                "type": "web_search",
            }
        ],
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=BookMetadata,
    )

    if hasattr(response, "usage"):
        input_tokens = getattr(response.usage, "input_tokens", 0)
        print(f"Input tokens: {input_tokens}")
        output_tokens = getattr(response.usage, "output_tokens", 0)
        print(f"Output tokens: {output_tokens}")

    metadata = response.output_parsed

    # Extract just the filename (without path) and add it to the parsed info
    filename = os.path.basename(file_path)
    metadata.filename = filename

    print(f"Retrieved metadata from {file_path}...")
    print(metadata)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        f.write(metadata.model_dump_json() + "\n")


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
    parser.add_argument("--debug", "-d", type=bool, default=False)
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
                file_path, model, max_first_word_count, max_last_word_count, output_file, print_text
            )
        else:
            print(f"Skipping {file}...")

    print("Done retrieving metadata from all files!")


if __name__ == "__main__":
    main()
