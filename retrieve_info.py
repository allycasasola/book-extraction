from openai import OpenAI
from pydantic import BaseModel
import os
import argparse
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# 30 words per page * 20 pages = 6000 words
MAX_FIRST_WORD_COUNT = 4000
MAX_LAST_WORD_COUNT = 2000
DEFAULT_MODEL = "gpt-5"
DEFAULT_DATA_DIR = f"{os.getenv('DATA_DIR')}/run-books"
DEFAULT_OUTPUT_FILE = f"{os.getenv('DATA_DIR')}/extracted_info.jsonl"


SYSTEM_PROMPT = """
You are a helpful assistant for bibliographic metadata extraction. You will be provided the first 4,000 words and the last 2,000 words of a specific manifestation of a book. This first 4,000 words may include the title page, copyright page, preface, or introduction. The last 2,000 words may include the conclusion, appendix, afterword, credits, bibliography, or other relevant information. From this text, extract, reason, research, and retrieve the following information as accurately as possible.
- Main title: the book's main title (excluding the series name, edition notes, or subtitle, unless part of the main title)
- Subtitle: the book's subtitle (if any)
- Series: the book's series (if any)
- Author: the main author(s)
- Translator: the translator(s) (if any)
- ISBN-13: the ISBN-13 of the manifestation of the book
- ISBN-10: the ISBN-10 of the manifestation of the book
- Year of publication: the year that the manifestation of the book was published
- Status: the copyright or licensing status of the book (e.g. "all rights reserved", "public domain", "cc-by", "cc-by-nc-sa", "cc-by-nd", "cc0", "orphan work", "government work", "fair use")
- Publisher: the name of the publisher of the manifestation of the book

Rules and disambiguation guidelines:
- Use only the provided text plus cautious web search if the text is insufficient. If a field cannot be determined with high confidence, set it to null.
- For ISBN-13 and ISBN-10, do not include the dashes.
- If you can determine the ISBN-13, set the ISBN-10 to null. If you cannot determine the ISBN-13, set the ISBN-13 to null, and look for the ISBN-10. If you cannot find neither, set both to null.
- If there are multiple authors, separate them with commas.
- If there are multiple translators, separate them with commas.
- If there are multiple publishers, separate them with semicolons.
- Never make up information. Only use the information provided in the text and the information retrieved from the web search.
"""


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


class BookInfo(BaseModel):
    title: str | None
    subtitle: str | None
    series: str | None
    author: str | None
    translator: str | None
    year_of_publication: int | None
    status: StatusType | None
    isbn_13: str | None
    isbn_10: str | None
    publisher: str | None


def get_text(text, max_first_word_count, max_last_word_count):
    new_text = ""
    new_text += "START OFBEGINNING TEXT:\n"
    new_text += text[:max_first_word_count]
    new_text += "END OF BEGINNING TEXT\n"
    new_text +=  "START OF FINAL TEXT:\n"
    new_text += text[-max_last_word_count:]
    new_text += "END OF FINAL TEXT\n"
    return new_text

def retrieve_info(file_path, model, max_first_word_count, max_last_word_count, output_file):
    with open(file_path, "r") as f:
        text = f.read()
    text = get_text(text, max_first_word_count, max_last_word_count)

    if len(text) > 6085: # 6000 words + 85 words for the start and end markers
        print(f"Text is too long: {len(text)} words")
        return

    response = client.responses.parse(
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
        text_format=BookInfo,
    )

    info = response.output_parsed

    print(f"Retrieved info from {file_path}...")
    print(info)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        f.write(info.model_dump_json() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max_first_word_count", "-f", type=int, default=MAX_FIRST_WORD_COUNT)
    parser.add_argument("--max_last_word_count", "-l", type=int, default=MAX_LAST_WORD_COUNT)
    parser.add_argument("--output_file", "-o", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--start_index", "-s", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir
    model = args.model
    max_first_word_count = args.max_first_word_count
    max_last_word_count = args.max_last_word_count
    output_file = args.output_file
    start_index = args.start_index
    cur_index = 0
    for file in os.listdir(data_dir):
        print(f"Current index: {cur_index}")
        if file.endswith(".txt") and cur_index >= start_index:
            file_path = os.path.join(data_dir, file)
            retrieve_info(file_path, model, max_first_word_count, max_last_word_count, output_file)
        elif file.endswith(".txt"):
            print(f"Skipping {file}...")
        cur_index += 1

    print("Done retrieving info from all files!")

if __name__ == "__main__":
    main()