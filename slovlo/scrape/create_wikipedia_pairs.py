import os
import re
import json
from io import TextIOWrapper
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
from bs4 import BeautifulSoup
from strictfire import StrictFire

element_selectors = ["p", "ul", "ol", "h2", "h3", "h4", "h5", "h6"]
content_selectors = [f"#mw-content-text section > {es}" for es in element_selectors]
content_selector = ", ".join(content_selectors)

ignore_section_with_header_prefix = [
    "glej tudi",
    "vir",
    "literatura",
    "sklic",
    "zunanje povezave",
    "galerija",
]


def replace_repeating_newlines(text: str) -> str:
    return re.sub(r"\n\n\n+", "\n\n", text)


def replace_repeating_spaces(text: str) -> str:
    return re.sub(r" +", " ", text)


def get_pairs_from_article(
    soup: BeautifulSoup, min_paragraph_length=32
) -> List[Tuple[str, str, str]]:
    title_el = soup.select_one("title")
    if title_el is None:
        return []

    title = title_el.text.strip()

    # Remove citations
    for sup_el in soup.select("sup"):
        sup_el.decompose()

    content_els = soup.select(content_selector)

    sections = []
    current_header, current_paragraphs = None, []
    for content_el in content_els:
        tag, text = content_el.name.lower().strip(), content_el.text.strip()

        is_paragraph = tag in ["p", "ul", "ol"]
        if is_paragraph:
            current_paragraphs.append(text)
        else:
            sections.append(
                {"header": current_header, "paragraphs": current_paragraphs}
            )
            current_header, current_paragraphs = text, []

    if len(current_paragraphs) > 0:
        sections.append({"header": current_header, "paragraphs": current_paragraphs})

    if len(sections) == 0:
        return []

    pairs = []
    for section in sections:
        if len(section["paragraphs"]) == 0:
            continue

        if section["header"] is None:
            section_title = title
        else:
            invalid_header = False
            for ignore_prefix in ignore_section_with_header_prefix:
                if section["header"].lower().startswith(ignore_prefix):
                    invalid_header = True
                    break

            if invalid_header:
                continue

            section_title = title + " " + section["header"]

        paragraph = replace_repeating_spaces(
            replace_repeating_newlines("\n\n".join(section["paragraphs"]))
        )
        if len(paragraph) < min_paragraph_length:
            continue

        pairs.append((title, section_title, paragraph))

    return pairs


def get_pairs_from_articles(
    paths: List[str], min_paragraph_length=32
) -> List[Tuple[str, str, str]]:
    pairs = []
    for article_path in tqdm(paths):
        with open(article_path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            pairs.extend(
                get_pairs_from_article(soup, min_paragraph_length=min_paragraph_length)
            )
    return pairs


def find_files(dir: str) -> List[str]:
    files_list = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            files_list.append(os.path.join(root, file))

    return files_list


def main(
    dir: str,
    output_path: str,
    min_paragraph_length: int = 32,
):
    pairs = get_pairs_from_articles(
        find_files(dir), min_paragraph_length=min_paragraph_length
    )
    pairs = [
        {"id": id_, "query": query, "document": document}
        for (id_, query, document) in pairs
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    StrictFire(main)
