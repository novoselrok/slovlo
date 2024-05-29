import os
import json
from glob import glob
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
from strictfire import StrictFire


def get_pairs_from_article(
    article: Dict[str, Any],
    min_title_length: int = 32,
    min_lead_length: int = 32,
    min_paragraph_length: int = 32,
) -> List[Tuple[str, str, str]]:
    pairs = []

    id_ = article["url"]
    title = article["title"].strip()
    title_subtitle = (title + " " + article["subtitle"]).strip()
    if len(title) >= min_title_length:
        if len(article["lead"]) >= min_lead_length:
            pairs.append((id_, title, article["lead"]))
            if title != title_subtitle:
                pairs.append((id_, title_subtitle, article["lead"]))

        if len(article["article"]) > 0:
            first_paragraph = "\n\n".join(article["article"][0])

            if len(first_paragraph) >= min_paragraph_length:
                pairs.append((id_, title, first_paragraph))
                if title != title_subtitle:
                    pairs.append((id_, title_subtitle, first_paragraph))

    if len(article["article"]) > 0:
        for section in article["article"]:
            if section["header"] is None:
                continue

            title_header = (title + " " + section["header"]).strip()
            paragraph = "\n\n".join(section["paragraphs"])
            if len(paragraph) >= min_paragraph_length:
                pairs.append((id_, title_header, paragraph))

    return pairs


def get_pairs_from_content(
    path: str,
    min_title_length: int = 32,
    min_lead_length: int = 32,
    min_paragraph_length: int = 32,
) -> List[Tuple[str, str, str]]:
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))

    pairs = []
    for article in articles:
        pairs.extend(
            get_pairs_from_article(
                article,
                min_title_length=min_title_length,
                min_lead_length=min_lead_length,
                min_paragraph_length=min_paragraph_length,
            )
        )
    return pairs


def main(
    dir: str,
    output_path: str,
    min_title_length: int = 32,
    min_lead_length: int = 32,
    min_paragraph_length: int = 32,
):
    content_files = glob(os.path.join(dir, "*.jsonl"))

    pairs = []
    for path in tqdm(content_files):
        pairs.extend(
            get_pairs_from_content(
                path,
                min_title_length=min_title_length,
                min_lead_length=min_lead_length,
                min_paragraph_length=min_paragraph_length,
            )
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
