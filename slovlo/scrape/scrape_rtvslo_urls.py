import os
import json
import time
from typing import Optional, Dict, Any

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from strictfire import StrictFire

from gather_rtvslo_urls import USER_AGENT


def scrape_rtvslo_article_url(
    url: str,
    num_retries: int = 3,
    base_retry_delay: float = 1,
) -> Optional[Dict[str, Any]]:
    response = None
    retry_delay = base_retry_delay
    for _ in range(num_retries + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT})
            break
        except:
            print(f"Request to '{url}' failed. Retrying ...")
            time.sleep(retry_delay)
            retry_delay *= 2

    if response is None or response.status_code != 200:
        return None

    try:
        soup = BeautifulSoup(response.content, "html.parser")

        title_el = soup.select_one("header h1")
        subtitle_el = soup.select_one("header .subtitle")
        lead_el = soup.select_one("header .lead")
        article_els = soup.select("article h2, article h3, article h4, article p")

        sections = []

        current_header, current_paragraphs = None, []
        for article_el in article_els:
            el_text = article_el.text.strip()
            if len(el_text) == 0:
                continue

            is_paragraph = article_el.name.lower().strip() == "p"
            if is_paragraph:
                current_paragraphs.append(el_text)
            else:
                sections.append(
                    {"header": current_header, "paragraphs": current_paragraphs}
                )
                current_header, current_paragraphs = el_text, []

        if len(current_paragraphs) > 0:
            sections.append(
                {"header": current_header, "paragraphs": current_paragraphs}
            )

        return {
            "url": url,
            "title": title_el.text.strip(),
            "subtitle": subtitle_el.text.strip(),
            "lead": lead_el.text.strip(),
            "article": sections,
        }
    except Exception as e:
        print("Error:", e)
        print(f"Failed to parse url: '{url}'")
        return None


def main(
    urls_path: str,
    output_dir: str,
    output_batch_size: int = 10_000,
    delay_between_requests: float = 0.2,
    request_retries: int = 3,
    base_retry_delay: float = 1,
):
    with open(urls_path, encoding="utf-8") as f:
        urls = json.load(f)

    articles, output_batch = [], 0
    for url in tqdm(urls):
        time.sleep(delay_between_requests)

        article = scrape_rtvslo_article_url(
            url, num_retries=request_retries, base_retry_delay=base_retry_delay
        )

        if article is None:
            print(f"Failed to scrape url: '{url}'")
            continue

        articles.append(article)

        if len(articles) == output_batch_size:
            path = os.path.join(output_dir, f"content_{output_batch}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for article in articles:
                    f.write(json.dumps(article) + "\n")

            articles, output_batch = [], output_batch + 1

    if len(articles) > 0:
        path = os.path.join(output_dir, f"content_{output_batch}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article) + "\n")


if __name__ == "__main__":
    StrictFire(main)
