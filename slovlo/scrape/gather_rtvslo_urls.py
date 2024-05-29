import os
import json
import time
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from strictfire import StrictFire

USER_AGENT = "slovlo-bot/1.0"

BASE_URL = "https://www.rtvslo.si"
ARCHIVE_CATEGORIES = [
    "evropska-unija",
    "gospodarstvo",
    "sport",
    "znanost-in-tehnologija",
    "lokalne-novice",
    "zdravje",
    "okolje",
    "slovenija",
    "crna-kronika",
    "svet",
    "kultura",
    "zabava-in-slog",
    "posebna-izdaja",
]


def gather_page_urls(
    base_url: str,
    page_num: int,
    num_retries: int = 0,
    base_retry_delay: float = 0.5,
) -> Optional[List[str]]:
    response = None
    retry_delay = base_retry_delay
    for _ in range(num_retries + 1):
        try:
            url = f"{base_url}?page={page_num}"
            response = requests.get(url, headers={"User-Agent": USER_AGENT})
            break
        except:
            print(f"Request to '{base_url}', page {page_num} failed. Retrying ...")
            time.sleep(retry_delay)
            retry_delay *= 2

    if response is None or response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    a_tags = soup.select(".article-archive-item h3 > a")
    links = [a.get("href") for a in a_tags if a.get("href")]
    return [f"{BASE_URL}{link}" for link in links]


def main(
    output_dir: str,
    delay_between_requests: float = 0.2,
    request_retries: int = 5,
    base_retry_delay: float = 1,
):
    for category in ARCHIVE_CATEGORIES:
        archive_url = f"{BASE_URL}/{category}/arhiv/"

        urls = []
        page_num = 0
        while True:
            print(f"Scraping '{archive_url}', page {page_num} ...")

            page_urls = gather_page_urls(
                archive_url,
                page_num,
                num_retries=request_retries,
                base_retry_delay=base_retry_delay,
            )
            if page_urls is None:
                break

            urls.extend(page_urls)
            page_num += 1

            time.sleep(delay_between_requests)

        with open(
            os.path.join(output_dir, f"{category}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(list(set(urls)), f)


if __name__ == "__main__":
    StrictFire(main)
