import os
import json
from glob import glob

from strictfire import StrictFire


def main(dir: str, output_path: str):
    files = glob(os.path.join(dir, "*.json"))

    urls = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            urls.extend(json.load(f))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list(set(urls)), f, indent=2)


if __name__ == "__main__":
    StrictFire(main)
