import os
import json
import html
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from strictfire import StrictFire


def clean_text(text: str) -> str:
    t = text.strip()
    t = html.unescape(text)
    t = t.replace("&#x200B;", "")
    return t


def find_title_body_pairs(
    entities: Dict[str, Any], min_body_length: int
) -> List[Tuple[str, str, str]]:
    pairs = []
    for _, entity in entities.items():
        if entity["type"] != "self":
            continue

        if len(entity["body"]) < min_body_length:
            continue

        pairs.append((entity["id"], entity["title"], entity["body"]))

    return pairs


def find_submission_comment_pairs(
    entities: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    submission_comments = defaultdict(list)

    for id, entity in entities.items():
        if entity["type"] != "comment":
            continue

        parent_type, parent_id = entity["parent_id"].split("_")
        if parent_type != "t3":
            continue

        if parent_id not in entities:
            continue

        submission_comments[parent_id].append(id)

    pairs = []
    for submission_id, comment_ids in submission_comments.items():
        submission = entities[submission_id]
        comments = [entities[comment_id] for comment_id in comment_ids]

        query = (submission["title"] + " " + submission.get("body", "")).strip()

        for comment in comments:
            pairs.append((submission_id, query, comment["body"]))

    return pairs


def find_comment_comment_pairs(entities: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    top_level_comments = {}

    for id, entity in entities.items():
        if entity["type"] != "comment":
            continue

        parent_type, _ = entity["parent_id"].split("_")
        if parent_type != "t3":
            continue

        top_level_comments[id] = []

    for id, entity in entities.items():
        if entity["type"] != "comment":
            continue

        parent_type, parent_id = entity["parent_id"].split("_")
        if parent_type != "t1":
            continue

        if parent_id not in top_level_comments:
            continue

        top_level_comments[parent_id].append(id)

    pairs = []
    for top_level_comment_id, reply_comment_ids in top_level_comments.items():
        if len(reply_comment_ids) == 0:
            continue

        top_level_comment = entities[top_level_comment_id]
        reply_comments = [entities[comment_id] for comment_id in reply_comment_ids]

        for reply_comment in reply_comments:
            pairs.append(
                (top_level_comment_id, top_level_comment["body"], reply_comment["body"])
            )

    return pairs


def main(
    dir: str,
    output_path: str,
    min_title_length: int = 32,
    min_self_body_length: int = 32,
    min_comment_body_length: int = 32,
):
    submissions_file, comments_file = (
        os.path.join(dir, "Slovenia_submissions"),
        os.path.join(dir, "Slovenia_comments"),
    )

    submissions = []
    with open(submissions_file, encoding="utf-8") as f:
        for line in f:
            submissions.append(json.loads(line))

    comments = []
    with open(comments_file, encoding="utf-8") as f:
        for line in f:
            comments.append(json.loads(line))

    entities = {}

    for submission in submissions:
        if submission["score"] <= 1:
            continue

        is_self = submission["is_self"]
        if is_self:
            body = submission["selftext"].strip()
            if body.lower() in ["[deleted]", "[removed]"]:
                continue

            body = "\n".join(
                [
                    line
                    for line in body.split("\n")
                    if not line.strip().startswith("[View Poll]")
                ]
            )

            clean_submission = {
                "id": submission["id"],
                "title": clean_text(submission["title"]),
                "body": clean_text(body),
                "type": "self",
            }
        else:
            clean_submission = {
                "id": submission["id"],
                "title": clean_text(submission["title"]),
                "type": "link",
            }

        if len(submission["title"]) < min_title_length:
            continue

        if submission["title"].lower() in ["[deleted]", "[removed]"]:
            continue

        entities[clean_submission["id"]] = clean_submission

    for comment in comments:
        if comment["score"] <= 1:
            continue

        body = comment["body"].strip()
        if body.lower() in ["[deleted]", "[removed]"]:
            continue

        if "[gif]" in body:
            continue

        body = "\n".join(
            [line for line in body.split("\n") if not line.startswith("http")]
        )

        if len(body) < min_comment_body_length:
            continue

        if not isinstance(comment["parent_id"], str):
            continue

        clean_comment = {
            "id": comment["id"],
            "parent_id": comment["parent_id"],
            "body": clean_text(body),
            "type": "comment",
        }

        entities[clean_comment["id"]] = clean_comment

    title_body_pairs = find_title_body_pairs(
        entities, min_body_length=min_self_body_length
    )
    submission_comment_pairs = find_submission_comment_pairs(entities)
    comment_comment_pairs = find_comment_comment_pairs(entities)

    pairs = title_body_pairs + submission_comment_pairs + comment_comment_pairs
    pairs = [
        {"id": id_, "query": query, "document": document}
        for (id_, query, document) in pairs
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    StrictFire(main)
