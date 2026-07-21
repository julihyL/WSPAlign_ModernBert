# coding=utf-8

import argparse
import json
from pathlib import Path


def create_subset(
    input_file: Path,
    output_file: Path,
    max_qas: int,
) -> None:
    """
    Create a smaller SQuAD-format dataset containing at most max_qas questions.
    """

    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if max_qas <= 0:
        raise ValueError("--max_qas must be greater than 0")

    with input_file.open("r", encoding="utf-8") as f:
        original = json.load(f)

    if "data" not in original:
        raise ValueError(
            f"{input_file} does not appear to be a SQuAD-format file: "
            "missing top-level 'data' field."
        )

    subset_articles = []
    selected_qas = 0
    selected_paragraphs = 0

    for article in original["data"]:
        new_article = {
            "title": article.get("title", ""),
            "paragraphs": [],
        }

        for paragraph in article.get("paragraphs", []):
            remaining = max_qas - selected_qas

            if remaining <= 0:
                break

            qas = paragraph.get("qas", [])
            selected = qas[:remaining]

            if not selected:
                continue

            new_paragraph = {
                key: value
                for key, value in paragraph.items()
                if key != "qas"
            }
            new_paragraph["qas"] = selected

            new_article["paragraphs"].append(new_paragraph)

            selected_qas += len(selected)
            selected_paragraphs += 1

        if new_article["paragraphs"]:
            subset_articles.append(new_article)

        if selected_qas >= max_qas:
            break

    subset = {
        key: value
        for key, value in original.items()
        if key != "data"
    }
    subset["data"] = subset_articles

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Input file:           {input_file}")
    print(f"Output file:          {output_file}")
    print(f"Selected articles:    {len(subset_articles)}")
    print(f"Selected paragraphs:  {selected_paragraphs}")
    print(f"Selected QA examples: {selected_qas}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small subset of a SQuAD-format JSON dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the original SQuAD-format JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the smaller JSON file will be written.",
    )

    parser.add_argument(
        "--max_qas",
        required=True,
        type=int,
        help="Maximum number of QA examples to retain.",
    )

    args = parser.parse_args()

    create_subset(
        input_file=args.input,
        output_file=args.output,
        max_qas=args.max_qas,
    )


if __name__ == "__main__":
    main()