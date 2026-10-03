#!/usr/bin/env python3
"""Evaluate saved LVLM recommendation outputs without calling a model API.

Generated titles are mapped to the candidate pool with a conservative,
target-independent rule.  Failed and missing outputs remain in the evaluation
cohort and receive zero credit.
"""

import argparse
import ast
import csv
import html
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


CANDIDATE_MARKERS = (
    re.compile(r"candidate items in the item pool\s*:", re.IGNORECASE),
    re.compile(
        r"pre-ranked item recommendation sequence.*?from highest to lowest\s*:",
        re.DOTALL | re.IGNORECASE,
    ),
)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def normalize_title(value):
    """Normalize harmless Unicode, HTML, quote, dash, case, and space differences."""
    value = unicodedata.normalize("NFKC", html.unescape(str(value)))
    value = value.translate(
        str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
    )
    return re.sub(r"\s+", " ", value.casefold()).strip()


def canonical_tokens(value):
    """Tokenize a title while treating punctuation and symbols as formatting."""
    return re.findall(r"\w+", normalize_title(value), flags=re.UNICODE)


def conservative_alias(left, right):
    """Accept formatting variants or an informative title truncation.

    A truncation must contain at least four tokens and twenty characters.  This
    intentionally rejects semantic guesses between different models, variants,
    colors, sizes, editions, or product names.
    """
    left_tokens, right_tokens = canonical_tokens(left), canonical_tokens(right)
    if left_tokens == right_tokens:
        return True
    shorter, longer = sorted((left_tokens, right_tokens), key=len)
    if len(shorter) < 4 or len(" ".join(shorter)) < 20:
        return False
    return shorter == longer[: len(shorter)]


def _list_literal_after(prompt, marker):
    """Return a balanced Python list literal following a prompt marker."""
    start = prompt.find("[", marker.end())
    if start < 0:
        raise ValueError("candidate list not found")

    depth = 0
    quote = None
    escaped = False
    for index in range(start, len(prompt)):
        character = prompt[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in ("'", '"'):
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return prompt[start : index + 1]
    raise ValueError("unterminated candidate list")


def parse_candidates(prompt):
    """Extract candidates from direct-ranking and pre-ranked reranking prompts."""
    marker = None
    for pattern in CANDIDATE_MARKERS:
        marker = pattern.search(prompt)
        if marker is not None:
            break
    if marker is None:
        raise ValueError("candidate list not found")
    candidates = ast.literal_eval(_list_literal_after(prompt, marker))
    if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
        raise ValueError("invalid candidate list")
    return [html.unescape(item) for item in candidates]


def parse_recommendations(info):
    """Return recommendation titles and a status without mutating the input."""
    response = info.get("api_response", {})
    if isinstance(response, str):
        if not response.strip():
            return [], "empty_response"
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            fenced = FENCED_JSON_RE.search(response)
            if fenced is None:
                return [], "json_error"
            try:
                response = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                return [], "json_error"
    if not isinstance(response, dict):
        return [], "invalid_response"
    recommendations = response.get("recommendations", [])
    if not isinstance(recommendations, list) or not recommendations:
        return [], "empty_recommendations"
    return [html.unescape(str(item)) for item in recommendations], "ok"


def map_to_candidate(generated, candidates):
    """Map one generated title without using knowledge of the target item."""
    generated_norm = normalize_title(generated)
    exact = [
        index
        for index, candidate in enumerate(candidates)
        if normalize_title(candidate) == generated_norm
    ]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(exact) > 1:
        return None, "ambiguous"

    aliases = [
        index
        for index, candidate in enumerate(candidates)
        if conservative_alias(generated, candidate)
    ]
    if len(aliases) == 1:
        return aliases[0], "conservative"
    return None, "ambiguous"


def metric_values(rank, topks):
    values = {}
    for topk in topks:
        hit = float(rank is not None and rank <= topk)
        ndcg = 1.0 / math.log2(rank + 1) if hit else 0.0
        values[f"hits_at_{topk}_pct"] = hit
        values[f"ndcg_at_{topk}_pct"] = ndcg
    return values


def evaluate_file(path, cohort_size=400, topks=(1, 3, 5, 10, 20, 30)):
    """Evaluate one processed_data.json file using a fixed cohort denominator."""
    data = json.loads(Path(path).read_text())
    if len(data) > cohort_size:
        raise ValueError(f"stored users ({len(data)}) exceed cohort size ({cohort_size})")

    totals = Counter()
    statuses = Counter()
    mapping_statuses = Counter()

    for info in data.values():
        recommendations, response_status = parse_recommendations(info)
        statuses[response_status] += 1
        if not recommendations:
            continue

        try:
            candidates = parse_candidates(info["prompt"])
        except (KeyError, ValueError, SyntaxError):
            statuses["candidate_parse_error"] += 1
            continue

        target = html.unescape(info["target"]["titles"][0])
        target_matches = [
            index
            for index, candidate in enumerate(candidates)
            if normalize_title(candidate) == normalize_title(target)
        ]
        if len(target_matches) != 1:
            statuses["target_not_unique"] += 1
            continue
        target_index = target_matches[0]

        rank = None
        for position, generated in enumerate(recommendations, start=1):
            candidate_index, mapping_status = map_to_candidate(generated, candidates)
            mapping_statuses[mapping_status] += 1
            # Every generated entry retains its original position.  Ambiguous
            # and duplicate outputs are never removed or allowed to shift a
            # later target upward.
            if candidate_index == target_index and rank is None:
                rank = position

        totals.update(metric_values(rank, topks))

    dataset = next(
        (part for part in Path(path).parts if re.fullmatch(r"(beauty|clothing|sports|toys)_\d+", part)),
        "unknown",
    )
    template_model = Path(path).parent.name.removeprefix("prompts_")
    template, _, model = template_model.partition("_")
    row = {
        "dataset": dataset,
        "template": template,
        "model": model,
        "cohort_size": cohort_size,
        "stored_users": len(data),
        "missing_users": cohort_size - len(data),
        **{f"status_{key}": value for key, value in sorted(statuses.items())},
        **{f"mapping_{key}": value for key, value in sorted(mapping_statuses.items())},
    }
    for key in (f"{metric}_at_{topk}_pct" for topk in topks for metric in ("hits", "ndcg")):
        row[key] = 100.0 * totals[key] / cohort_size
    return row


def evaluate_directory(root, cohort_size=400, topks=(1, 3, 5, 10, 20, 30)):
    paths = sorted(Path(root).rglob("processed_data.json"))
    if not paths:
        raise ValueError(f"no processed_data.json files found under {root}")
    return [evaluate_file(path, cohort_size=cohort_size, topks=topks) for path in paths]


def write_csv(rows, output):
    fields = sorted({key for row in rows for key in row})
    with Path(output).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing processed_data.json files")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--cohort-size", type=int, default=400)
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 3, 5, 10, 20, 30])
    args = parser.parse_args()

    topks = tuple(sorted(set(args.topks)))
    if args.cohort_size <= 0 or not topks or topks[0] <= 0:
        parser.error("cohort size and top-k values must be positive")
    rows = evaluate_directory(args.root, cohort_size=args.cohort_size, topks=topks)
    write_csv(rows, args.output)
    print(f"evaluated {len(rows)} files -> {args.output}")


if __name__ == "__main__":
    main()
