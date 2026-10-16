"""Bounded query-adaptive evidence selection; never generates an answer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re


STOP = set("a an the is are was were do does did i you my your me we it that this to of in on at and or for with what which who when where how please show tell recall remember image photo picture".split())
ALIASES = (("bike", "bicycle", "cycling"), ("car", "automobile"), ("sofa", "couch"),
           ("purchase", "bought", "buy"), ("visit", "visited", "went"),
           ("prefer", "preference", "favorite", "favourite"), ("repair", "fixed", "fix"))
COLORS = {"red": [1, 0, 0], "blue": [0, 0, 1], "green": [0, .5, 0],
          "yellow": [1, 1, 0], "white": [1, 1, 1], "black": [0, 0, 0],
          "purple": [.5, 0, .5], "orange": [1, .65, 0]}


def tokens(text, *, expand=False):
    words = re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.casefold())
    result = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            result.extend(word[i:i+2] for i in range(max(1, len(word)-1)))
        elif word not in STOP:
            result.append(word)
    if expand:
        present = set(result)
        for group in ALIASES:
            if present.intersection(group):
                result.extend(group)
    return list(dict.fromkeys(result))


def fts_query(words):
    # FTS operators from input are never interpreted as query syntax.
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words[:128])


def query_plan(text, options, images):
    words = tokens(text, expand=True)
    option_words = tokens(" ".join(options or []), expand=True)
    lower = text.casefold()
    dates = []
    for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text):
        try:
            start = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            dates.append((int(start.timestamp()*1000), int((start+timedelta(days=1)).timestamp()*1000)))
        except (ValueError, OverflowError):
            pass
    return {"words": words, "options": option_words, "images": images, "dates": dates,
            "before": bool(re.search(r"\b(before|previous|preceding)\b|之前|前一个", lower)),
            "after": bool(re.search(r"\b(after|next|following)\b|之后|下一个", lower)),
            "latest": bool(re.search(r"\b(latest|current|currently|now|recent)\b|目前|现在|最新", lower)),
            "multi": bool(re.search(r"\b(both|between|compare|changed|changes|all|summary|summarize)\b|变化|总结|比较", lower)),
            "colors": [value for name, value in COLORS.items() if name in words]}


def visual_similarity(query, record):
    if query["digest"] == record["digest"]:
        return 1.0
    a, b = query.get("features", {}), record.get("features", {})
    if "hist" not in a or "hist" not in b:
        return 0.0
    hamming = 1 - (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()/64
    histogram = sum(min(x, y) for x, y in zip(a["hist"], b["hist"]))
    return .25*hamming + .75*histogram


def rank_evidence(rows, sparse_ids, option_ids, plan, *, strategy, top_k, ablation="none"):
    """RRF candidates, source term coverage and optional evidence-complement expansion.

    The algorithm uses only source text/metadata; labels and future queries are absent.
    Expansion is bounded to temporal neighbours of high-ranking anchors in the same session.
    """
    ranks = {mid: i for i, mid in enumerate(sparse_ids)}
    oranks = {mid: i for i, mid in enumerate(option_ids)}
    ordered = sorted(rows, key=lambda row: row["sequence"])
    scores = {}
    max_sequence = max((row["sequence"] for row in rows), default=1)
    source_times = [row["timestamp"] for row in rows if row["timestamp"] is not None]
    earliest_time, latest_time = min(source_times,default=0), max(source_times,default=0)
    for row in rows:
        mid, words = row["id"], set(row["tokens"])
        if strategy == "raw":
            score = 1 / (1 + max_sequence - row["sequence"])
        else:
            score = 1/(60+ranks[mid]) if mid in ranks else 0.0
            score += .25/(60+oranks[mid]) if mid in oranks else 0.0
            if strategy == "adaptive":
                overlap = len(words.intersection(plan["words"])) / max(1, len(set(plan["words"])))
                score += .035*overlap
                if plan["images"] and row["images"]:
                    sim = sum(max(visual_similarity(q, r) for r in row["images"]) for q in plan["images"])/len(plan["images"])
                    score += .065*sim
                if plan["colors"] and row["images"]:
                    color = max(1-math.sqrt(sum((a-b)**2 for a,b in zip(target, image["features"].get("mean", [0,0,0]))))/math.sqrt(3)
                                for target in plan["colors"] for image in row["images"])
                    score += .01*color
                if row["timestamp"] is not None and any(a <= row["timestamp"] < b for a,b in plan["dates"]):
                    score += .04
                if plan["latest"] and overlap:
                    recency = ((row["timestamp"]-earliest_time)/max(1,latest_time-earliest_time)
                               if row["timestamp"] is not None else 0) if source_times else row["sequence"]/max_sequence
                    score += .025 * recency
        scores[mid] = score
    if strategy == "adaptive" and ablation != "no_neighbours" and (plan["before"] or plan["after"] or plan["multi"]):
        anchors = sorted(rows, key=lambda row: (-scores[row["id"]], row["sequence"]))[:3]
        neighbours = []
        for anchor in anchors:
            same = [row for row in ordered if row["session"] == anchor["session"]]
            # Source timestamps dominate when both records have timestamps; sequence breaks ties.
            same.sort(key=lambda row: (row["timestamp"] if row["timestamp"] is not None else 0, row["sequence"]))
            pos = next(i for i,row in enumerate(same) if row["id"] == anchor["id"])
            shifts = ([-1] if plan["before"] else []) + ([1] if plan["after"] else [])
            if plan["multi"]:
                shifts += [-1, 1]
            for shift in shifts:
                if 0 <= pos+shift < len(same):
                    neighbours.append((same[pos+shift]["id"], scores[anchor["id"]] * .95))
        for mid, bonus in neighbours:
            scores[mid] = max(scores[mid], bonus)
    ranked = sorted(rows, key=lambda row: (-scores[row["id"]], row["sequence"], row["id"]))
    if strategy != "adaptive" or not plan["multi"] or ablation == "no_coverage":
        return [(row, scores[row["id"]]) for row in ranked[:top_k]]
    # Diminishing reward for source terms already covered; do not eliminate identical
    # pictures/text with different timestamps (they may be distinct events).
    chosen, covered, sessions = [], set(), set()
    pool = ranked[:max(200, top_k*3)]
    while pool and len(chosen) < top_k:
        row = max(pool, key=lambda r: scores[r["id"]] + .004*len(set(r["tokens"]).intersection(plan["words"])-covered)
                  + (.001 if r["session"] not in sessions else 0))
        pool.remove(row)
        chosen.append((row, scores[row["id"]]))
        covered.update(row["tokens"])
        sessions.add(row["session"])
    return chosen
