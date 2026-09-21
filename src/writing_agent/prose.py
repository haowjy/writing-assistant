"""Versioned prose features and numerical comparisons; optional models load explicitly."""

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, median

from writing_agent.catalog import fingerprint, save_json
from writing_agent.scoring import measurement


@dataclass(frozen=True)
class FeatureConfig:
    tokenizer: str = "google/gemma-4-12B"
    tokenizer_revision: str = "023679ed352de9bb66cc873c9009ce3482585c08"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_revision: str = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    chunk_tokens: int = 256
    version: int = 2


def ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def distribution_l2(generated: list[list], reference: list[list], n: int) -> float | None:
    p, q = Counter(), Counter()
    for text in generated:
        p.update(ngrams(text, n))
    for text in reference:
        q.update(ngrams(text, n))
    m, r = sum(p.values()), sum(q.values())
    if not m or not r:
        return None
    return math.sqrt(sum((p[g] / m - q[g] / r) ** 2 for g in p.keys() | q.keys()))


def _vectors(vectors: list[list[float]]) -> None:
    if not vectors or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("Embeddings must be nonempty and have equal dimensions")
    if any(not math.isfinite(x) for v in vectors for x in v):
        raise ValueError("Embeddings must be finite")


def bandwidth(reference: list[list[float]]) -> float:
    _vectors(reference)
    distances = [math.dist(a, b) for a, b in combinations(reference, 2) if a != b]
    if not distances:
        raise ValueError("Bandwidth needs distinct development reference embeddings")
    return median(distances)


def mmd_squared(generated: list[list[float]], reference: list[list[float]], sigma: float):
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("RBF bandwidth must be positive and finite")
    if len(generated) < 2 or len(reference) < 2:
        return None
    _vectors(generated + reference)

    def kernel(a, b):
        return math.exp(-sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / (2 * sigma**2))

    m, r = len(generated), len(reference)
    return (
        2 * sum(kernel(a, b) for a, b in combinations(generated, 2)) / (m * (m - 1))
        + 2 * sum(kernel(a, b) for a, b in combinations(reference, 2)) / (r * (r - 1))
        - 2 * sum(kernel(a, b) for a in generated for b in reference) / (m * r)
    )


def dispersion(vectors: list[list[float]]) -> float | None:
    if len(vectors) < 2:
        return None
    _vectors(vectors)
    if any(not any(v) for v in vectors):
        raise ValueError("Cosine distance undefined for a zero vector")
    return mean(
        1
        - sum(x * y for x, y in zip(a, b, strict=True))
        / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
        for a, b in combinations(vectors, 2)
    )


def self_bleu(samples: list[list], order: int = 4) -> float | None:
    """Sentence BLEU, add-one smoothing for all orders, closest reference length."""
    if len(samples) < 2 or any(not sample for sample in samples):
        return None
    values = []
    for i, hypothesis in enumerate(samples):
        references = samples[:i] + samples[i + 1 :]
        precisions = []
        for n in range(1, order + 1):
            counts = ngrams(hypothesis, n)
            maximum = Counter()
            for reference in references:
                maximum |= ngrams(reference, n)
            precisions.append((sum((counts & maximum).values()) + 1) / (sum(counts.values()) + 1))
        length = min((len(ref) for ref in references), key=lambda x: (abs(x - len(hypothesis)), x))
        penalty = math.exp(min(0, 1 - length / len(hypothesis)))
        values.append(penalty * math.exp(mean(math.log(p) for p in precisions)))
    return mean(values)


def lexical_features(text: str) -> dict:
    words = re.findall(r"\b\w+(?:['’]\w+)*\b", text.casefold())
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    openings = [" ".join(s.casefold().split()[:3]) for s in sentences]
    counts = {n: ngrams(words, n) for n in (1, 2, 3)}
    tri = counts[3]
    return {
        "word_count": len(words),
        "word_tokenizer": "unicode-word-casefold-v1",
        "distinct": {str(n): len(c) / sum(c.values()) if c else None for n, c in counts.items()},
        "type_token_ratio": len(set(words)) / len(words) if words else None,
        "sentence_lengths": [len(s.split()) for s in sentences],
        "paragraph_lengths": [len(p.split()) for p in paragraphs],
        "duplicate_paragraph_rate": (len(paragraphs) - len(set(paragraphs))) / len(paragraphs)
        if paragraphs
        else None,
        "repeated_opening_rate": (len(openings) - len(set(openings))) / len(openings)
        if openings
        else None,
        "repeated_trigram_rate": sum(c - 1 for c in tri.values()) / sum(tri.values())
        if tri
        else None,
    }


def overlap(tokens: list, source: list, n: int = 3) -> float | None:
    counts, reference = ngrams(tokens, n), ngrams(source, n)
    return (
        sum(count for gram, count in counts.items() if gram in reference) / sum(counts.values())
        if counts
        else None
    )


class ProseFeatures:
    """Explicit local-model loading and artifact/config keyed feature caching.

    Construction does no model work. Missing dependencies/weights are surfaced as
    unavailable features. allow_download must be requested at method call time.
    """

    def __init__(self, cache: Path, config: FeatureConfig | None = None):
        self.cache = cache
        self.config = config or FeatureConfig()
        self._tokenizer = None
        self._embedding = None
        self._embedding_tokenizer = None

    def tokens(self, text: str, *, allow_download=False) -> list[int]:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer,
                revision=self.config.tokenizer_revision,
                local_files_only=not allow_download,
            )
        return self._tokenizer.encode(text, add_special_tokens=False)

    def embedding(self, text: str, *, allow_download=False) -> list[float]:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._embedding is None:
            args = {
                "revision": self.config.embedding_revision,
                "local_files_only": not allow_download,
            }
            self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model, **args
            )
            self._embedding = AutoModel.from_pretrained(self.config.embedding_model, **args).eval()
        tokenizer = self._embedding_tokenizer
        if not text.strip():
            raise ValueError("Cannot embed empty prose")
        special = tokenizer.num_special_tokens_to_add(pair=False)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self.config.chunk_tokens + special,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        weighted, total = None, 0
        for i in range(len(encoded["input_ids"])):
            inputs = {k: encoded[k][i : i + 1] for k in ("input_ids", "attention_mask")}
            count = int(inputs["attention_mask"].sum()) - special
            if count <= 0:
                continue
            with torch.no_grad():
                hidden = self._embedding(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                vector = (hidden * mask).sum(1) / mask.sum(1)
            weighted = vector * count if weighted is None else weighted + vector * count
            total += count
        if total == 0:
            raise ValueError("Cannot embed empty prose")
        vector = weighted / total
        return torch.nn.functional.normalize(vector, p=2, dim=1)[0].tolist()

    def extract(self, text: str, *, tokens=False, embeddings=False, allow_download=False) -> dict:
        config = asdict(self.config)
        key = fingerprint({"text": text, "config": config})
        path = self.cache / (key + ".json")
        record = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "prose_hash": fingerprint(text),
                "config": config,
                "lexical": lexical_features(text),
            }
        )
        for name, enabled, function in [
            ("tokens", tokens, self.tokens),
            ("embedding", embeddings, self.embedding),
        ]:
            if enabled and name not in record:
                try:
                    record[name] = function(text, allow_download=allow_download)
                    record.pop(name + "_error", None)
                except (ImportError, OSError, ValueError, RuntimeError) as exc:
                    record[name + "_error"] = f"{type(exc).__name__}: {exc}"
        save_json(path, record)
        return record


def prose_profile(
    texts: list[str],
    features: list[dict],
    *,
    references: list[dict] | None = None,
    sigma: float | None = None,
    repeated_prompt: bool = False,
    samples: int | None = None,
    prompt_tokens: list | None = None,
    source_tokens: list | None = None,
) -> dict:
    if len(texts) != len(features):
        raise ValueError("Text/features lengths differ")
    references = references or []
    if any(f["prose_hash"] != fingerprint(t) for t, f in zip(texts, features, strict=True)):
        raise ValueError("Features do not match selected prose")
    configs = {fingerprint(f["config"]) for f in features + references}
    if len(configs) > 1:
        raise ValueError("Feature configurations cannot be pooled")
    profile = {
        f"D{i}": measurement(status="not_applicable", reason="Inputs not provided")
        for i in range(1, 14)
    }
    gt = [f["tokens"] for f in features if "tokens" in f]
    rt = [f["tokens"] for f in references if "tokens" in f]
    ge = [f["embedding"] for f in features if "embedding" in f]
    re_ = [f["embedding"] for f in references if "embedding" in f]
    if references and len(gt) == len(features) and len(rt) == len(references) and gt:
        distances = {str(n): distribution_l2(gt, rt, n) for n in (1, 2, 3)}
        profile["D1"] = measurement(distances)
    else:
        profile["D1"] = measurement(
            status="missing_reference" if not references else "unavailable",
            reason="Complete pinned-tokenizer features and reference required",
        )
    if sigma and len(ge) == len(features) and len(re_) == len(references):
        value = mmd_squared(ge, re_, sigma)
        profile["D2"] = measurement(
            value,
            status="ok" if value is not None else "insufficient_samples",
            method="learned_representation",
            sigma=sigma,
            reason=None
            if value is not None
            else "At least two independent outputs and references required",
        )
    else:
        profile["D2"] = measurement(
            status="unavailable",
            method="learned_representation",
            reason="Complete embeddings and frozen bandwidth required",
        )
    for key, reason in {
        "D3": "Larger matched corpus and MAUVE feature configuration required",
        "D12": "Candidate log-probability backend required",
        "D13": "Authentic author edits required",
    }.items():
        profile[key] = measurement(status="not_implemented", reason=reason)
    if features:
        profile["D5"] = measurement(
            {
                str(n): mean(values)
                if (
                    values := [
                        f["lexical"]["distinct"][str(n)]
                        for f in features
                        if f["lexical"]["distinct"][str(n)] is not None
                    ]
                )
                else None
                for n in (1, 2, 3)
            },
            aggregation="artifact_macro",
            tokenizer="unicode-word-casefold-v1",
        )
        pooled = [re.findall(r"\b\w+(?:['’]\w+)*\b", t.casefold()) for t in texts]
        profile["D5"]["pooled"] = {}
        for n in (1, 2, 3):
            counts = Counter()
            for tokens in pooled:
                counts.update(ngrams(tokens, n))
            profile["D5"]["pooled"][str(n)] = len(counts) / sum(counts.values()) if counts else None
        profile["D9"] = measurement([f["lexical"] for f in features])
        profile["D11"] = measurement(
            {"within_trigram": [f["lexical"]["repeated_trigram_rate"] for f in features]}
        )
    if gt and len(gt) == len(features) and (prompt_tokens is not None or source_tokens is not None):
        profile["D10"] = measurement(
            {
                name: [overlap(t, context) for t in gt]
                for name, context in [("prompt", prompt_tokens), ("source", source_tokens)]
                if context is not None
            },
            n=3,
        )
    if repeated_prompt and len(features) >= 2:
        if len(gt) == len(features):
            profile["D4"] = measurement(self_bleu(gt), order=4, smoothing="add-one-all-orders")
            sets = [set(ngrams(t, 3)) for t in gt]
            pairs = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a and b]
            profile["D11"]["value"]["cross_jaccard"] = mean(pairs) if pairs else None
        if len(ge) == len(features):
            profile["D6"] = measurement(dispersion(ge), method="learned_representation")
        profile["D11"]["value"]["duplicate_output_rate"] = (len(texts) - len(set(texts))) / len(
            texts
        )
    if not texts:
        profile = {
            key: measurement(status="not_applicable", reason="No designated prose in this task")
            for key in profile
        }
    else:
        for key in ("D4", "D6"):
            if profile[key]["status"] == "not_applicable":
                profile[key] = measurement(
                    status="insufficient_samples",
                    reason="At least two saved alternatives for the same prompt required",
                )
        for key in ("D7", "D8"):
            profile[key] = measurement(
                status="not_applicable", reason="No task-paired reference supplied"
            )
    if repeated_prompt:
        # The floors belong wherever a distributional number is produced, not only in the
        # pooled caller, so a direct `repeated_prompt=True` cannot return an unmarked
        # under-powered value. The caller supplies the count of independent attempts when
        # it knows it; otherwise the number of outputs is the only available proxy.
        count = len(features) if samples is None else samples
        profile.update({name: sample_power(name, profile[name], count) for name in DISTRIBUTIONAL})
    return {
        "version": 2,
        "metrics": profile,
        "samples": len(features),
        "reference_samples": len(references),
        "feature_configs": sorted(configs),
        "prose_hashes": [f["prose_hash"] for f in features],
        "reference_hashes": [f["prose_hash"] for f in references],
    }


# Samples required before a distributional measurement is reported at all, and before it
# carries enough power to compare two models. These are our floors, not validated
# thresholds: MMD is a U-statistic whose variance dominates the estimate below a few tens
# of samples, and the technique this instrument is borrowed from reported against 2,000.
# Below the floor a measure is unavailable rather than a noisy number.
MINIMUM_SAMPLES = {"D1": 2, "D2": 20, "D4": 8, "D6": 8}
RELIABLE_SAMPLES = {"D1": 25, "D2": 50, "D4": 25, "D6": 25}
DISTRIBUTIONAL = tuple(MINIMUM_SAMPLES)


def sample_power(metric: str, entry: dict, samples: int) -> dict:
    """Attach the sample count to a distribution measurement, or withhold it.

    A distribution measured from four outputs and one measured from two hundred are not
    the same quantity, so the count travels with the value and an under-powered measure
    reports `insufficient_samples` instead of a number someone might quote.
    """
    if entry.get("status") != "ok" or entry.get("value") is None:
        # A measurement with no value has no power to report. Labelling it `ok` would
        # claim a sample count for a number that does not exist.
        return entry
    floor = MINIMUM_SAMPLES.get(metric)
    if floor is None:
        return {**entry, "samples": samples}
    if samples < floor:
        return measurement(
            status="insufficient_samples",
            method=entry.get("method", "deterministic"),
            reason=f"{metric} needs at least {floor} samples; this run has {samples}",
            samples=samples,
        )
    return {
        **entry,
        "samples": samples,
        "power": "ok" if samples >= RELIABLE_SAMPLES[metric] else "low",
    }


def sampling_plan(samples: int) -> dict:
    """Map a sample count onto the distributional measures it can actually support.

    The floors are the ones `sample_power` enforces, so a plan computed here and a result
    produced later cannot disagree. Use it to size a run before spending the GPU time: it
    answers how many attempts a claim needs, and names the measures a cheaper run has to
    give up rather than leaving them to fail quietly at scoring time.
    """
    if type(samples) is not int or samples < 1:
        raise ValueError("Sample count must be a positive integer")
    return {
        metric: (
            "insufficient"
            if samples < MINIMUM_SAMPLES[metric]
            else "low"
            if samples < RELIABLE_SAMPLES[metric]
            else "ok"
        )
        for metric in DISTRIBUTIONAL
    }


def sample_distribution(
    cards: list[dict],
    extractor: ProseFeatures,
    *,
    references: list[dict] | None = None,
    sigma: float | None = None,
    embeddings: bool = True,
    allow_download: bool = False,
) -> dict:
    """Distribution measurements pooled across repeated attempts of one scenario.

    `score_prose` profiles one attempt against a reference corpus, so measures defined
    across outputs -- MMD, self-BLEU, within-prompt dispersion -- can never be computed
    from it. This pools every attempt of a single scenario, which is the only thing that
    makes them exist at all, and reports the sample count beside each value.

    Token features are always requested because the n-gram measures are cheap and local.
    Embeddings are separate because they need a pinned model, and leaving them off should
    cost D2 and D6 rather than every measure in the profile.

    Identical outputs are kept. They are the signal the duplicate-rate measure is looking
    for, and dropping them would hide the failure mode this instrument exists to detect.
    """
    if not cards:
        raise ValueError("No attempts to pool")
    scenarios = {card["scenario_id"] for card in cards}
    if len(scenarios) != 1:
        raise ValueError(f"Pool attempts of one scenario; got {sorted(scenarios)}")
    # One scenario id can be run by several models or conditions. Those outputs belong to
    # different distributions, and the feature-config guard cannot catch it because a
    # single extractor produces every feature record.
    settings = {
        (fingerprint(card.get("model", {})), card.get("condition", "unspecified")) for card in cards
    }
    if len(settings) > 1:
        raise ValueError(
            "Pool attempts of one model and condition; got "
            f"{len(settings)} distinct settings for {sorted(scenarios)[0]}"
        )
    texts = [a["text"] for card in cards for a in card["artifacts"] if a["status"] == "ok"]
    features = [
        extractor.extract(text, tokens=True, embeddings=embeddings, allow_download=allow_download)
        for text in texts
    ]
    profile = prose_profile(
        texts,
        features,
        references=references,
        sigma=sigma,
        repeated_prompt=True,
        samples=len(cards),
    )
    return {
        "scenario_id": scenarios.pop(),
        # `attempts` is the independent unit the floors are about. `outputs` can exceed it
        # when one attempt writes several prose files, which are correlated rather than
        # independent draws and must not be counted toward a sample floor.
        "attempts": len(cards),
        "outputs": len(texts),
        "samples": len(cards),
        "reference_samples": profile["reference_samples"],
        "metrics": profile["metrics"],
    }


def paired_similarity(
    candidate: str,
    reference: str,
    *,
    bert_model_path: Path | None = None,
    bert_revision: str | None = None,
) -> dict:
    """Local BERTScore checkpoint only; no implicit large-model download."""
    results = {}
    try:
        from rouge_score import rouge_scorer

        scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False).score(
            reference, candidate
        )
        results["D7"] = measurement(
            {
                key: {"precision": s.precision, "recall": s.recall, "f1": s.fmeasure}
                for key, s in scores.items()
            },
            tokenizer="rouge-score-default",
            stemming=False,
        )
    except ImportError:
        results["D7"] = measurement(status="unavailable", reason="Install the metrics extra")
    if bert_model_path is None or bert_revision is None:
        results["D8"] = measurement(
            status="unavailable",
            method="learned_representation",
            reason="Local BERTScore model directory and explicit revision required",
        )
    else:
        from bert_score import score

        p, r, f = score(
            [candidate],
            [reference],
            model_type=str(bert_model_path),
            num_layers=17,
            idf=False,
            rescale_with_baseline=False,
            device="cpu",
        )
        results["D8"] = measurement(
            {"precision": p.item(), "recall": r.item(), "f1": f.item()},
            method="learned_representation",
            model_revision=bert_revision,
            model_path=str(bert_model_path),
            num_layers=17,
            idf=False,
            rescale_with_baseline=False,
        )
    return results


def compare_groups(
    scorecards: list[dict],
    references: list[dict],
    extractor: ProseFeatures,
    *,
    group_by=("model", "family", "provenance"),
    sigma=None,
    model_features=False,
) -> list[dict]:
    """Compare saved prose by selected metadata, with a caller-selected reference set.

    References are catalog records. Optional style/length grouping also filters
    references on those fields. Unknown style must be reviewed, never guessed.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    permitted = {
        "model",
        "family",
        "condition",
        "provenance",
        "input_provenance",
        "instruction_specificity",
        "genre",
        "style",
        "length",
    }
    if set(group_by) - permitted:
        raise ValueError("Unknown grouping field")
    if any(r["provenance"] != "human" or r["role"] == "train" for r in references):
        raise ValueError("Reference collection requires human prose outside training")

    def length_bin(text):
        count = len(text.split())
        return "under_120" if count < 120 else "120_499" if count < 500 else "500_plus"

    for card in scorecards:
        for artifact in card["artifacts"]:
            if artifact["status"] != "ok":
                continue
            metadata = {
                **card,
                "model": fingerprint(card.get("model", {})),
                "length": length_bin(artifact["text"]),
                "instruction_specificity": card.get("instruction_specificity", "unspecified"),
                "genre": card.get("genre", "unspecified"),
            }
            groups[tuple(metadata[k] for k in group_by)].append((card, artifact))
    comparisons = []
    for key, entries in sorted(groups.items()):
        labels = dict(zip(group_by, key, strict=True))
        selected = [
            r
            for r in references
            if ("style" not in labels or r.get("style") == labels["style"])
            and ("length" not in labels or length_bin(r["text"]) == labels["length"])
        ]
        texts = [artifact["text"] for _, artifact in entries]
        generated = [
            extractor.extract(t, tokens=model_features, embeddings=model_features) for t in texts
        ]
        reference_features = [
            extractor.extract(r["text"], tokens=model_features, embeddings=model_features)
            for r in selected
        ]
        profile = prose_profile(texts, generated, references=reference_features, sigma=sigma)
        comparisons.append(
            {
                "group": labels,
                **profile,
                "model_configuration": entries[0][0].get("model", {}),
                "reference_ids": [r["id"] for r in selected],
                "reference_policy": "caller-selected collection",
                "source_groups": sorted(
                    {g for card, _ in entries for g in card.get("source_groups", [])}
                ),
            }
        )
    return comparisons


def score_prose(
    card: dict,
    scenario: dict,
    result: dict,
    extractor: ProseFeatures,
    *,
    references: list[dict] | None = None,
    sigma: float | None = None,
    model_features: bool = True,
    allow_download: bool = False,
) -> dict:
    """Measure designated prose from saved attempts; never generate candidate text.

    References are caller-selected feature records. Single-output MMD remains
    insufficient_samples; chunks are not promoted to independent observations.
    """
    texts = [a["text"] for a in card["artifacts"] if a["status"] == "ok"]
    features = [
        extractor.extract(
            t, tokens=model_features, embeddings=model_features, allow_download=allow_download
        )
        for t in texts
    ]
    context_features = {}
    if texts and model_features:
        contexts = {"prompt": scenario["visible"]["brief"]}
        sources = result.get("before", {})
        if sources:
            contexts["source"] = "\n\n".join(sources[k] for k in sorted(sources))
        context_features = {
            name: extractor.extract(text, tokens=True, allow_download=allow_download)
            for name, text in contexts.items()
        }
    profile = prose_profile(
        texts,
        features,
        references=references,
        sigma=sigma,
        prompt_tokens=context_features.get("prompt", {}).get("tokens"),
        source_tokens=context_features.get("source", {}).get("tokens"),
    )
    paired = scenario["labels"].get("paired_reference")
    if paired is not None and len(texts) == 1:
        profile["metrics"].update(paired_similarity(texts[0], paired))
    profile["feature_errors"] = [
        {"hash": f["prose_hash"], **{k: v for k, v in f.items() if k.endswith("_error")}}
        for f in features + list(context_features.values())
        if any(k.endswith("_error") for k in f)
    ]
    profile["context_hashes"] = {k: v["prose_hash"] for k, v in context_features.items()}
    profile["source_overlap_scope"] = "initial supplied files, not evidence of actual retrieval"
    return profile
