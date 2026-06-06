from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
import sys

_YAML_IMPORT_ERROR: Optional[BaseException] = None
try:
    import yaml
except ImportError as _e:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_IMPORT_ERROR = _e

_DEFAULT_SYSTEM_PROMPT = (
    "You are a strict evaluator (judge). "
    "The sample is tagged with language code `{language_tag}`. "
    "Given an input and a reference answer, decide whether the reference is correct. "
    "Respond with a single JSON object and nothing else, with exactly these keys: "
    "`correct` (boolean), `score` (number between 0 and 1), `reason` (string)."
)

_DEFAULT_USER_PROMPT = "Evaluate the following JSON payload:\n{payload}\n"


def _safe_model_slug(model: str) -> str:
    s = model.strip().replace("\\", "/")
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.replace("/", "_"))
    return s.strip("_") or "model"


def _output_path_from_args(
    dataset_name: str,
    lang: str,
    judge_model: str,
    out_dir: str,
    extra_lang_count: int,
    max_rounds: int,
) -> str:
    slug = _safe_model_slug(judge_model)
    total_langs = 1 + extra_lang_count
    return os.path.join(
        out_dir,
        f"{dataset_name}.{lang}.{slug}.multiconsensus.extra{extra_lang_count}.total{total_langs}.rounds{max_rounds}.jsonl",
    )


def _dataset_stem_from_input(input_path: str) -> str:
    return os.path.splitext(os.path.basename(input_path))[0]


def _dataset_yaml_path(input_path: str) -> str:
    d = os.path.dirname(os.path.abspath(input_path))
    stem = _dataset_stem_from_input(input_path)
    return os.path.join(d, f"{stem}.yaml")


def _load_dataset_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from _YAML_IMPORT_ERROR
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML must be a mapping at top level: {path}")
    return data


def _resolve_key_pattern(key: str, lang: str) -> str:
    return key.replace("{lang}", lang).replace("{language_tag}", lang)


def _serialize_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


def _named_placeholder_value(name: str, lang_obj: Dict[str, Any], lang: str, cfg: Dict[str, Any]) -> str:
    spec = cfg.get(name)
    if not isinstance(spec, dict):
        return ""
    if spec.get("whole_lang_obj"):
        return json.dumps(lang_obj, ensure_ascii=False, default=str)
    ik = spec.get("key")
    if ik is None:
        return ""
    v = lang_obj.get(_resolve_key_pattern(str(ik), lang))
    return _serialize_value(v)


def _fill_user_template(template: str, lang_obj: Dict[str, Any], lang: str, cfg: Dict[str, Any]) -> str:
    formatter = string.Formatter()
    values: Dict[str, str] = {}
    for _, field, _, _ in formatter.parse(template):
        if field is None:
            continue
        values[field] = _named_placeholder_value(field, lang_obj, lang, cfg)
    try:
        return template.format(**values)
    except KeyError as e:
        raise ValueError(f"user_prompt placeholder missing from YAML or empty: {e}") from e


def _build_round_messages(
    lang_code: str,
    lang_obj: Dict[str, Any],
    system_template: str,
    user_template: str,
    dataset_cfg: Dict[str, Any],
    history_reference_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    task_type = str(dataset_cfg.get("task_type", "generic"))
    system = system_template.replace("{language_tag}", lang_code).replace("{task_type}", task_type)
    user = _fill_user_template(user_template, lang_obj, lang_code, dataset_cfg)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if history_reference_text:
        msgs.append({"role": "user", "content": history_reference_text})
    msgs.append({"role": "user", "content": user})
    return msgs


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    escaped = False
    candidate: Optional[str] = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                break
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    bool_match = re.search(r'"correct"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if not bool_match:
        bool_match = re.search(r"\bcorrect\s*:\s*(true|false)\b", text, flags=re.IGNORECASE)
    if not bool_match:
        return None
    correct = bool_match.group(1).lower() == "true"

    reason = ""
    reason_key = re.search(r'"reason"\s*:\s*', text, flags=re.IGNORECASE)
    if reason_key:
        tail = text[reason_key.end() :].strip()
        if tail.startswith('"'):
            out_chars: List[str] = []
            escaped = False
            i = 1
            while i < len(tail):
                ch = tail[i]
                if escaped:
                    out_chars.append(ch)
                    escaped = False
                    i += 1
                    continue
                if ch == "\\":
                    escaped = True
                    i += 1
                    continue
                if ch == '"':
                    j = i + 1
                    while j < len(tail) and tail[j].isspace():
                        j += 1
                    if j >= len(tail) or tail[j] in ",}":
                        break
                    out_chars.append(ch)
                    i += 1
                    continue
                out_chars.append(ch)
                i += 1
            reason = "".join(out_chars).strip()
        else:
            m = re.search(r"[,}]", tail)
            reason = (tail[: m.start()] if m else tail).strip()
    if not reason:
        reason = "No reason provided"

    return {"correct": correct, "reason": reason}


def _call_sglang_openai_compatible(
    *,
    base_url: str,
    api_path: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    timeout_s: int,
    temperature: float,
    max_tokens: int,
) -> str:
    url = base_url.rstrip("/") + "/" + api_path.lstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message", {})
        content = msg.get("content")
        if content is None:
            return data["choices"][0].get("text", "")
        return content
    return json.dumps(data, ensure_ascii=False)


def consistency_rate_counts(false_count: int, true_count: int) -> float:
    total = false_count + true_count
    if total <= 0:
        return float("nan")
    return float(max(false_count, true_count) / total)


def _vote_from_judge(j: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(j, dict) or "correct" not in j:
        return None
    try:
        return 1 if bool(j["correct"]) else 0
    except Exception:
        return None


def _build_history_reference_text(
    target_lang: str,
    prev_round_index: int,
    consistency_rate: float,
    summaries: List[Dict[str, Any]],
    prev_final_judge: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("History reference from previous iteration:")
    lines.append(f"- Previous round index: {prev_round_index}")
    lines.append(f"- Previous cross-language consistency rate: {consistency_rate}")
    if isinstance(prev_final_judge, dict):
        pf = prev_final_judge.get("correct")
        if pf is True:
            pf_text = "true"
        elif pf is False:
            pf_text = "false"
        else:
            pf_text = "null"
        pr = prev_final_judge.get("reason")
        pr_text = str(pr).strip() if pr not in (None, "") else "no reason provided"
        lines.append(f"- Your previous final verdict: correct={pf_text}; reason={pr_text}")
    lines.append("- Previous language judgments and reasons (including target language):")
    for s in summaries:
        lc = str(s.get("lang", ""))
        c = s.get("correct")
        if c is True:
            verdict = "true"
        elif c is False:
            verdict = "false"
        else:
            verdict = "null"
        reason = s.get("reason")
        err = s.get("error")
        reason_text = str(reason).strip() if reason not in (None, "") else ""
        if not reason_text and err:
            reason_text = f"no valid judgment ({err})"
        if not reason_text:
            reason_text = "no reason provided"
        lines.append(f"  - {lc}: correct={verdict}; reason={reason_text}")
    lines.append(
        "- Please reconsider your judgment by incorporating cross-lingual historical judgments as additional diagnostic signals for the current evaluation."
    )
    return "\n".join(lines)


def _build_round_history_reference_text(
    current_lang: str,
    prev_round_index: int,
    consistency_rate: float,
    summaries: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("History reference from previous iteration:")
    lines.append(f"- Previous round index: {prev_round_index}")
    lines.append(f"- Previous cross-language consistency rate: {consistency_rate}")
    lines.append("- Previous language judgments and reasons:")
    for s in summaries:
        lc = str(s.get("lang", ""))
        c = s.get("correct")
        if c is True:
            verdict = "true"
        elif c is False:
            verdict = "false"
        else:
            verdict = "null"
        reason = s.get("reason")
        err = s.get("error")
        reason_text = str(reason).strip() if reason not in (None, "") else ""
        if not reason_text and err:
            reason_text = f"no valid judgment ({err})"
        if not reason_text:
            reason_text = "no reason provided"
        lines.append(f"  - {lc}: correct={verdict}; reason={reason_text}")
    lines.append("- Please reconsider your judgment by incorporating cross-lingual historical judgments as additional diagnostic signals for the current evaluation.")
    return "\n".join(lines)


def _pick_langs(raw: Dict[str, Any], target: str, k_extra: int, rng: random.Random) -> Optional[List[str]]:
    langs = [k for k in raw.keys() if k != "general" and isinstance(raw[k], dict)]
    if target not in langs:
        return None
    rest = [x for x in langs if x != target]
    if len(rest) < k_extra:
        return None
    return [target] + rng.sample(rest, k_extra)


def _run_consistency_round(
    *,
    raw: Dict[str, Any],
    lang: str,
    judge_model: str,
    base_url: str,
    api_path: str,
    api_key: Optional[str],
    timeout_s: int,
    temperature: float,
    max_tokens: int,
    round_parallel: int,
    system_prompt_template: str,
    user_prompt_template: str,
    dataset_cfg: Dict[str, Any],
    rng: random.Random,
    extra_lang_count: int,
    round_index: int,
    history_reference_by_lang: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    lang_order = _pick_langs(raw, lang, extra_lang_count, rng)
    if lang_order is None:
        return {
            "round_index": round_index,
            "error": f"need at least {1 + extra_lang_count} language blocks in record",
            "languages_order": [],
            "round_judges": {},
            "vote_counts": {"incorrect": 0, "correct": 0, "valid_votes": 0, "expected_votes": 0},
            "consistency_rate": float("nan"),
        }

    round_results: Dict[str, Any] = {}

    def one_judge(lc: str) -> Tuple[str, Dict[str, Any]]:
        lo = raw[lc]
        msgs = _build_round_messages(
            lc,
            lo,
            system_prompt_template,
            user_prompt_template,
            dataset_cfg,
            history_reference_text=(
                history_reference_by_lang.get(lc)
                if isinstance(history_reference_by_lang, dict)
                else None
            ),
        )
        try:
            raw_out = _call_sglang_openai_compatible(
                base_url=base_url,
                api_path=api_path,
                model=judge_model,
                messages=msgs,
                api_key=api_key,
                timeout_s=timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            return lc, {"messages": msgs, "judge": None, "judge_raw": None, "error": str(e)}
        parsed = _extract_first_json_object(raw_out)
        return lc, {"messages": msgs, "judge": parsed, "judge_raw": raw_out}

    if round_parallel <= 1:
        for lc in lang_order:
            code, payload = one_judge(lc)
            round_results[code] = payload
    else:
        with ThreadPoolExecutor(max_workers=min(round_parallel, len(lang_order))) as ex:
            futs = {ex.submit(one_judge, lc): lc for lc in lang_order}
            for fut in as_completed(futs):
                code, payload = fut.result()
                round_results[code] = payload

    votes: List[int] = []
    summaries: List[Dict[str, Any]] = []
    for lc in lang_order:
        block = round_results.get(lc, {})
        j = block.get("judge")
        v = _vote_from_judge(j)
        summaries.append(
            {
                "lang": lc,
                "correct": (bool(j["correct"]) if isinstance(j, dict) and "correct" in j else None),
                "reason": (j.get("reason") if isinstance(j, dict) else None),
                "score": (j.get("score") if isinstance(j, dict) else None),
                "error": block.get("error"),
            }
        )
        if v is not None:
            votes.append(v)

    n_false = sum(1 for x in votes if x == 0)
    n_true = sum(1 for x in votes if x == 1)
    consistency_rate = (
        consistency_rate_counts(n_false, n_true) if len(votes) == len(lang_order) else float("nan")
    )

    return {
        "round_index": round_index,
        "languages_order": lang_order,
        "round_judges": round_results,
        "round_summaries": summaries,
        "vote_counts": {
            "incorrect": n_false,
            "correct": n_true,
            "valid_votes": len(votes),
            "expected_votes": len(lang_order),
        },
        "consistency_rate": consistency_rate,
    }


def _run_target_final_judge(
    *,
    raw: Dict[str, Any],
    lang: str,
    judge_model: str,
    base_url: str,
    api_path: str,
    api_key: Optional[str],
    timeout_s: int,
    temperature: float,
    max_tokens: int,
    system_prompt_template: str,
    user_prompt_template: str,
    dataset_cfg: Dict[str, Any],
    history_reference_text: Optional[str] = None,
) -> Dict[str, Any]:
    target_obj = raw.get(lang)
    if not isinstance(target_obj, dict):
        return {
            "judge_final_messages": None,
            "judge_final": None,
            "judge_final_raw": None,
            "error_final": f"target language block missing: {lang}",
        }

    task_type = str(dataset_cfg.get("task_type", "generic"))
    final_system = system_prompt_template.replace("{language_tag}", lang).replace("{task_type}", task_type)
    target_user_text = _fill_user_template(user_prompt_template, target_obj, lang, dataset_cfg)
    final_messages: List[Dict[str, str]] = [
        {"role": "system", "content": final_system},
    ]
    if history_reference_text:
        final_messages.append({"role": "user", "content": history_reference_text})
    final_messages.append({"role": "user", "content": target_user_text})
    try:
        final_raw = _call_sglang_openai_compatible(
            base_url=base_url,
            api_path=api_path,
            model=judge_model,
            messages=final_messages,
            api_key=api_key,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        final_parsed = _extract_first_json_object(final_raw)
        return {
            "judge_final_messages": final_messages,
            "judge_final": final_parsed if isinstance(final_parsed, dict) else None,
            "judge_final_raw": final_raw,
        }
    except Exception as e:
        return {
            "judge_final_messages": final_messages,
            "judge_final": None,
            "judge_final_raw": None,
            "error_final": f"{type(e).__name__}: {e}",
        }


def _process_record(
    item: Tuple[int, Dict[str, Any]],
    *,
    lang: str,
    judge_model: str,
    base_url: str,
    api_path: str,
    api_key: Optional[str],
    timeout_s: int,
    temperature: float,
    max_tokens: int,
    round_parallel: int,
    system_prompt_template: str,
    user_prompt_template: str,
    dataset_cfg: Dict[str, Any],
    rng: random.Random,
    extra_lang_count: int,
    max_rounds: int,
    consistency_threshold: float,
) -> Dict[str, Any]:
    _idx, raw = item
    out: Dict[str, Any] = {
        "raw": raw,
        "general": raw.get("general", {}),
        "lang": lang,
        "max_rounds": max_rounds,
        "consistency_threshold": consistency_threshold,
        "rounds": [],
        "stopped_early": False,
        "stop_round": None,
    }

    # Round 0: raw single-sample target judgment (no multilang consistency context).
    round0 = {"round_index": 0}
    round0.update(
        _run_target_final_judge(
            raw=raw,
            lang=lang,
            judge_model=judge_model,
            base_url=base_url,
            api_path=api_path,
            api_key=api_key,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            dataset_cfg=dataset_cfg,
            history_reference_text=None,
        )
    )
    out["rounds"].append(round0)

    prev_round_for_history: Optional[Dict[str, Any]] = None
    # Every round: consistency + final target decision.
    # Stop rule applies from round 2 onward, based on CURRENT round consistency rate.
    for r in range(1, max_rounds + 1):
        round_history_by_lang: Optional[Dict[str, str]] = None
        if prev_round_for_history is not None:
            prev_round_index = int(prev_round_for_history.get("round_index", r - 1))
            prev_consistency_rate = float(prev_round_for_history.get("consistency_rate", float("nan")))
            prev_summaries = (
                prev_round_for_history.get("round_summaries", [])
                if isinstance(prev_round_for_history.get("round_summaries"), list)
                else []
            )
            round_history_by_lang = {}
            for lc in [k for k in raw.keys() if k != "general" and isinstance(raw[k], dict)]:
                round_history_by_lang[lc] = _build_round_history_reference_text(
                    current_lang=lc,
                    prev_round_index=prev_round_index,
                    consistency_rate=prev_consistency_rate,
                    summaries=prev_summaries,
                )

        rr = _run_consistency_round(
            raw=raw,
            lang=lang,
            judge_model=judge_model,
            base_url=base_url,
            api_path=api_path,
            api_key=api_key,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
            round_parallel=round_parallel,
            system_prompt_template=system_prompt_template,
            user_prompt_template=user_prompt_template,
            dataset_cfg=dataset_cfg,
            rng=rng,
            extra_lang_count=extra_lang_count,
            round_index=r,
            history_reference_by_lang=round_history_by_lang,
        )
        history_text: Optional[str] = None
        if r == 1:
            # Round 1 should already use multilang consistency information.
            history_text = _build_history_reference_text(
                target_lang=lang,
                prev_round_index=1,
                consistency_rate=float(rr.get("consistency_rate", float("nan"))),
                summaries=(rr.get("round_summaries", []) if isinstance(rr.get("round_summaries"), list) else []),
                prev_final_judge=None,
            )
        elif prev_round_for_history is not None:
            history_text = _build_history_reference_text(
                target_lang=lang,
                prev_round_index=int(prev_round_for_history.get("round_index", r - 1)),
                consistency_rate=float(prev_round_for_history.get("consistency_rate", float("nan"))),
                summaries=(
                    prev_round_for_history.get("round_summaries", [])
                    if isinstance(prev_round_for_history.get("round_summaries"), list)
                    else []
                ),
                prev_final_judge=(
                    prev_round_for_history.get("judge_final")
                    if isinstance(prev_round_for_history.get("judge_final"), dict)
                    else None
                ),
            )
        rr.update(
            _run_target_final_judge(
                raw=raw,
                lang=lang,
                judge_model=judge_model,
                base_url=base_url,
                api_path=api_path,
                api_key=api_key,
                timeout_s=timeout_s,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt_template=system_prompt_template,
                user_prompt_template=user_prompt_template,
                dataset_cfg=dataset_cfg,
                history_reference_text=history_text,
            )
        )
        out["rounds"].append(rr)
        prev_round_for_history = rr
        current_consistency_rate = rr.get("consistency_rate")
        if (
            r >= 1
            and isinstance(current_consistency_rate, (int, float))
            and float(current_consistency_rate) == float(current_consistency_rate)
            and float(current_consistency_rate) >= consistency_threshold
        ):
            out["stopped_early"] = True
            out["stop_round"] = r
            break

    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-round evolve judge: each round does consistency + final verdict, with early stop."
    )
    ap.add_argument("--input", required=True, help="Path to unified_data/<dataset>.jsonl")
    ap.add_argument("--lang", required=True, help="Target language key to finalize")
    ap.add_argument("--judge-model", required=True, help="Model name for sglang")

    ap.add_argument("--base-url", default="http://localhost:30000")
    ap.add_argument("--api-path", default="/v1/chat/completions")
    ap.add_argument("--api-key", default=None)

    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="judge_outputs")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=-1)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers over records")
    ap.add_argument("--round-parallel", type=int, default=4, help="Parallel API calls inside each record")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed for auxiliary languages")
    ap.add_argument(
        "--extra-lang-count",
        type=int,
        default=3,
        help="How many additional languages (besides target) to sample for consistency.",
    )
    ap.add_argument("--max-rounds", type=int, default=3, help="Max iteration rounds per record (>=1).")
    ap.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.75,
        help="If consistency rate reaches this threshold, stop this record early.",
    )

    args = ap.parse_args()
    if args.extra_lang_count < 1:
        raise ValueError("--extra-lang-count must be >= 1")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")

    yaml_path = _dataset_yaml_path(args.input)
    dataset_cfg = _load_dataset_yaml(yaml_path)

    if isinstance(dataset_cfg.get("system_prompt"), str):
        system_tmpl = dataset_cfg["system_prompt"]
    else:
        system_tmpl = _DEFAULT_SYSTEM_PROMPT

    if isinstance(dataset_cfg.get("user_prompt"), str):
        user_tmpl = dataset_cfg["user_prompt"]
    else:
        user_tmpl = _DEFAULT_USER_PROMPT

    dataset_name = _dataset_stem_from_input(args.input)
    if args.out:
        out_path = args.out
    else:
        out_path = _output_path_from_args(
            dataset_name,
            args.lang,
            args.judge_model,
            args.out_dir,
            args.extra_lang_count,
            args.max_rounds,
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(out_path)
    if not os.path.isfile(out_path):

        tasks: List[Tuple[int, Dict[str, Any]]] = []
        with open(args.input, "r", encoding="utf-8") as r:
            for idx, line in enumerate(r):
                if idx < args.start:
                    continue
                if args.max_records >= 0 and len(tasks) >= args.max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                tasks.append((idx, json.loads(line)))

        def _run_one(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            idx, _ = item
            rng = random.Random((args.seed * 1000003 + idx) % (2**31))
            return _process_record(
                item,
                lang=args.lang,
                judge_model=args.judge_model,
                base_url=args.base_url,
                api_path=args.api_path,
                api_key=args.api_key,
                timeout_s=args.timeout_s,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                round_parallel=args.round_parallel,
                system_prompt_template=system_tmpl,
                user_prompt_template=user_tmpl,
                dataset_cfg=dataset_cfg,
                rng=rng,
                extra_lang_count=args.extra_lang_count,
                max_rounds=args.max_rounds,
                consistency_threshold=args.consistency_threshold,
            )

        results: List[Dict[str, Any]]
        if args.workers <= 1:
            results = [_run_one(t) for t in tasks]
        else:
            results = [None] * len(tasks)  # type: ignore
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                future_to_pos = {ex.submit(_run_one, tasks[i]): i for i in range(len(tasks))}
                for fut in as_completed(future_to_pos):
                    pos = future_to_pos[fut]
                    results[pos] = fut.result()

        with open(out_path, "w", encoding="utf-8", newline="\n") as w:
            for out_obj in results:
                w.write(json.dumps(out_obj, ensure_ascii=False, indent=4) + "\n")

        print(f"wrote {len(results)} lines -> {out_path}")


if __name__ == "__main__":
    main()

