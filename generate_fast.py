import csv
import json
import os
import resource
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime

import ollama

try:
    import psutil
except Exception:
    psutil = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "retail_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "responses")
SYSTEM_PROMPTS = {
    "minimal": (
        "You are an assistant that analyzes user input related to a shopping cart. "
        "Your task is to extract cart operations from the user's message.\n"
        "Return ONLY a JSON array where each item has:\n"
        "- action (one of: 'add', 'remove')\n"
        "- product (the product name exactly as the user mentions it)\n"
        "- quantity (an integer; if not specified, assume 1)\n"
        "Output format:\n"
        "[{ \"action\": \"add\", \"product\": \"...\", \"quantity\": 1 }]"
    ),
    "extended": (
        "You are a shopping cart assistant. "
        "Your task is to extract cart operations from the user's message:\n"
        "- 'action': classify the intent as either 'add' or 'remove', even if the user uses other words or phrases "
        "(e.g., 'put', 'insert', 'delete', 'take out', 'nix', 'ring up', 'pack in', 'skip' etc.).\n"
        "- product: the exact product name as mentioned by the user\n"
        "- quantity: an integer (default to 1 if not specified).\n"
        "Return only a valid JSON array of objects, with no explanations, formatting, or extra text. "
        "The output must follow this format exactly:\n"
        "[{ \"action\": \"...\", \"product\": \"...\", \"quantity\": ... }]"
    ),
    "few_shot": (
        "You are a shopping-cart assistant whose only job is to parse the user's request and output a JSON array "
        "where every element has this exact schema:\n\n"
        "{\n"
        "  \"action\": \"<add|remove>\",\n"
        "  \"product\": \"<exact product name>\",\n"
        "  \"quantity\": <integer>\n"
        "}\n\n"
        "Rules:\n"
        "1. \"action\" must be either \"add\" or \"remove\". Map any synonyms (\"put in\", \"insert\", "
        "\"take out\", \"nix\", \"delete\", etc.) to these two.\n"
        "2. \"product\" is exactly what the customer wants, stripped of any action words or numbers.\n"
        "3. \"quantity\" is an integer. If the user does not specify a number, default to 1.\n"
        "4. Output ONLY the JSON array - no markdown, no explanations, no extra keys or text.\n\n"
        "Examples:\n\n"
        "User: Toss in 14 razors in there, throw in 20 pies, load up with 19 coffees, and also nix a pair of bananas\n"
        "Output:\n"
        "[\n"
        "{\"action\":\"add\",\"product\":\"razors\",\"quantity\":14},\n"
        "{\"action\":\"add\",\"product\":\"pies\",\"quantity\":20},\n"
        "{\"action\":\"add\",\"product\":\"coffees\",\"quantity\":19},\n"
        "{\"action\":\"remove\",\"product\":\"bananas\",\"quantity\":2}\n"
        "]\n\n"
        "User: Yo lose a dozen soaps, stick in a few potatoes in there, then take away a couple wraps\n"
        "Output:\n"
        "[\n"
        "{\"action\":\"remove\",\"product\":\"soaps\",\"quantity\":12},\n"
        "{\"action\":\"add\",\"product\":\"potatoes\",\"quantity\":3},\n"
        "{\"action\":\"remove\",\"product\":\"wraps\",\"quantity\":2}\n"
        "]\n\n"
        "User: Add apples\n"
        "Output:\n"
        "[\n"
        "{\"action\":\"add\",\"product\":\"apples\",\"quantity\":1}\n"
        "]\n\n"
        "Now parse the user's next message."
    ),
}

MODELS = [
    "llama3.2:3b",
    "granite4:3b",
    "mistral:7b",
    "llama3.1:8b",
    "gpt-oss:20b",
    "gemma4:31b",
]


os.makedirs(OUTPUT_DIR, exist_ok=True)


def model_slug(model_name):
    return "".join(ch if ch.isalnum() else "_" for ch in model_name)

def load_dataset():
    rows = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "user_input" not in reader.fieldnames:
            raise ValueError("retail_dataset.csv must contain a 'user_input' column")
        for idx, row in enumerate(reader, start=1):
            user_input = (row.get("user_input") or "").strip()
            if not user_input:
                continue
            try:
                ground_truth_ops = json.loads(row.get("items_json", "[]"))
            except Exception:
                ground_truth_ops = []
            rows.append({
                "id": str(idx),
                "user_input": user_input,
                "ground_truth_ops": ground_truth_ops,
            })
    return rows


def query_model(model, system_prompt, user_input):
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        return {
            "text": response.get("message", {}).get("content", ""),
            "error": False,
            "raw": response,
        }
    except Exception as e:
        return {
            "text": f"ERROR: {str(e)}",
            "error": True,
            "raw": {"error": str(e)},
        }


def parse_response_payload(text):
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        pass

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        try:
            return json.loads(raw[first_arr:last_arr + 1])
        except Exception:
            pass

    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        try:
            return json.loads(raw[first_obj:last_obj + 1])
        except Exception:
            pass

    return None


def payload_to_operations(payload):
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def response_to_operations(response_text):
    return payload_to_operations(parse_response_payload(response_text))


def save_metrics(metrics, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def mean(values):
    if not values:
        return None
    if hasattr(statistics, "fmean"):
        return statistics.fmean(values)
    return sum(values) / float(len(values))


def percentile(values, p):
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def op_tuple(op):
    def to_hashable(value):
        if isinstance(value, dict):
            return tuple(sorted((k, to_hashable(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(to_hashable(v) for v in value)
        return value

    return (
        to_hashable(op.get("action")),
        to_hashable(op.get("product")),
        to_hashable(op.get("quantity")),
    )


def is_schema_valid_op(op):
    if not isinstance(op, dict):
        return False
    required = {"action", "product", "quantity"}
    if not required.issubset(set(op.keys())):
        return False
    action = op.get("action")
    if not isinstance(action, str):
        return False
    if action not in {"add", "remove"}:
        return False
    if not isinstance(op.get("product"), str):
        return False
    quantity = op.get("quantity")
    if isinstance(quantity, bool):
        return False
    return isinstance(quantity, int)


def query_gpu_stats():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
        if proc.returncode != 0:
            return None
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            return None

        gpu_utils = []
        mem_used = 0.0
        mem_total = 0.0
        power_vals = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_utils.append(float(parts[0]))
            mem_used += float(parts[1])
            mem_total += float(parts[2])
            power_vals.append(float(parts[3]))

        if not gpu_utils:
            return None

        return {
            "gpu_util_percent": sum(gpu_utils) / len(gpu_utils),
            "vram_used_mb": mem_used,
            "vram_total_mb": mem_total,
            "power_watts": sum(power_vals) / len(power_vals) if power_vals else None,
        }
    except Exception:
        return None


def compute_quality_metrics(results, ground_truth_by_id):
    json_valid = 0
    schema_valid = 0

    tp = 0
    fp = 0
    fn = 0

    order_exact = 0
    orderless_exact = 0

    op_count_exact = 0
    op_count_abs_error_sum = 0

    action_match = 0
    product_match = 0
    quantity_match = 0
    field_total = 0

    for item in results:
        payload = parse_response_payload(item.get("response", ""))
        pred_ops = item.get("parsed_ops")
        if not isinstance(pred_ops, list):
            pred_ops = response_to_operations(item.get("response", ""))
        gt_ops = ground_truth_by_id.get(str(item.get("id", "")), [])

        if payload is not None:
            json_valid += 1

        if pred_ops and all(is_schema_valid_op(op) for op in pred_ops):
            schema_valid += 1

        pred_counter = Counter(op_tuple(op) for op in pred_ops)
        gt_counter = Counter(op_tuple(op) for op in gt_ops if isinstance(op, dict))
        local_tp = sum(min(pred_counter[k], gt_counter[k]) for k in pred_counter)
        local_fp = sum(pred_counter.values()) - local_tp
        local_fn = sum(gt_counter.values()) - local_tp
        tp += local_tp
        fp += local_fp
        fn += local_fn

        if pred_ops == gt_ops:
            order_exact += 1
        if pred_counter == gt_counter:
            orderless_exact += 1

        if len(pred_ops) == len(gt_ops):
            op_count_exact += 1
        op_count_abs_error_sum += abs(len(pred_ops) - len(gt_ops))

        max_len = max(len(pred_ops), len(gt_ops))
        for i in range(max_len):
            pred_op = pred_ops[i] if i < len(pred_ops) else {}
            gt_op = gt_ops[i] if i < len(gt_ops) and isinstance(gt_ops[i], dict) else {}
            field_total += 1
            if pred_op.get("action") == gt_op.get("action"):
                action_match += 1
            if pred_op.get("product") == gt_op.get("product"):
                product_match += 1
            if pred_op.get("quantity") == gt_op.get("quantity"):
                quantity_match += 1

    n = len(results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "json_validity_rate": json_valid / n if n else 0.0,
        "schema_validity_rate": schema_valid / n if n else 0.0,
        "operation_precision": precision,
        "operation_recall": recall,
        "operation_f1": f1,
        "full_sample_exact_match_ordered_rate": order_exact / n if n else 0.0,
        "full_sample_exact_match_unordered_rate": orderless_exact / n if n else 0.0,
        "operation_count_exact_match_rate": op_count_exact / n if n else 0.0,
        "operation_count_mae": op_count_abs_error_sum / n if n else 0.0,
        "action_accuracy": action_match / field_total if field_total else 0.0,
        "product_accuracy": product_match / field_total if field_total else 0.0,
        "quantity_accuracy": quantity_match / field_total if field_total else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate_stats(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def compute_performance_metrics(results, variant_runtime_sec, processed_this_run, skipped_this_run):
    latencies = [r.get("latency_sec") for r in results if isinstance(r.get("latency_sec"), (int, float))]
    success_count = 0
    error_count = 0
    prompt_tokens = []
    completion_tokens = []
    eval_durations = []

    cpu_samples = [r.get("cpu_percent") for r in results if isinstance(r.get("cpu_percent"), (int, float))]
    ram_samples = [r.get("ram_mb") for r in results if isinstance(r.get("ram_mb"), (int, float))]
    gpu_util_samples = [r.get("gpu_util_percent") for r in results if isinstance(r.get("gpu_util_percent"), (int, float))]
    vram_used_samples = [r.get("vram_used_mb") for r in results if isinstance(r.get("vram_used_mb"), (int, float))]
    power_samples = [r.get("gpu_power_watts") for r in results if isinstance(r.get("gpu_power_watts"), (int, float))]

    for r in results:
        if r.get("error"):
            error_count += 1
        else:
            success_count += 1
        if isinstance(r.get("prompt_eval_count"), (int, float)):
            prompt_tokens.append(float(r.get("prompt_eval_count")))
        if isinstance(r.get("eval_count"), (int, float)):
            completion_tokens.append(float(r.get("eval_count")))
        if isinstance(r.get("eval_duration_sec"), (int, float)):
            eval_durations.append(float(r.get("eval_duration_sec")))

    maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ram_peak_mb = maxrss_kb / 1024.0

    total_samples = len(results)
    throughput = processed_this_run / variant_runtime_sec if variant_runtime_sec > 0 else 0.0

    return {
        "total_runtime_seconds": variant_runtime_sec,
        "processed_this_run": processed_this_run,
        "skipped_this_run": skipped_this_run,
        "total_samples_in_file": total_samples,
        "throughput_samples_per_second": throughput,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / total_samples if total_samples else 0.0,
        "error_rate": error_count / total_samples if total_samples else 0.0,
        "latency_seconds": aggregate_stats(latencies),
        "tokens": {
            "prompt_eval_count_sum": sum(prompt_tokens) if prompt_tokens else None,
            "eval_count_sum": sum(completion_tokens) if completion_tokens else None,
            "prompt_eval_count_avg": mean(prompt_tokens),
            "eval_count_avg": mean(completion_tokens),
            "eval_duration_seconds_avg": mean(eval_durations),
        },
        "resources": {
            "cpu_percent": aggregate_stats(cpu_samples),
            "ram_mb_avg": mean(ram_samples),
            "ram_mb_peak": ram_peak_mb,
            "gpu_util_percent": aggregate_stats(gpu_util_samples),
            "vram_used_mb": aggregate_stats(vram_used_samples),
            "gpu_power_watts": aggregate_stats(power_samples),
        },
    }


def save_results_json(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_results_csv(results, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "llm_ops_json",
            "ground_truth_ops_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            parsed_ops = item.get("parsed_ops")
            if not isinstance(parsed_ops, list):
                parsed_ops = response_to_operations(item.get("response", ""))

            writer.writerow(
                {
                    "id": item.get("id", ""),
                    "llm_ops_json": json.dumps(parsed_ops, ensure_ascii=False),
                    "ground_truth_ops_json": json.dumps(item.get("ground_truth_ops", []), ensure_ascii=False),
                }
            )


def main():
    dataset_rows = load_dataset()
    ground_truth_by_id = {row["id"]: row.get("ground_truth_ops", []) for row in dataset_rows}
    summary_entries = []
    summary_file = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    print(f"Loaded {len(dataset_rows)} user inputs.")

    for model in MODELS:
        model_dir = os.path.join(OUTPUT_DIR, model_slug(model))
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n>>> Model: {model}")

        for variant, system_prompt in SYSTEM_PROMPTS.items():
            output_json = os.path.join(model_dir, f"results_{variant}.json")
            output_csv = os.path.join(model_dir, f"results_{variant}.csv")
            output_metrics = os.path.join(model_dir, f"metrics_{variant}.json")
            variant_start = time.perf_counter()
            processed_this_run = 0
            skipped_this_run = 0

            # Load existing results so a restarted job does not lose previous work.
            if os.path.exists(output_json):
                with open(output_json, "r", encoding="utf-8") as f:
                    results = json.load(f)
                print(f"  [{variant}] Resuming: {len(results)} results already saved.")
            else:
                results = []
                print(f"  [{variant}] Starting fresh.")

            done = {r["id"] for r in results}

            for row in dataset_rows:
                sample_id = row["id"]
                user_input = row["user_input"]

                if sample_id in done:
                    print(f"    {sample_id}... skipped")
                    skipped_this_run += 1
                    continue

                print(f"    {sample_id}...", end=" ", flush=True)
                request_start = time.perf_counter()
                cpu_start = time.process_time()
                response_obj = query_model(model, system_prompt, user_input)
                cpu_end = time.process_time()
                request_end = time.perf_counter()

                response_text = response_obj.get("text", "")
                response_raw = response_obj.get("raw", {})
                parsed_ops = response_to_operations(response_text)
                eval_duration_ns = response_raw.get("eval_duration")
                eval_duration_sec = (
                    float(eval_duration_ns) / 1_000_000_000.0
                    if isinstance(eval_duration_ns, (int, float))
                    else None
                )

                wall_sec = request_end - request_start
                cpu_percent = (cpu_end - cpu_start) * 100.0 / wall_sec if wall_sec > 0 else None

                ram_mb = None
                if psutil is not None:
                    try:
                        ram_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
                    except Exception:
                        ram_mb = None

                gpu_stats = query_gpu_stats() or {}

                results.append(
                    {
                        "id": sample_id,
                        "model": model,
                        "prompt_variant": variant,
                        "user_input": user_input,
                        "response": response_text,
                        "parsed_ops": parsed_ops,
                        "ground_truth_ops": row.get("ground_truth_ops", []),
                        "error": bool(response_obj.get("error", False)),
                        "latency_sec": wall_sec,
                        "cpu_percent": cpu_percent,
                        "ram_mb": ram_mb,
                        "gpu_util_percent": gpu_stats.get("gpu_util_percent"),
                        "vram_used_mb": gpu_stats.get("vram_used_mb"),
                        "vram_total_mb": gpu_stats.get("vram_total_mb"),
                        "gpu_power_watts": gpu_stats.get("power_watts"),
                        "prompt_eval_count": response_raw.get("prompt_eval_count"),
                        "eval_count": response_raw.get("eval_count"),
                        "eval_duration_sec": eval_duration_sec,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                save_results_json(results, output_json)
                save_results_csv(results, output_csv)
                done.add(sample_id)
                processed_this_run += 1
                print("✓")

            variant_runtime_sec = time.perf_counter() - variant_start
            performance = compute_performance_metrics(
                results,
                variant_runtime_sec,
                processed_this_run,
                skipped_this_run,
            )
            quality = compute_quality_metrics(results, ground_truth_by_id)
            metrics = {
                "model": model,
                "prompt_variant": variant,
                "generated_at": datetime.now().isoformat(),
                "dataset_file": DATASET_FILE,
                "results_file_json": output_json,
                "results_file_csv": output_csv,
                "performance": performance,
                "quality": quality,
            }
            save_metrics(metrics, output_metrics)
            summary_entries.append(metrics)

            print(
                f"  [{variant}] Done! Results saved to {output_json} and {output_csv}; "
                f"metrics saved to {output_metrics}"
            )

    leaderboard = sorted(
        [
            {
                "model": m.get("model"),
                "prompt_variant": m.get("prompt_variant"),
                "operation_f1": m.get("quality", {}).get("operation_f1"),
                "full_sample_exact_match_ordered_rate": m.get("quality", {}).get(
                    "full_sample_exact_match_ordered_rate"
                ),
                "json_validity_rate": m.get("quality", {}).get("json_validity_rate"),
                "schema_validity_rate": m.get("quality", {}).get("schema_validity_rate"),
                "latency_mean_seconds": m.get("performance", {}).get("latency_seconds", {}).get("mean"),
                "throughput_samples_per_second": m.get("performance", {}).get("throughput_samples_per_second"),
                "success_rate": m.get("performance", {}).get("success_rate"),
                "metrics_file": os.path.join(
                    OUTPUT_DIR,
                    model_slug(m.get("model", "")),
                    f"metrics_{m.get('prompt_variant', '')}.json",
                ),
            }
            for m in summary_entries
        ],
        key=lambda row: (
            -float(row.get("operation_f1") or 0.0),
            -float(row.get("full_sample_exact_match_ordered_rate") or 0.0),
            float(row.get("latency_mean_seconds") or 1e18),
        ),
    )

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "dataset_file": DATASET_FILE,
        "total_models": len(MODELS),
        "total_prompt_variants": len(SYSTEM_PROMPTS),
        "total_entries": len(summary_entries),
        "entries": summary_entries,
        "leaderboard": leaderboard,
    }
    save_metrics(summary_payload, summary_file)
    print(f"\nSummary metrics saved to {summary_file}")

if __name__ == "__main__":
    main()