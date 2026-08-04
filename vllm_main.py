import argparse
import asyncio
import csv
import os
import time

import aiohttp
from tqdm import tqdm

PROMPT_TEMPLATE = "(Random seed {i}) Tell me a very long (1000+ words length) scary story about hiking through the woods"


async def send_query(session, url, model, i, max_tokens, semaphore):
    payload = {
        "model": model,
        "prompt": PROMPT_TEMPLATE.format(i=i),
        "max_tokens": max_tokens,
    }

    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                ok = resp.status == 200
        except Exception as err:
            return {"i": i, "ok": False, "elapsed": time.perf_counter() - start, "error": str(err)}

        return {
            "i": i,
            "ok": ok,
            "elapsed": time.perf_counter() - start,
            "error": None if ok else data,
            "completion_tokens": data.get("usage", {}).get("completion_tokens") if ok else None,
        }


async def run(args):
    concurrency = args.concurrency or args.num_queries
    semaphore = asyncio.Semaphore(concurrency)
    url = f"http://{args.host}:{args.port}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    # aiohttp's default TCPConnector caps total connections at 100 regardless
    # of the semaphore above; raise it to match the requested concurrency.
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start = time.perf_counter()
        tasks = [
            asyncio.create_task(send_query(session, url, args.model, i, args.max_tokens, semaphore))
            for i in range(args.num_queries)
        ]

        results = []
        with tqdm(total=len(tasks), unit="req") as pbar:
            for coro in asyncio.as_completed(tasks):
                results.append(await coro)
                pbar.update(1)

        total_elapsed = time.perf_counter() - start

    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]

    print(f"Endpoint:        {url}")
    print(f"Queries:         {args.num_queries}")
    print(f"Concurrency:     {concurrency}")
    print(f"Total time:      {total_elapsed:.2f}s")
    print(f"Successes:       {len(successes)}")
    print(f"Failures:        {len(failures)}")

    if successes:
        avg_latency = sum(r["elapsed"] for r in successes) / len(successes)
        print(f"Avg latency:     {avg_latency:.2f}s")
        print(f"Throughput:      {len(successes) / total_elapsed:.2f} req/s")

        token_counts = [r["completion_tokens"] for r in successes if r["completion_tokens"] is not None]
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            hit_limit = sum(1 for t in token_counts if t >= args.max_tokens)
            tokens_per_sec = sum(token_counts) / total_elapsed
            print(f"Avg completion tokens: {avg_tokens:.1f} / {args.max_tokens} max")
            print(f"Hit max-tokens limit:   {hit_limit}/{len(token_counts)} ({100 * hit_limit / len(token_counts):.1f}%)")
            print(f"Tokens/sec:      {tokens_per_sec:.1f}")

            if args.csv:
                file_exists = os.path.exists(args.csv)
                with open(args.csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["run_name", "test", "batch_size", "packed", "bench", "perf"])
                    writer.writerow([args.run_name, "", "", "", args.bench, tokens_per_sec])

    if failures:
        print("\nFirst failures:")
        for r in failures[:5]:
            print(f"  [{r['i']}] {r['error']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fire N concurrent queries at a vLLM server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="model", help="Value passed as --served-model-name to the vLLM server")
    parser.add_argument("-n", "--num-queries", type=int, default=100)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent in-flight requests (defaults to num-queries, i.e. all at once)",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600, help="Per-session total timeout in seconds")
    parser.add_argument("--csv", default=None, help="Append tokens/sec as a row to this CSV (e.g. results.csv)")
    parser.add_argument("--run-name", default="vllm-main", help="Value written to the CSV's run_name column")
    parser.add_argument("--bench", default="vllm-main", help="Value written to the CSV's bench column")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
