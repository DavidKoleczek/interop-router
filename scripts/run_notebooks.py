"""Execute all example notebooks and save outputs in-place."""

import argparse
import asyncio
from pathlib import Path

import nbclient
import nbformat


async def run_notebook(path: Path, timeout: int, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        print(f"Running {path.name}...")
        nb = nbformat.read(path, as_version=4)
        client = nbclient.NotebookClient(nb, timeout=timeout, kernel_name="python3")
        await client.async_execute()
        nbformat.write(nb, path)
        print(f"  Done: {path.name}")


async def run_all(notebooks: list[Path], timeout: int, max_concurrency: int) -> None:
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [run_notebook(nb, timeout, semaphore) for nb in notebooks]
    await asyncio.gather(*tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute all example notebooks and save outputs in-place.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per notebook in seconds")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Maximum notebooks to run in parallel")
    args = parser.parse_args()

    examples_dir = Path(__file__).parents[1] / "examples"
    notebooks = sorted(examples_dir.glob("*.ipynb"))

    asyncio.run(run_all(notebooks, args.timeout, args.max_concurrency))


if __name__ == "__main__":
    main()
