"""
CLI entry point for LifeBench_eval.

Usage:
    python cli.py --dataset lifebench --system lifemem
    python cli.py --dataset lifebench --system lifemem --smoke
    python cli.py --dataset lifebench --system lifemem --stages search answer evaluate
"""
import argparse
import asyncio
import os
import sys
import io
from pathlib import Path

# Fix for Windows ASCII encoding issue with Chinese characters
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Disable asyncio debug mode before any asyncio operations
os.environ["PYTHONASYNCIODEBUG"] = "0"

from dotenv import load_dotenv

project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load LifeBench_eval/.env for LLM_API_KEY, VECTORIZE_API_KEY, etc.
load_dotenv(project_root / ".env")

from src.adapters.registry import create_adapter
from src.builders.registry import create_builder
from src.evaluators.registry import create_evaluator
from src.loaders.registry import load_dataset
from src.pipeline import Pipeline
from src.pipeline.runner_multi_thread import PipelineMultiThread
from src.utils.config import load_yaml, normalize_system_config


def deep_merge_config(base: dict, override: dict) -> dict:
    """Deep merge configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LifeBench Memory System Evaluation Framework")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., lifebench)"
    )
    parser.add_argument(
        "--system", type=str, required=True, help="System name (e.g., lifemem)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run (add, search, answer, evaluate). Default: all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode",
    )
    parser.add_argument(
        "--smoke-messages",
        type=int,
        default=10,
        help="Smoke test: number of messages (default: 10)",
    )
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=3,
        help="Smoke test: number of questions (default: 3)",
    )
    parser.add_argument(
        "--from-conv",
        type=int,
        default=0,
        help="Starting conversation index (inclusive, 0-based). Default: 0",
    )
    parser.add_argument(
        "--to-conv",
        type=int,
        default=None,
        help="Ending conversation index (exclusive). Default: all",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for distinguishing multiple runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/{dataset}-{system}[-{run_name}]",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (detailed ingestion logs)",
    )
    parser.add_argument(
        "--track-cost",
        action="store_true",
        help="Track DeepSeek API cost by querying balance before/after each add",
    )
    parser.add_argument(
        "--multi-thread",
        action="store_true",
        help="Use multi-threaded batch processing runner",
    )

    args = parser.parse_args()

    print("\n[bold cyan]Loading configurations...[/bold cyan]")

    config_root = project_root / "config"

    # Load dataset configuration
    dataset_config_path = config_root / "datasets" / f"{args.dataset}.yaml"
    if not dataset_config_path.exists():
        print(f"[red]❌ Dataset config not found: {dataset_config_path}[/red]")
        return

    dataset_config = load_yaml(str(dataset_config_path))
    print(f"  ✅ Loaded dataset config: {args.dataset}")

    if "memory_language" in dataset_config:
        os.environ["MEMORY_LANGUAGE"] = dataset_config["memory_language"]

    # Load system configuration
    system_config_path = config_root / "systems" / f"{args.system}.yaml"
    if not system_config_path.exists():
        print(f"[red]❌ System config not found: {system_config_path}[/red]")
        return

    system_config = load_yaml(str(system_config_path))
    print(f"  ✅ Loaded system config: {args.system}")

    # Apply dataset-specific overrides
    if (
        "dataset_overrides" in system_config
        and args.dataset in system_config["dataset_overrides"]
    ):
        overrides = system_config["dataset_overrides"][args.dataset]
        system_config = deep_merge_config(system_config, overrides)
        print(f"  🔧 Applied dataset overrides for {args.dataset}")

    try:
        system_config = normalize_system_config(system_config)
    except ValueError as exc:
        print(f"[red]❌ Invalid system config: {exc}[/red]")
        return

    # Run system setup via builder (env script, docker, etc.)
    builder_config = system_config.get("builder")
    builder = None
    if builder_config:
        print("\n[bold cyan]Initializing memory system...[/bold cyan]")
        try:
            builder = create_builder(builder_config, system_config, project_root=str(project_root))
            if not await builder.build():
                print("[red]❌ System initialization failed, aborting[/red]")
                return
            print("[green]✅ System initialized[/green]")
        except Exception as e:
            print(f"[red]❌ Builder error: {e}[/red]")
            return
    else:
        print("\n[yellow]⚠️ No builder configured, skipping system initialization[/yellow]")

    # Load dataset
    print(f"\n[bold cyan]Loading dataset: {args.dataset}[/bold cyan]")

    data_path = dataset_config["data"]["path"]
    if not Path(data_path).is_absolute():
        eval_data_path = config_root.parent / "data" / data_path
        if eval_data_path.exists():
            data_path = eval_data_path
        else:
            data_path = config_root.parent / data_path

    max_content_length = dataset_config.get("data", {}).get("max_content_length", None)

    dataset = load_dataset(
        args.dataset,
        str(data_path),
        max_content_length=max_content_length,
        dataset_format=dataset_config.get("data", {}).get("format"),
    )

    print(
        f"  ✅ Loaded {len(dataset.samples)} conversations, {sum(len(s.qa_pairs) for s in dataset.samples)} QA pairs"
    )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.run_name:
            output_dir = project_root / "results" / f"{args.dataset}-{args.system}-{args.run_name}"
        else:
            output_dir = project_root / "results" / f"{args.dataset}-{args.system}"

    print(f"\n[bold cyan]Initializing components...[/bold cyan]")

    system_config["dataset_name"] = args.dataset

    adapter = create_adapter(
        system_config["adapter"], system_config, output_dir=output_dir
    )
    print(f"  ✅ Created adapter: {adapter.get_system_info()['name']}")

    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"], dataset_config["evaluation"]
    )
    print(f"  ✅ Created evaluator: {evaluator.get_name()}")

    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])

    if args.multi_thread:
        pipeline = PipelineMultiThread(
            adapter=adapter,
            evaluator=evaluator,
            output_dir=output_dir,
            filter_categories=filter_categories,
            debug=args.debug,
            track_cost=args.track_cost,
        )
        print(f"  ✅ Created multi-threaded pipeline")
    else:
        pipeline = Pipeline(
            adapter=adapter,
            evaluator=evaluator,
            output_dir=output_dir,
            filter_categories=filter_categories,
            debug=args.debug,
            track_cost=args.track_cost,
        )

    print(f"  ✅ Created pipeline, output: {output_dir}")

    try:
        await pipeline.run(
            dataset=dataset,
            stages=args.stages,
            smoke_test=args.smoke,
            smoke_messages=args.smoke_messages,
            smoke_questions=args.smoke_questions,
            from_conv=args.from_conv,
            to_conv=args.to_conv,
        )

        print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")

    finally:
        if hasattr(adapter, "close"):
            await adapter.close()
        if hasattr(evaluator, "close"):
            await evaluator.close()
        if builder:
            await builder.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
