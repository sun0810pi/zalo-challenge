"""Official Docker inference entrypoint for ZAC 2026 LifeMate.

Follows Docker_Guide_ZAC2026_LifeMate.pdf specifications:
- Reads split data mounted at /data/<split>/ (auto-detects public_test / private_test)
- Imports zaic_tools dynamically from /data/<split>/ (never from /code)
- Runs 100% offline with fixed deterministic seed (42) and greedy decoding
- Outputs /result/submission.jsonl with {"id", "tool_calls", "state_hash", "time"}
- Exposes shared helpers imported by predict_notebook.ipynb so CLI and notebook
  submissions produce identical predictions.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from src.agent import (  # noqa: E402
    LifeMateAgent,
    LLMBackend,
    ReActSessionState,
    build_preprocessor,
    fix_all_seeds,
    step_react_state,
)
from src.asr import Transcriber  # noqa: E402
from src.router import SemanticToolRouter  # noqa: E402


def seed_everything(seed: int = 42) -> None:
    """Fix all RNG seeds and deterministic flags for reproducible inference."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    fix_all_seeds(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def detect_split(root: Path, preferred: str | None = None) -> str:
    """Auto-detect split directory inside `root` containing `<split>_tools_catalog.json`."""
    if preferred:
        cand = root / preferred
        if cand.is_dir() and (cand / f"{preferred}_tools_catalog.json").is_file():
            return preferred

    if not root.exists():
        raise FileNotFoundError(f"Data root directory does not exist: {root}")

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (child / f"{child.name}_tools_catalog.json").is_file():
            return child.name

    raise FileNotFoundError(
        f"No valid split directory (containing '<split>_tools_catalog.json') found under: {root}"
    )


def _is_populated_model_dir(path: Path) -> bool:
    """Return True if `path` is a directory containing model config or weights (not an empty placeholder)."""
    if not path.is_dir():
        return False
    marker_files = ("config.json", "model.bin", "adapter_config.json")
    if any((path / m).is_file() for m in marker_files):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def resolve_model_path(
    explicit: str | None,
    env_var: str,
    candidate_rel_paths: list[str],
    default_id: str,
) -> str:
    """Resolve model path from CLI arg -> env var -> bundled local directories -> fallback ID."""
    if explicit:
        return explicit
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val

    for base in (CODE_DIR, Path.cwd()):
        for rel in candidate_rel_paths:
            cand = base / rel
            if _is_populated_model_dir(cand):
                return str(cand.resolve())
    return default_id


def resolve_optional_lora(explicit: str | None, env_var: str, candidate_rel_paths: list[str]) -> str | None:
    """Resolve optional LoRA adapter path if provided or bundled locally."""
    if explicit:
        return explicit
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val
    for base in (CODE_DIR, Path.cwd()):
        for rel in candidate_rel_paths:
            cand = base / rel
            if cand.is_dir() and (cand / "adapter_config.json").is_file():
                return str(cand.resolve())
    return None


def _resolve_runtime_paths(
    model_path: str | None = None,
    lora_path: str | None = None,
    asr_model_path: str | None = None,
    asr_lora_path: str | None = None,
    gpu_memory_utilization: float | None = None,
    default_gpu_util: str = "0.65",
) -> dict[str, Any]:
    resolved_llm = resolve_model_path(
        model_path,
        "MODEL_PATH",
        [
            "saved_models/llm",
            "saved_models/qwen3-4b-merged",
            "saved_models/qwen3-4b",
            "models/qwen3-4b-merged",
            "models/qwen3-4b",
        ],
        "Qwen/Qwen3-4B",
    )
    resolved_llm_lora = resolve_optional_lora(
        lora_path,
        "LORA_PATH",
        ["saved_models/llm_lora", "models/llm_lora"],
    )
    resolved_asr = resolve_model_path(
        asr_model_path,
        "ASR_PATH",
        [
            "saved_models/asr",
            "saved_models/whisper-large-v3-turbo",
            "saved_models/whisper",
            "models/whisper-large-v3-turbo",
            "models/whisper",
        ],
        "deepdml/faster-whisper-large-v3-turbo-ct2",
    )
    resolved_asr_lora = resolve_optional_lora(
        asr_lora_path,
        "ASR_LORA_PATH",
        ["saved_models/asr_lora", "models/asr_lora"],
    )
    gpu_util = (
        gpu_memory_utilization
        if gpu_memory_utilization is not None
        else float(os.environ.get("GPU_MEMORY_UTILIZATION", default_gpu_util))
    )
    return {
        "resolved_llm": resolved_llm,
        "resolved_llm_lora": resolved_llm_lora,
        "resolved_asr": resolved_asr,
        "resolved_asr_lora": resolved_asr_lora,
        "gpu_util": gpu_util,
    }


def load_pipeline_resources(
    model_path: str | None = None,
    lora_path: str | None = None,
    asr_model_path: str | None = None,
    asr_lora_path: str | None = None,
    backend: str = "auto",
    preprocessor_mode: str = "safe",
    gpu_memory_utilization: float | None = None,
) -> dict[str, Any]:
    """Load ASR, LLM, preprocessor, and router once; return resources + load time in ms."""
    t_load = time.perf_counter()
    paths = _resolve_runtime_paths(
        model_path=model_path,
        lora_path=lora_path,
        asr_model_path=asr_model_path,
        asr_lora_path=asr_lora_path,
        gpu_memory_utilization=gpu_memory_utilization,
        default_gpu_util="0.65",
    )

    transcriber = Transcriber(
        model_name=paths["resolved_asr"],
        lora_path=paths["resolved_asr_lora"],
        cache_path=None,  # Never write cache into read-only /data mount
        use_gt_audio_text=False,
    )
    # Eagerly load ASR weights during model-load phase so per-sample timing excludes cold model load
    transcriber._load_model()

    llm = LLMBackend(
        model_name_or_path=paths["resolved_llm"],
        lora_path=paths["resolved_llm_lora"],
        backend=backend,
        max_model_len=16384,
        gpu_memory_utilization=paths["gpu_util"],
    )

    preprocessor = build_preprocessor(mode=preprocessor_mode, sim_threshold=0.80)
    router = SemanticToolRouter(top_k_recommended=4)

    load_ms = int((time.perf_counter() - t_load) * 1000)
    print(f"load model: {load_ms} ms")

    return {
        "transcriber": transcriber,
        "llm": llm,
        "preprocessor": preprocessor,
        "router": router,
        "load_ms": load_ms,
        "resolved_llm": paths["resolved_llm"],
        "resolved_asr": paths["resolved_asr"],
    }


def build_sample_agent(
    resources: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    max_steps: int = 8,
    max_apps: int | None = None,
    enable_fuzzy_normalizer: bool = True,
    enable_hybrid_engine: bool = True,
    enable_action_triage: bool = True,
) -> LifeMateAgent:
    """Construct the deterministic LifeMateAgent bound to the split's tool catalog."""
    return LifeMateAgent(
        llm=resources["llm"],
        catalog_tools_map=catalog_tools_map,
        max_steps=max_steps,
        preprocessor=resources["preprocessor"],
        router=resources["router"],
        catalog_apps_map=catalog_apps_map,
        enable_fuzzy_normalizer=enable_fuzzy_normalizer,
        enable_hybrid_engine=enable_hybrid_engine,
        enable_action_triage=enable_action_triage,
        max_apps=max_apps,
    )


def infer_one(
    item: dict[str, Any],
    env: Any,
    transcriber: Transcriber,
    agent: LifeMateAgent,
    split_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    """Run ASR + deterministic ReAct agent on a single sample and ToolEnv."""
    sample_copy = copy.deepcopy(item)
    final_text, hist_transcripts = transcriber.transcribe_sample(sample_copy, split_dir)

    agent.run_sample(
        env=env,
        sample=sample_copy,
        final_user_text=final_text,
        history_transcripts=hist_transcripts,
        temperature=0.0,
    )
    return env.get_tool_calls(), env.get_state_hash()


def _run_batched_vllm_predict(
    test_cases: list[dict[str, Any]],
    resolved_split: str,
    root_path: Path,
    split_dir: Path,
    ToolEnvCls: Any,
    transcriber: Any,
    llm: Any,
    preprocessor: Any,
    router: SemanticToolRouter,
    catalog_tools_map: dict[str, dict[str, Any]],
    catalog_apps_map: dict[str, dict[str, Any]],
    max_steps: int,
    max_apps: int | None,
    enable_fuzzy_normalizer: bool,
    enable_hybrid_engine: bool,
    enable_action_triage: bool,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Run concurrent multi-turn ReAct inference using `ReActSessionState` + `llm.generate_batch`."""
    completed_results: dict[int, dict[str, Any]] = {}
    active_states: list[ReActSessionState] = []
    next_idx = 0

    def _record_done(st: ReActSessionState) -> None:
        completed_results[st.index] = {
            "id": st.sample_id,
            "tool_calls": st.tool_calls,
            "state_hash": st.state_hash,
            "time": max(0, int(st.elapsed * 1000)),
        }

    def _refill() -> None:
        nonlocal next_idx
        while len(active_states) < batch_size and next_idx < len(test_cases):
            idx = next_idx
            item = test_cases[next_idx]
            next_idx += 1

            sample_copy = copy.deepcopy(item)
            env = ToolEnvCls(resolved_split, sample_copy["id"], root=root_path)
            raw_text, raw_hist = transcriber.transcribe_sample(sample_copy, split_dir)
            cl_text = preprocessor.preprocess(raw_text, sample_copy)
            cl_hist = {
                k: preprocessor.preprocess(v, sample_copy)
                for k, v in (raw_hist or {}).items()
            }

            state = ReActSessionState(
                index=idx,
                sample=sample_copy,
                env=env,
                cleaned_user_text=cl_text,
                cleaned_hist_transcripts=cl_hist,
                catalog_tools_map=catalog_tools_map,
                catalog_apps_map=catalog_apps_map,
                router=router,
                max_steps=max_steps,
                enable_fuzzy_normalizer=enable_fuzzy_normalizer,
                enable_hybrid_engine=enable_hybrid_engine,
                enable_action_triage=enable_action_triage,
                max_apps=max_apps,
            )
            if state.finished:
                _record_done(state)
            else:
                active_states.append(state)

    _refill()
    while active_states:
        pending = [s for s in active_states if not s.finished]
        if not pending:
            break
        replies = llm.generate_batch([s.messages for s in pending])
        for st, reply in zip(pending, replies):
            step_react_state(st, reply)
            if st.finished:
                _record_done(st)
        active_states = [s for s in active_states if not s.finished]
        _refill()

    return [completed_results[i] for i in range(len(test_cases))]


def run_predict(
    root: str | Path = "/data",
    split: str | None = None,
    output: str | Path = "/result/submission.jsonl",
    model_path: str | None = None,
    lora_path: str | None = None,
    asr_model_path: str | None = None,
    asr_lora_path: str | None = None,
    backend: str = "auto",
    preprocessor_mode: str = "safe",
    gpu_memory_utilization: float | None = None,
    max_steps: int = 8,
    max_apps: int | None = None,
    limit: int = 0,
    enable_fuzzy_normalizer: bool = True,
    enable_hybrid_engine: bool = True,
    enable_action_triage: bool = True,
    seed: int = 42,
    resources: dict[str, Any] | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Run full deterministic inference and write JSONL output with per-sample `time` (ms)."""
    seed_everything(seed)

    root_path = Path(root)
    resolved_split = detect_split(root_path, preferred=split)
    split_dir = root_path / resolved_split

    split_resolved = str(split_dir.resolve())
    if split_resolved not in sys.path:
        sys.path.insert(0, split_resolved)
    from zaic_tools import ToolEnv  # noqa: E402
    from zaic_tools.validate import catalog_tools  # noqa: E402

    catalog = json.loads((split_dir / f"{resolved_split}_tools_catalog.json").read_text(encoding="utf-8"))
    catalog_tools_map = catalog_tools(catalog)
    catalog_apps_map = {app["id"]: app for app in catalog.get("apps", [])}

    test_cases = [
        json.loads(line)
        for line in (split_dir / f"{resolved_split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > 0:
        test_cases = test_cases[:limit]

    t_phase_start = time.perf_counter()
    if resources is None:
        # Two-phase GPU lifecycle: pre-transcribe all audio -> unload Whisper & re-enable cuDNN -> start LLM
        paths = _resolve_runtime_paths(
            model_path=model_path,
            lora_path=lora_path,
            asr_model_path=asr_model_path,
            asr_lora_path=asr_lora_path,
            gpu_memory_utilization=gpu_memory_utilization,
            default_gpu_util="0.72",
        )
        transcriber = Transcriber(
            model_name=paths["resolved_asr"],
            lora_path=paths["resolved_asr_lora"],
            cache_path=None,  # Never write cache into read-only /data mount
            use_gt_audio_text=False,
        )
        transcriber.pretranscribe_samples(test_cases, split_dir)
        transcriber.unload_model()

        llm = LLMBackend(
            model_name_or_path=paths["resolved_llm"],
            lora_path=paths["resolved_llm_lora"],
            backend=backend,
            max_model_len=16384,
            gpu_memory_utilization=paths["gpu_util"],
        )
        preprocessor = build_preprocessor(mode=preprocessor_mode, sim_threshold=0.80)
        router = SemanticToolRouter(top_k_recommended=4)
        load_ms = int((time.perf_counter() - t_phase_start) * 1000)
        print(f"load model: {load_ms} ms")
        resources = {
            "transcriber": transcriber,
            "llm": llm,
            "preprocessor": preprocessor,
            "router": router,
            "load_ms": load_ms,
        }
    else:
        transcriber = resources["transcriber"]
        llm = resources["llm"]
        preprocessor = resources["preprocessor"]
        router = resources["router"]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"output will be saved in {out_path}")

    total_start = time.perf_counter()
    use_batched = (
        batch_size > 1
        and len(test_cases) > 1
        and getattr(llm, "backend", None) == "vllm"
        and hasattr(llm, "generate_batch")
    )

    if use_batched:
        results = _run_batched_vllm_predict(
            test_cases=test_cases,
            resolved_split=resolved_split,
            root_path=root_path,
            split_dir=split_dir,
            ToolEnvCls=ToolEnv,
            transcriber=transcriber,
            llm=llm,
            preprocessor=preprocessor,
            router=router,
            catalog_tools_map=catalog_tools_map,
            catalog_apps_map=catalog_apps_map,
            max_steps=max_steps,
            max_apps=max_apps,
            enable_fuzzy_normalizer=enable_fuzzy_normalizer,
            enable_hybrid_engine=enable_hybrid_engine,
            enable_action_triage=enable_action_triage,
            batch_size=batch_size,
        )
        with open(out_path, "w", encoding="utf-8") as fout:
            for row in results:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        agent = build_sample_agent(
            resources=resources,
            catalog_tools_map=catalog_tools_map,
            catalog_apps_map=catalog_apps_map,
            max_steps=max_steps,
            max_apps=max_apps,
            enable_fuzzy_normalizer=enable_fuzzy_normalizer,
            enable_hybrid_engine=enable_hybrid_engine,
            enable_action_triage=enable_action_triage,
        )
        results = []
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in test_cases:
                t1 = time.perf_counter()
                env = ToolEnv(resolved_split, item["id"], root=root_path)
                tool_calls, state_hash = infer_one(item, env, transcriber, agent, split_dir)
                ms = int((time.perf_counter() - t1) * 1000)
                row = {
                    "id": item["id"],
                    "tool_calls": tool_calls,
                    "state_hash": state_hash,
                    "time": ms,
                }
                results.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_ms = int((time.perf_counter() - total_start) * 1000)
    avg_ms = total_ms / max(1, len(results))
    print(f"predict: {len(results)} mẫu, {total_ms} ms, trung bình {avg_ms:.1f} ms/mẫu")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Official ZAC 2026 LifeMate Docker inference entrypoint")
    ap.add_argument("--root", default=os.environ.get("DATA_DIR", "/data"))
    ap.add_argument("--split", default=os.environ.get("SPLIT", None))
    ap.add_argument("--output", default=os.environ.get("OUTPUT_FILE", "/result/submission.jsonl"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--asr-model", default=None)
    ap.add_argument("--asr-lora-path", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    ap.add_argument("--preprocessor", default="safe", choices=["safe", "a", "b"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-apps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    ap.add_argument("--no-batched", action="store_true")
    ap.add_argument("--no-fuzzy-normalizer", action="store_true")
    ap.add_argument("--no-hybrid-engine", action="store_true")
    ap.add_argument("--no-action-triage", action="store_true")
    args = ap.parse_args()

    eff_batch_size = 1 if (args.no_batched or os.environ.get("ZAC_SERIAL_LLM") == "1") else args.batch_size
    run_predict(
        root=args.root,
        split=args.split,
        output=args.output,
        model_path=args.model,
        lora_path=args.lora_path,
        asr_model_path=args.asr_model,
        asr_lora_path=args.asr_lora_path,
        backend=args.backend,
        preprocessor_mode=args.preprocessor,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_steps=args.max_steps,
        max_apps=args.max_apps,
        limit=args.limit,
        enable_fuzzy_normalizer=not args.no_fuzzy_normalizer,
        enable_hybrid_engine=not args.no_hybrid_engine,
        enable_action_triage=not args.no_action_triage,
        batch_size=eff_batch_size,
    )


if __name__ == "__main__":
    main()
