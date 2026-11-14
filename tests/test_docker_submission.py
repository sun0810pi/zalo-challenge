"""Unit and conformance tests for ZAC 2026 LifeMate Docker submission layout."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import predict  # noqa: E402


class _FakeLLM:
    def generate(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        assert temperature == 0.0
        # Return empty tool call if already called once, else query smart_home
        if any(m.get("role") == "user" and m.get("content", "").startswith("<tool_response>") for m in messages):
            return "<think>done</think>\n<tool_call>\n</tool_call>"
        return '<think>check status</think>\n<tool_call>\n{"name": "smart_home.get_status", "arguments": {}}\n</tool_call>'


class _FakeTranscriber:
    def transcribe_sample(self, sample: dict, split_dir: Path) -> tuple[str, dict[str, str]]:
        return "kiểm tra trạng thái nhà thông minh", {}


class TestDockerSubmission(unittest.TestCase):
    def test_detect_split_autodetects_private_and_public(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            priv = data_root / "private_test"
            priv.mkdir()
            (priv / "private_test_tools_catalog.json").write_text('{"apps": []}', encoding="utf-8")

            self.assertEqual(predict.detect_split(data_root), "private_test")
            self.assertEqual(predict.detect_split(data_root, preferred="nonexistent"), "private_test")

    def test_resolve_model_path_prefers_local_saved_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            local_llm = base / "saved_models" / "llm"
            local_llm.mkdir(parents=True)

            with patch.object(predict, "CODE_DIR", base):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("MODEL_PATH", None)
                    # Placeholder directory with only .gitkeep must NOT be selected over fallback ID
                    (local_llm / ".gitkeep").write_text("", encoding="utf-8")
                    self.assertEqual(
                        predict.resolve_model_path(None, "MODEL_PATH", ["saved_models/llm"], "Qwen/Qwen3-4B"),
                        "Qwen/Qwen3-4B",
                    )
                    (local_llm / "config.json").write_text("{}", encoding="utf-8")
                    resolved = predict.resolve_model_path(
                        None,
                        "MODEL_PATH",
                        ["saved_models/llm"],
                        "Qwen/Qwen3-4B",
                    )
                    self.assertEqual(Path(resolved).resolve(), local_llm.resolve())

    def test_prepare_saved_models_check(self) -> None:
        from scripts.prepare_saved_models import check_saved_models

        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(tmp) / "saved_models"
            (saved / "llm").mkdir(parents=True)
            (saved / "asr").mkdir(parents=True)
            self.assertFalse(check_saved_models(saved))

            (saved / "llm" / "config.json").write_text("{}", encoding="utf-8")
            (saved / "llm" / "model.safetensors").write_bytes(b"0")
            (saved / "asr" / "model.bin").write_bytes(b"0")
            self.assertTrue(check_saved_models(saved))

    def test_run_predict_end_to_end_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_root = tmp_path / "data"
            split_dir = data_root / "private_test"
            zaic_dir = split_dir / "zaic_tools"
            zaic_dir.mkdir(parents=True)

            (zaic_dir / "__init__.py").write_text(
                "class ToolEnv:\n"
                "    def __init__(self, split, item_id, root='.'):\n"
                "        self.calls = []\n"
                "    def validate(self, name, args):\n"
                "        return {'ok': True}\n"
                "    def call(self, name, args):\n"
                "        self.calls.append({'name': name, 'arguments': args})\n"
                "        return {'ok': True, 'devices': []}\n"
                "    def get_tool_calls(self):\n"
                "        return list(self.calls)\n"
                "    def get_state_hash(self):\n"
                "        return 'hash_abc123'\n",
                encoding="utf-8",
            )
            (zaic_dir / "validate.py").write_text(
                "def catalog_tools(catalog):\n"
                "    out = {}\n"
                "    for app in catalog.get('apps', []):\n"
                "        for t in app.get('tools', []):\n"
                "            out[t['name']] = t\n"
                "    return out\n",
                encoding="utf-8",
            )

            catalog = {
                "apps": [
                    {
                        "id": "smart_home",
                        "name_vi": "Nhà thông minh",
                        "description": "Điều khiển thiết bị",
                        "tools": [
                            {
                                "name": "smart_home.get_status",
                                "kind": "read",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ]
            }
            (split_dir / "private_test_tools_catalog.json").write_text(
                json.dumps(catalog, ensure_ascii=False), encoding="utf-8"
            )
            sample = {
                "id": "priv_0001",
                "audio": "audio/priv_0001/priv_0001.wav",
                "context": {
                    "current_time": "2026-11-20T10:00:00+07:00",
                    "installed_apps": {"smart_home": {"tools": ["smart_home.get_status"]}},
                },
                "history": [],
            }
            (split_dir / "private_test.jsonl").write_text(
                json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            out_file = tmp_path / "result" / "submission.jsonl"
            fake_resources = {
                "transcriber": _FakeTranscriber(),
                "llm": _FakeLLM(),
                "preprocessor": predict.build_preprocessor("safe"),
                "router": predict.SemanticToolRouter(top_k_recommended=4),
                "load_ms": 10,
            }

            # Remove any cached zaic_tools before running
            for mod_name in list(sys.modules):
                if mod_name == "zaic_tools" or mod_name.startswith("zaic_tools."):
                    sys.modules.pop(mod_name, None)

            results = predict.run_predict(
                root=data_root,
                output=out_file,
                resources=fake_resources,
                enable_action_triage=False,
            )

            self.assertTrue(out_file.is_file())
            lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(len(results), 1)
            row = lines[0]
            self.assertEqual(row["id"], "priv_0001")
            self.assertEqual(row["state_hash"], "hash_abc123")
            self.assertIsInstance(row["time"], int)
            self.assertGreaterEqual(row["time"], 0)
            self.assertEqual(row["tool_calls"][0]["name"], "smart_home.get_status")
            # Ensure nothing was written into data_root
            self.assertFalse((data_root / "cache").exists())

    def test_docker_and_notebook_conformance(self) -> None:
        dockerfile_text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("nvidia/cuda:13.2.0-devel-ubuntu22.04", dockerfile_text)
        self.assertIn("WORKDIR /code", dockerfile_text)
        self.assertNotIn("ENTRYPOINT", dockerfile_text)
        self.assertIn("/code/training_code", dockerfile_text)

        reqs_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("jupyterlab", reqs_text.lower())

        dockerignore_text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for excluded in ("train/", "public_test/", "private_test/", "cache/"):
            self.assertIn(excluded, dockerignore_text)

        for sh_name in ("predict.sh", "start_jupyter.sh", "entrypoint.sh"):
            raw_bytes = (ROOT / sh_name).read_bytes()
            self.assertNotIn(b"\r\n", raw_bytes, f"{sh_name} must use LF line endings for Linux Docker")

        nb = json.loads((ROOT / "predict_notebook.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 4)
        joined_src = "\n".join("".join(c.get("source", [])) for c in code_cells)
        self.assertIn("seed_everything(42)", joined_src)
        self.assertIn("/result", joined_src)
        self.assertIn("jupyter_submission.jsonl", joined_src)
        self.assertIn('"time": ms', joined_src)

        self.assertTrue((ROOT / "saved_models" / "llm" / ".gitkeep").is_file())
        self.assertTrue((ROOT / "saved_models" / "asr" / ".gitkeep").is_file())
        self.assertTrue((ROOT / "HANDOFF_DOCKER.md").is_file())

        gitignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in ("saved_models/llm/*", "!saved_models/llm/.gitkeep", "saved_models/asr/*", "!saved_models/asr/.gitkeep", "*.safetensors", "*.tar.gz"):
            self.assertIn(pattern, gitignore_text)


if __name__ == "__main__":
    unittest.main()
