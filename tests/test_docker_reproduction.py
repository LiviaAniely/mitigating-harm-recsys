import hashlib
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "reproduction" / "docker" / "generate_splits.py"


def load_split_module():
    spec = importlib.util.spec_from_file_location("generate_splits", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SplitGenerationTests(unittest.TestCase):
    def test_run_zero_matches_existing_split_hashes(self):
        module = load_split_module()
        source = (
            REPO_ROOT
            / "KuaiRand-Harm"
            / "training"
            / "single_and_repeated_interactions_is_click.csv.gzip"
        )
        expected = {
            "train_0_False_0.3.txt": "67cd390e2e12ac59ec7b7c7e97348799ece51a3503d6125db5d8b14e17aa85ec",
            "test_calibration_0_False_0.3.txt": "396758744da675c90b0885c06adcb6276abc5a59484ebe9a6b778f9e9b9cd25f",
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = module.generate_splits(source, Path(tmp), runs=1)
            self.assertEqual({path.name for path in paths}, set(expected))
            for path in paths:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, expected[path.name])

    def test_rejects_non_positive_run_count(self):
        module = load_split_module()
        with self.assertRaisesRegex(ValueError, "runs must be positive"):
            module.generate_splits(Path("unused"), Path("unused"), runs=0)

    def test_truncated_split_is_detected_and_repaired(self):
        module = load_split_module()
        rows = 40
        data = pd.DataFrame(
            {
                "user_id": range(rows),
                "video_id": range(100, 100 + rows),
                "is_hate": [0, 1] * (rows // 2),
                "fraction_play_time": [0.5] * rows,
                "is_hate_y": [0] * rows,
                "fraction_play_time_y": [0.7] * rows,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.csv.gzip"
            output = root / "splits"
            data.to_csv(source, index=False, compression="gzip")
            module.ensure_splits(source, output, runs=1)
            self.assertTrue(module.validate_splits(source, output, runs=1))
            test_path = output / "test_calibration_0_False_0.3.txt"
            test_path.write_text(test_path.read_text().splitlines()[0] + "\n")
            self.assertFalse(module.validate_splits(source, output, runs=1))
            module.ensure_splits(source, output, runs=1)
            self.assertTrue(module.validate_splits(source, output, runs=1))
            with test_path.open() as stream:
                self.assertEqual(sum(1 for _ in stream), 12)


class DockerRuntimeTests(unittest.TestCase):
    def test_runtime_is_cpu_only_and_pinned(self):
        docker_dir = REPO_ROOT / "reproduction" / "docker"
        dockerfile = (docker_dir / "Dockerfile").read_text()
        requirements = (docker_dir / "requirements.txt").read_text()
        self.assertIn("FROM python:3.10-slim", dockerfile)
        self.assertIn("time", dockerfile)
        self.assertIn("util-linux", dockerfile)
        self.assertIn("torch==2.5.0+cpu", requirements)
        for package in ("pandas", "numpy", "scipy", "scikit-learn", "tqdm"):
            self.assertRegex(requirements, rf"(?m)^{package}==")
        forbidden = ("cuda", "torch_geometric", "pyg_lib", "torchvision", "torchaudio")
        for name in forbidden:
            self.assertNotIn(name, requirements.lower())


class ExperimentEntrypointTests(unittest.TestCase):
    def test_entrypoint_has_safe_exact_experiment_contract(self):
        script = (
            REPO_ROOT / "reproduction" / "docker" / "run_experiment.sh"
        ).read_text()
        self.assertIn("set -Eeuo pipefail", script)
        self.assertIn('RUNS="${RUNS:-10}"', script)
        for flag in (
            "--score-model lightgcl",
            "--score-type harm",
            "--method hybrid",
            "--beta 0.0",
            "--epoch 100",
        ):
            self.assertIn(flag, script)
        self.assertIn('cd "$WORKSPACE"', script)
        self.assertIn('if [[ -e "$FINAL_RESULT" ]]', script)
        self.assertIn("flock -n", script)
        self.assertIn("PIPESTATUS", script)
        self.assertIn('ln "$STAGED_RESULT" "$FINAL_RESULT"', script)
        self.assertNotIn('mv "$TEMP_RESULT" "$FINAL_RESULT"', script)
        self.assertIn("tee", script)

    def test_entrypoint_refuses_to_overwrite_existing_result(self):
        script = REPO_ROOT / "reproduction" / "docker" / "run_experiment.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            output.mkdir()
            expected = output / (
                "results_kuairand_hybrid_lightgcl_harm_10_0.0_0.3_100_"
                "False_all.csv"
            )
            expected.write_text("existing")
            result = subprocess.run(
                ["bash", str(script)],
                env={**os.environ, "WORKSPACE": str(root), "OUTPUT_DIR": str(output)},
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Refusing to overwrite", result.stderr)
            self.assertEqual(expected.read_text(), "existing")

    def test_entrypoint_refuses_concurrent_same_run(self):
        script = REPO_ROOT / "reproduction" / "docker" / "run_experiment.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / ".reproduction-10.lock"
            holder = subprocess.Popen(
                ["flock", str(lock_path), "sleep", "10"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                result = subprocess.run(
                    ["bash", str(script)],
                    env={**os.environ, "WORKSPACE": str(root)},
                    text=True,
                    capture_output=True,
                    timeout=5,
                )
            finally:
                holder.terminate()
                holder.wait(timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already running", result.stderr)


class ComposeRunbookTests(unittest.TestCase):
    def test_compose_uses_small_context_bind_mount_and_host_identity(self):
        compose = (REPO_ROOT / "reproduction" / "docker" / "compose.yaml").read_text()
        self.assertIn("reproduction:", compose)
        self.assertNotIn("activity6", compose.lower())
        self.assertNotIn("activity_6", compose.lower())
        self.assertRegex(compose, r"(?m)^\s+context: \.$")
        self.assertIn("../../:/workspace", compose)
        self.assertIn("${LOCAL_UID:-1000}:${LOCAL_GID:-1000}", compose)
        self.assertIn("run_experiment.sh", compose)
        self.assertIn("restart: \"no\"", compose)

    def test_runbook_documents_sudo_detached_progress_and_result(self):
        runbook = (REPO_ROOT / "reproduction" / "docker" / "README.md").read_text()
        self.assertIn("sudo env LOCAL_UID=", runbook)
        self.assertIn("docker compose", runbook)
        self.assertIn("up -d reproduction", runbook)
        self.assertIn("logs -f reproduction", runbook)
        self.assertIn("reproduction/results/main", runbook)
        self.assertNotIn("activity6", runbook.lower())
        self.assertNotIn("activity_6", runbook.lower())


if __name__ == "__main__":
    unittest.main()
