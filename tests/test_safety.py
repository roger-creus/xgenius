"""Tests for the safety validator."""

import json
import os
import tempfile

import pytest

from xgenius.config import load_config, XGeniusConfig, SafetyConfig, ClusterConfig, SlurmConfig, WatcherConfig, ProjectConfig
from xgenius.safety import SafetyValidator


def _make_config(safety_overrides=None, tmp_dir=None) -> XGeniusConfig:
    """Create a test config with optional safety overrides."""
    safety = SafetyConfig(**(safety_overrides or {}))
    config = XGeniusConfig(
        project=ProjectConfig(name="test"),
        safety=safety,
        watcher=WatcherConfig(),
        clusters={
            "test-cluster": ClusterConfig(
                name="test-cluster",
                hostname="test",
                username="testuser",
                project_path="/home/testuser/project",
                scratch_path="/scratch/testuser",
                image_path="/scratch/testuser/images",
                slurm=SlurmConfig(num_gpus=1, num_cpus=8, memory="32G", walltime="12:00:00"),
            )
        },
        config_path=os.path.join(tmp_dir or tempfile.mkdtemp(), "xgenius.toml"),
    )
    return config


class TestCommandValidation:
    def test_allowed_python_command(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python train.py --lr 0.001")
        assert result.allowed

    def test_allowed_python_module(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python -m pytest tests/")
        assert result.allowed

    def test_reject_bash_command(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("bash script.sh")
        assert not result.allowed
        assert "must start with" in result.reason.lower()

    def test_reject_rm_rf(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python -c 'import os; os.system(\"rm -rf /\")'")
        assert not result.allowed
        assert "rm -rf" in result.reason.lower()

    def test_reject_sudo(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python -c 'import os; os.system(\"sudo apt install foo\")'")
        assert not result.allowed

    def test_reject_shell_injection_semicolon(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python train.py; rm -rf /")
        assert not result.allowed

    def test_reject_shell_injection_pipe(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python train.py | cat /etc/passwd")
        assert not result.allowed

    def test_reject_shell_injection_backtick(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python train.py `whoami`")
        assert not result.allowed

    def test_reject_shell_injection_dollar_paren(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("python train.py $(cat /etc/passwd)")
        assert not result.allowed

    def test_reject_empty_command(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_command("")
        assert not result.allowed

    def test_custom_allowed_prefixes(self):
        config = _make_config({"allowed_command_prefixes": ["python", "julia"]})
        v = SafetyValidator(config)
        assert v.validate_command("julia script.jl").allowed
        assert v.validate_command("python train.py").allowed
        assert not v.validate_command("bash run.sh").allowed


class TestResourceValidation:
    def test_within_limits(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_job_submission(num_gpus=1, num_cpus=8, memory="32G", walltime="12:00:00")
        assert result.allowed

    def test_exceed_gpu_limit(self):
        config = _make_config({"max_gpus_per_job": 2})
        v = SafetyValidator(config)
        result = v.validate_job_submission(num_gpus=4, num_cpus=8, memory="32G", walltime="12:00:00")
        assert not result.allowed
        assert "GPU" in result.reason

    def test_exceed_cpu_limit(self):
        config = _make_config({"max_cpus_per_job": 8})
        v = SafetyValidator(config)
        result = v.validate_job_submission(num_gpus=1, num_cpus=16, memory="32G", walltime="12:00:00")
        assert not result.allowed
        assert "CPU" in result.reason

    def test_exceed_memory_limit(self):
        config = _make_config({"max_memory_per_job": "32G"})
        v = SafetyValidator(config)
        result = v.validate_job_submission(num_gpus=1, num_cpus=8, memory="128G", walltime="12:00:00")
        assert not result.allowed
        assert "memory" in result.reason.lower()

    def test_exceed_walltime_limit(self):
        config = _make_config({"max_walltime": "12:00:00"})
        v = SafetyValidator(config)
        result = v.validate_job_submission(num_gpus=1, num_cpus=8, memory="32G", walltime="48:00:00")
        assert not result.allowed
        assert "walltime" in result.reason.lower()


class TestPathValidation:
    def test_valid_path(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_path("src/train.py", "/home/user/project")
        assert result.allowed

    def test_path_escape(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_path("../../etc/passwd", "/home/user/project")
        assert not result.allowed

    def test_absolute_path_outside(self):
        config = _make_config()
        v = SafetyValidator(config)
        result = v.validate_path("/etc/passwd", "/home/user/project")
        assert not result.allowed


class TestAuditLogging:
    def test_log_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp_dir=tmp)
            xgenius_dir = os.path.join(tmp, ".xgenius")
            os.makedirs(xgenius_dir, exist_ok=True)
            config.config_path = os.path.join(tmp, "xgenius.toml")

            v = SafetyValidator(config)
            result = v.validate_command("python train.py")
            v.log_action("test_action", {"command": "python train.py"}, result)

            entries = v.get_audit_log()
            assert len(entries) == 1
            assert entries[0]["action"] == "test_action"
            assert entries[0]["allowed"] is True


class TestConfigLoading:
    def test_load_valid_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "xgenius.toml")
            import tomli_w
            data = {
                "project": {"name": "test-project"},
                "safety": {"max_gpus_per_job": 2},
                "clusters": {
                    "mycluster": {
                        "hostname": "mycluster.edu",
                        "username": "testuser",
                        "project_path": "/home/testuser/project",
                        "scratch_path": "/scratch/testuser",
                    }
                },
            }
            with open(config_path, "wb") as f:
                tomli_w.dump(data, f)

            config = load_config(config_path)
            assert config.project.name == "test-project"
            assert config.safety.max_gpus_per_job == 2
            assert "mycluster" in config.clusters
            assert config.clusters["mycluster"].username == "testuser"

    def test_missing_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/xgenius.toml")

    def test_missing_required_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "xgenius.toml")
            import tomli_w
            data = {
                "clusters": {
                    "bad": {
                        "hostname": "test",
                        "username": "",  # empty
                        "project_path": "/abs/path",
                        "scratch_path": "/abs/scratch",
                    }
                }
            }
            with open(config_path, "wb") as f:
                tomli_w.dump(data, f)

            with pytest.raises(ValueError):
                load_config(config_path)

    def test_relative_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "xgenius.toml")
            import tomli_w
            data = {
                "clusters": {
                    "bad": {
                        "hostname": "test",
                        "username": "user",
                        "project_path": "relative/path",  # not absolute
                        "scratch_path": "/abs/scratch",
                    }
                }
            }
            with open(config_path, "wb") as f:
                tomli_w.dump(data, f)

            with pytest.raises(ValueError, match="absolute"):
                load_config(config_path)
