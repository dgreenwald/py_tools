from subprocess import CompletedProcess

import pytest
from py_tools import cluster


def test_write_single_job_script(tmp_path):
    job = cluster.SlurmJob(
        name="example-final",
        command=("python", "worker.py", "--input", tmp_path / "input file.csv"),
        workdir=tmp_path / "repo dir",
        log_dir=tmp_path / "logs",
        resources=cluster.SlurmResources(
            time="04:00:00", memory="16G", account="research", cpus_per_task=2
        ),
        activate=tmp_path / "venv" / "bin" / "activate",
    )

    script = cluster.write_slurm_script(job, tmp_path / "job.slurm")
    contents = script.read_text()

    assert "#SBATCH --job-name=example-final" in contents
    assert "#SBATCH --account=research" in contents
    assert "#SBATCH --cpus-per-task=2" in contents
    assert f"#SBATCH --output={tmp_path}/logs/%x_%j.out" in contents
    assert "#SBATCH --array=" not in contents
    assert f"source {tmp_path}/venv/bin/activate" in contents
    assert f"cd '{tmp_path}/repo dir'" in contents
    assert f"--input '{tmp_path}/input file.csv'" in contents
    assert script.stat().st_mode & 0o111


def test_write_capped_array_expands_only_task_id(tmp_path):
    job = cluster.SlurmJob(
        name="example-array",
        command=(
            "/usr/bin/time",
            "-v",
            "python",
            "worker.py",
            "--job-index",
            cluster.SLURM_ARRAY_TASK_ID,
        ),
        workdir=tmp_path,
        log_dir=tmp_path / "logs",
        resources=cluster.SlurmResources(time="08:00:00", memory="32G"),
        array=cluster.SlurmArray(task_count=108, max_concurrent=7),
    )

    contents = cluster.write_slurm_script(job, tmp_path / "array.slurm").read_text()

    assert "#SBATCH --array=0-107%7" in contents
    assert f"#SBATCH --output={tmp_path}/logs/%x_%A_%a.out" in contents
    assert '--job-index "${SLURM_ARRAY_TASK_ID}"' in contents


@pytest.mark.parametrize(
    "factory, match",
    [
        (lambda: cluster.SlurmArray(0), "task_count"),
        (lambda: cluster.SlurmArray(1, 0), "max_concurrent"),
        (lambda: cluster.SlurmResources("1:00", "1G", cpus_per_task=0), "cpus"),
        (lambda: cluster.EnvironmentVariable("bad-name"), "environment-variable"),
    ],
)
def test_typed_values_reject_invalid_inputs(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_job_rejects_empty_command_and_variable_paths(tmp_path):
    resources = cluster.SlurmResources(time="1:00:00", memory="1G")
    with pytest.raises(ValueError, match="command"):
        cluster.SlurmJob("empty", (), tmp_path, tmp_path, resources)
    with pytest.raises(ValueError, match="concrete"):
        cluster.SlurmJob("variables", ("true",), "$HOME/repo", tmp_path, resources)


def test_writer_rejects_variable_destination(tmp_path):
    job = cluster.SlurmJob(
        "example",
        ("true",),
        tmp_path,
        tmp_path,
        cluster.SlurmResources(time="1:00:00", memory="1G"),
    )
    with pytest.raises(ValueError, match="concrete"):
        cluster.write_slurm_script(job, "$HOME/job.slurm")


def test_submit_slurm_returns_structured_parsable_result(tmp_path, monkeypatch):
    script = tmp_path / "job.slurm"
    script.write_text("#!/bin/bash\n")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, "12345;cluster-a\n", "")

    monkeypatch.setattr(cluster.subprocess, "run", fake_run)
    result = cluster.submit_slurm(script)

    assert result == cluster.SlurmSubmission("12345", "cluster-a", "12345;cluster-a")
    assert calls == [
        (
            ["sbatch", "--parsable", str(script.resolve())],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_submit_slurm_rejects_missing_script(tmp_path):
    with pytest.raises(FileNotFoundError):
        cluster.submit_slurm(tmp_path / "missing.slurm")


def test_submit_slurm_accepts_job_id_without_cluster(tmp_path, monkeypatch):
    script = tmp_path / "job.slurm"
    script.write_text("#!/bin/bash\n")
    monkeypatch.setattr(
        cluster.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "12345\n", ""),
    )

    assert cluster.submit_slurm(script) == cluster.SlurmSubmission(
        "12345", None, "12345"
    )
