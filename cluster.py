"""Typed helpers for generating and submitting Slurm batch jobs."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class EnvironmentVariable:
    """A shell environment variable that should expand in a rendered command."""

    name: str

    def __post_init__(self) -> None:
        if not _ENVIRONMENT_NAME.fullmatch(self.name):
            raise ValueError(f"Invalid environment-variable name {self.name!r}")


SLURM_ARRAY_TASK_ID = EnvironmentVariable("SLURM_ARRAY_TASK_ID")
CommandArgument: TypeAlias = str | os.PathLike[str] | EnvironmentVariable

__all__ = (
    "SLURM_ARRAY_TASK_ID",
    "EnvironmentVariable",
    "SlurmArray",
    "SlurmJob",
    "SlurmResources",
    "SlurmSubmission",
    "submit_slurm",
    "write_slurm_script",
)


@dataclass(frozen=True)
class SlurmResources:
    """Core resources requested for one Slurm job."""

    time: str
    memory: str
    account: str | None = None
    cpus_per_task: int | None = None

    def __post_init__(self) -> None:
        _validate_directive(self.time, "time")
        _validate_directive(self.memory, "memory")
        if self.account is not None:
            _validate_directive(self.account, "account")
        if self.cpus_per_task is not None and self.cpus_per_task <= 0:
            raise ValueError("cpus_per_task must be positive")


@dataclass(frozen=True)
class SlurmArray:
    """A contiguous, zero-indexed Slurm job array."""

    task_count: int
    max_concurrent: int | None = None

    def __post_init__(self) -> None:
        if self.task_count <= 0:
            raise ValueError("task_count must be positive")
        if self.max_concurrent is not None and self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")

    @property
    def directive(self) -> str:
        value = f"0-{self.task_count - 1}"
        if self.max_concurrent is not None:
            value += f"%{self.max_concurrent}"
        return value


@dataclass(frozen=True)
class SlurmJob:
    """A batch command and the Slurm resources needed to execute it."""

    name: str
    command: tuple[CommandArgument, ...]
    workdir: str | os.PathLike[str]
    log_dir: str | os.PathLike[str]
    resources: SlurmResources
    activate: str | os.PathLike[str] | None = None
    array: SlurmArray | None = None

    def __post_init__(self) -> None:
        _validate_directive(self.name, "job name")
        if not self.command:
            raise ValueError("command cannot be empty")
        _concrete_path(self.workdir, "workdir")
        _concrete_path(self.log_dir, "log_dir")
        if self.activate is not None:
            _concrete_path(self.activate, "activate")


@dataclass(frozen=True)
class SlurmSubmission:
    """Structured result returned by ``sbatch --parsable``."""

    job_id: str
    cluster: str | None
    stdout: str


def write_slurm_script(job: SlurmJob, destination: str | os.PathLike[str]) -> Path:
    """Render *job* to an executable Slurm script and return its path."""
    destination = _concrete_path(destination, "destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_dir = _concrete_path(job.log_dir, "log_dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    workdir = _concrete_path(job.workdir, "workdir")

    directives = [
        f"#SBATCH --time={job.resources.time}",
        f"#SBATCH --job-name={job.name}",
    ]
    if job.resources.account is not None:
        directives.append(f"#SBATCH --account={job.resources.account}")
    log_pattern = "%x_%A_%a" if job.array is not None else "%x_%j"
    directives.extend(
        [
            f"#SBATCH --output={log_dir.as_posix()}/{log_pattern}.out",
            f"#SBATCH --error={log_dir.as_posix()}/{log_pattern}.err",
            f"#SBATCH --mem={job.resources.memory}",
        ]
    )
    if job.resources.cpus_per_task is not None:
        directives.append(f"#SBATCH --cpus-per-task={job.resources.cpus_per_task}")
    if job.array is not None:
        directives.append(f"#SBATCH --array={job.array.directive}")

    body = ["#!/bin/bash", *directives, "", "set -euo pipefail"]
    if job.activate is not None:
        activate = _concrete_path(job.activate, "activate")
        body.append(f"source {shlex.quote(str(activate))}")
    body.append(f"cd {shlex.quote(str(workdir))}")
    body.append(_render_command(job.command))
    _atomic_script(destination, "\n".join(body) + "\n")
    return destination


def submit_slurm(script: str | os.PathLike[str]) -> SlurmSubmission:
    """Submit a generated script with ``sbatch --parsable``."""
    script = Path(script).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Slurm script not found: {script}")
    result = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("sbatch returned an empty job identifier")
    job_id, separator, cluster_name = stdout.partition(";")
    if not job_id:
        raise RuntimeError(f"Invalid sbatch response: {stdout!r}")
    return SlurmSubmission(job_id, cluster_name if separator else None, stdout)


def _render_command(command: tuple[CommandArgument, ...]) -> str:
    rendered = []
    for argument in command:
        if isinstance(argument, EnvironmentVariable):
            rendered.append(f'"${{{argument.name}}}"')
        else:
            rendered.append(shlex.quote(os.fspath(argument)))
    return " ".join(rendered)


def _validate_directive(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} must be a nonempty single-line string")


def _concrete_path(value: str | os.PathLike[str], name: str) -> Path:
    text = os.fspath(value)
    if "$" in text:
        raise ValueError(f"{name} must be concrete; environment variables are unsupported")
    if "\n" in text or "\r" in text:
        raise ValueError(f"{name} must be a single-line path")
    return Path(text).expanduser().resolve()


def _atomic_script(destination: Path, contents: str) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(contents)
            file.flush()
            os.fsync(file.fileno())
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
