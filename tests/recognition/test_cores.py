"""How many cores a container may actually use.

Every one of these is a regression test. `usable_cores` used to trust
`sched_getaffinity`, on the strength of a docstring calling it "the honest
answer". A rented 32-vCPU pod reported 256 and training spawned 256 workers onto
it, losing 2.6x to contention while every printed number looked fine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mrz_ai.training import recognition
from mrz_ai.training.recognition import _cgroup_quota, usable_cores


@pytest.fixture
def cgroup(tmp_path, monkeypatch):
    """A fake cgroup tree, so these tests say the same thing on any machine."""

    def write(version: str, *, quota: str | None = None, period: str = "100000") -> None:
        if version == "v2":
            (tmp_path / "cpu.max").write_text(f"{quota} {period}\n")
            paths = {"/sys/fs/cgroup/cpu.max": tmp_path / "cpu.max"}
        else:
            (tmp_path / "cfs_quota_us").write_text(f"{quota}\n")
            (tmp_path / "cfs_period_us").write_text(f"{period}\n")
            paths = {
                "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": tmp_path / "cfs_quota_us",
                "/sys/fs/cgroup/cpu/cpu.cfs_period_us": tmp_path / "cfs_period_us",
            }

        real_read = Path.read_text

        def read_text(self, *args, **kwargs):
            if str(self) in paths:
                return real_read(paths[str(self)], *args, **kwargs)
            if str(self).startswith("/sys/fs/cgroup"):
                raise FileNotFoundError(str(self))
            return real_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)

    return write


def _no_cgroup(monkeypatch) -> None:
    monkeypatch.setattr(recognition, "_cgroup_quota", lambda: None)


def test_a_quota_of_thirty_two_cores_is_read_from_cgroup_v2(cgroup) -> None:
    cgroup("v2", quota="3200000")  # 3200000/100000
    assert _cgroup_quota() == 32


def test_a_quota_of_thirty_two_cores_is_read_from_cgroup_v1(cgroup) -> None:
    cgroup("v1", quota="3200000")
    assert _cgroup_quota() == 32


def test_an_unlimited_v2_cgroup_imposes_no_limit(cgroup) -> None:
    cgroup("v2", quota="max")
    assert _cgroup_quota() is None


def test_an_unlimited_v1_cgroup_imposes_no_limit(cgroup) -> None:
    cgroup("v1", quota="-1")
    assert _cgroup_quota() is None


def test_no_cgroup_at_all_imposes_no_limit(monkeypatch) -> None:
    def missing(self, *args, **kwargs):
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(Path, "read_text", missing)
    assert _cgroup_quota() is None


def test_a_fractional_quota_rounds_down_but_never_to_zero(cgroup) -> None:
    cgroup("v2", quota="50000")  # half a core
    assert _cgroup_quota() == 1


def test_the_quota_wins_when_it_is_lower_than_the_affinity_mask(monkeypatch) -> None:
    """The whole bug. The pod that measured 11.5 it/s looked exactly like this."""
    monkeypatch.setattr(os, "cpu_count", lambda: 256)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(256)), raising=False)
    monkeypatch.setattr(recognition, "_cgroup_quota", lambda: 32)
    assert usable_cores() == 32


def test_the_affinity_mask_still_wins_when_it_is_the_lower_one(monkeypatch) -> None:
    """A quota does not make affinity irrelevant — either can be the real limit."""
    monkeypatch.setattr(os, "cpu_count", lambda: 256)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.setattr(recognition, "_cgroup_quota", lambda: 32)
    assert usable_cores() == 8


def test_an_unquota_d_pod_falls_back_to_the_affinity_mask(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 256)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(64)), raising=False)
    _no_cgroup(monkeypatch)
    assert usable_cores() == 64


def test_a_machine_without_affinity_still_answers(monkeypatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    _no_cgroup(monkeypatch)
    assert usable_cores() == 16


def test_it_never_returns_zero_workers(monkeypatch) -> None:
    """Zero would silently mean 'load in the main process' and halve the run."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(recognition, "_cgroup_quota", lambda: 0)
    assert usable_cores() >= 1


def test_the_real_machine_gives_a_sane_answer() -> None:
    """No mocks: whatever is under this test, the number has to be usable."""
    cores = usable_cores()
    assert 1 <= cores <= (os.cpu_count() or 4)
