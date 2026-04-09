#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from kaiwudrl.common.utils.train_test_utils import run_train_test

# To run the train_test, you must modify the algorithm name here. It must be one of algorithm_name_list.
# Simply modify the value of the algorithm_name variable.
# 运行train_test前必须修改这里的算法名字, 必须是 algorithm_name_list 里的一个, 修改algorithm_name的值即可
algorithm_name_list = ["ppo", "diy"]
algorithm_name = "ppo"


# Baseline defaults from official template / 官方模板默认参数
BASELINE_ENV_VARS = {
    "replay_buffer_capacity": "10",
    "preload_ratio": "0.2",
    "train_batch_size": "2",
    "dump_model_freq": "1",
}

# Keep original resume behavior configurable / 保留原有续训能力（可开关）
RESUME_TRAINING = os.environ.get("RESUME_TRAINING", "1").strip().lower() in {"1", "true", "yes", "on"}
BACKUP_MODEL_DIRS = [
    p.strip()
    for p in os.environ.get("BACKUP_MODEL_DIRS", "").split(os.pathsep)
    if p.strip()
]
if not BACKUP_MODEL_DIRS:
    BACKUP_MODEL_DIRS = [
        "backup_model",
        "ckpt/backup_model",
        "/workspace/backup_model",
        "/data/projects/gorge_chase/backup_model",
    ]


def _extract_step_from_name(name):
    match = re.search(r"model\.ckpt-(\d+)\.pkl$", str(name))
    if not match:
        return None
    return int(match.group(1))


def _iter_backup_candidates(root_dir):
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        return

    for pkl_file in root.rglob("*.pkl"):
        step = _extract_step_from_name(pkl_file.name)
        if step is None:
            continue
        yield {
            "step": step,
            "kind": "pkl",
            "path": pkl_file,
            "mtime": pkl_file.stat().st_mtime,
        }

    for zip_file in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                for member in zf.namelist():
                    step = _extract_step_from_name(member)
                    if step is None:
                        continue
                    yield {
                        "step": step,
                        "kind": "zip",
                        "path": zip_file,
                        "member": member,
                        "mtime": zip_file.stat().st_mtime,
                    }
        except zipfile.BadZipFile:
            continue


def _select_latest_backup_model():
    candidates = []
    for backup_dir in BACKUP_MODEL_DIRS:
        candidates.extend(_iter_backup_candidates(backup_dir) or [])

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x["step"], x["mtime"]), reverse=True)
    return candidates[0]


def _restore_checkpoint_to_agent_ckpt(algorithm, backup):
    agent_ckpt_dir = Path(f"agent_{algorithm}") / "ckpt"
    agent_ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = int(backup["step"])
    target_file = agent_ckpt_dir / f"model.ckpt-{step}.pkl"

    if backup["kind"] == "pkl":
        src_file = Path(backup["path"])
        if src_file.resolve() != target_file.resolve():
            shutil.copy2(src_file, target_file)
        return step, target_file

    with tempfile.TemporaryDirectory(prefix="resume_ckpt_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(backup["path"], "r") as zf:
            zf.extract(backup["member"], path=tmp_path)
        extracted = tmp_path / backup["member"]
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted, target_file)
    return step, target_file


def _build_env_vars_with_resume():
    env_vars = dict(BASELINE_ENV_VARS)

    if not RESUME_TRAINING:
        print("[train_test] RESUME_TRAINING disabled, start from scratch")
        return env_vars

    backup = _select_latest_backup_model()
    if backup is None:
        print("[train_test] no backup checkpoint found, start from scratch")
        return env_vars

    step, restored_file = _restore_checkpoint_to_agent_ckpt(algorithm_name, backup)
    env_vars.update(
        {
            "preload_model": "true",
            "preload_model_dir": f"agent_{algorithm_name}/ckpt",
            "preload_model_id": str(step),
        }
    )
    print(
        f"[train_test] resume from model.ckpt-{step}.pkl "
        f"(source: {backup['path']}, restored: {restored_file})"
    )
    return env_vars


if __name__ == "__main__":
    env_vars = _build_env_vars_with_resume()
    run_train_test(
        algorithm_name=algorithm_name,
        algorithm_name_list=algorithm_name_list,
        env_vars=env_vars,
    )
