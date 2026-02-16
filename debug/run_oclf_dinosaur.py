#!/usr/bin/env python3
"""
OCLF (Object Centric Learning Framework) を使用したDINOSAUR実験

このスクリプトは:
1. OCLFをセットアップ
2. MOVi-Cデータセットを準備
3. DINOSAUR実験を実行
"""

import os
import subprocess
import sys
from pathlib import Path

# パス設定
BASE_DIR = Path("/home/menserve/Object-centric-representation")
OCLF_DIR = BASE_DIR / "ocl_framework"
DATASET_DIR = OCLF_DIR / "scripts" / "datasets" / "outputs"


def run_cmd(cmd: str, cwd: str = None, check: bool = True) -> int:
    """コマンドを実行"""
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if check and result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
    return result.returncode


def check_poetry():
    """Poetryがインストールされているか確認"""
    result = subprocess.run("poetry --version", shell=True, capture_output=True)
    if result.returncode != 0:
        print("Poetry not found. Installing...")
        run_cmd("curl -sSL https://install.python-poetry.org | python3 -")
        os.environ["PATH"] = f"{Path.home()}/.local/bin:" + os.environ["PATH"]
    else:
        print(f"Poetry found: {result.stdout.decode().strip()}")


def setup_oclf():
    """OCLFの依存関係をインストール"""
    print("\n" + "="*50)
    print("Setting up OCLF...")
    print("="*50)
    
    os.chdir(OCLF_DIR)
    
    # Poetry環境をセットアップ
    run_cmd("poetry install", cwd=str(OCLF_DIR), check=False)


def prepare_dataset():
    """データセットを準備"""
    print("\n" + "="*50)
    print("Preparing dataset...")
    print("="*50)
    
    dataset_script_dir = OCLF_DIR / "scripts" / "datasets"
    
    # データセット変換スクリプトの依存関係
    if (dataset_script_dir / "pyproject.toml").exists():
        run_cmd("poetry install", cwd=str(dataset_script_dir), check=False)
    
    # MOVi-Cをダウンロード・変換（MOVi-Aは設定がないため）
    movi_c_dir = DATASET_DIR / "movi_c"
    if not movi_c_dir.exists():
        print("Downloading MOVi-C dataset...")
        run_cmd("bash download_and_convert.sh movi_c", cwd=str(dataset_script_dir), check=False)
    else:
        print(f"Dataset already exists at {movi_c_dir}")


def run_dinosaur_experiment():
    """DINOSAUR実験を実行"""
    print("\n" + "="*50)
    print("Running DINOSAUR experiment...")
    print("="*50)
    
    os.chdir(OCLF_DIR)
    os.environ["DATASET_PREFIX"] = str(DATASET_DIR)
    
    # 短いテスト実行（10エポック）
    cmd = """poetry run ocl_train \
        +experiment=projects/bridging/dinosaur/movi_c_feat_rec \
        trainer.max_epochs=10 \
        trainer.check_val_every_n_epoch=5 \
        trainer.log_every_n_steps=10 \
        datamodule.train_shards='{000000..000009}.tar' \
        datamodule.val_shards='{000000..000001}.tar' \
    """
    
    run_cmd(cmd, cwd=str(OCLF_DIR), check=False)


def main():
    print("="*60)
    print("OCLF DINOSAUR Experiment Runner")
    print("="*60)
    
    # 1. Poetryチェック
    print("\n[Step 1/4] Checking Poetry...")
    check_poetry()
    
    # 2. OCLFセットアップ
    print("\n[Step 2/4] Setting up OCLF...")
    setup_oclf()
    
    # 3. データセット準備
    print("\n[Step 3/4] Preparing dataset...")
    prepare_dataset()
    
    # 4. 実験実行
    print("\n[Step 4/4] Running experiment...")
    run_dinosaur_experiment()
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
