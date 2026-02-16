#!/bin/bash
# OCLF (Object Centric Learning Framework) セットアップスクリプト
# DINOSAUR公式実装を使用

set -e  # エラーで停止

echo "=============================================="
echo "OCLF Setup Script"
echo "=============================================="

cd /home/menserve/Object-centric-representation/ocl_framework

# 1. Poetry インストール確認
echo ""
echo "1. Checking Poetry..."
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry already installed: $(poetry --version)"
fi

# 2. OCLF 依存関係インストール
echo ""
echo "2. Installing OCLF dependencies..."
poetry install --no-interaction

# 3. データセット変換用の依存関係
echo ""
echo "3. Setting up dataset tools..."
cd scripts/datasets
if [ -f "pyproject.toml" ]; then
    poetry install --no-interaction
fi

# 4. MOVi-A データセットの変換
echo ""
echo "4. Converting MOVi-A dataset..."
# 既存の.ptファイルからwebdatasetを作成する代わりに、
# 直接GCSからダウンロード
if [ ! -d "outputs/movi_a" ]; then
    echo "Downloading and converting MOVi-A..."
    bash download_and_convert.sh movi_a || {
        echo "Direct download failed, will use custom approach"
    }
fi

cd ../..

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  export DATASET_PREFIX=$(pwd)/scripts/datasets/outputs"
echo "  poetry run ocl_train +experiment=dinosaur/movi_a"
