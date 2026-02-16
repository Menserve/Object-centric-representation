# Opus 監査結果 & 最終整理手順書（Sonnet引き継ぎ用）

> **作成**: 2026-02-16 Opus による監査  
> **目的**: Sonnet が実施した整理作業の監査結果と、残作業の手順書

---

## 1. 監査結果: Sonnetの作業

### ✅ 正しく実行された項目

| ステップ | 内容 | 状態 |
|---------|------|------|
| Step 1 | ディレクトリ作成 (experiments/main/logs, sub/logs, analysis, debug, docs/paper, docs/research_log, notebooks/phase1_exploration) | ✅ 完了 |
| Step 2 | Tier 1 ログ 8ファイル → experiments/main/logs/ | ✅ 完了（全8ファイル確認済） |
| Step 3 | Tier 2 ログ 10ファイル → experiments/sub/logs/ | ✅ 完了（全10ファイル確認済） |
| Step 4 | Tier 3 ログ 31ファイル削除 + logs/ ディレクトリ削除 | ✅ 完了 |
| Step 5 | .gitignore 更新（`!experiments/*/logs/*.log`追加） | ✅ 完了（`git add -n` で追跡可能を確認） |
| Step 6 | Python/Shellスクリプトの分類移動 | ✅ 概ね完了 |
| + α | READMEファイル作成（experiments/main, sub, analysis, debug） | ✅ 良い判断 |
| + α | トップレベルREADME.mdのディレクトリツリー更新 | ✅ 完了 |
| + α | docs/tex/ → docs/paper/ 移動 | ✅ 完了 |
| + α | docs/*.md → docs/research_log/ 移動 | ✅ 完了 |
| + α | 旧 docs/figures/ 削除（dog実験の画像） | ✅ 正しい判断 |

### ⚠️ 問題点（要修正 5件）

#### 問題1: ルートに5つの未整理ファイルが残存

```
analyze_final_model.py        → experiments/analysis/ へ移動
compare_normalized_backbones.py → experiments/analysis/ へ移動
diagnose_backbones.py          → experiments/analysis/ へ移動
run_oclf_dinosaur.py           → debug/ へ移動（OCLF試行用、未使用）
setup_oclf.sh                  → debug/ へ移動（OCLF試行用、未使用）
```

#### 問題2: src/archive/ が未処理

Phase 0 の探索ノートブック（ex_slot*.ipynb, ex_dino*.ipynb）とテストファイル。
3.4MB。`notebooks/phase1_exploration/archive/` へ移動すべき。

#### 問題3: src/__pycache__/ が残存

削除すべき。.gitignore に含まれているが、ディスク上に残っている。

#### 問題4: docs/paper/ に LaTeX ビルド成果物が混在

以下は .gitignore に追加して git 追跡から除外すべき:
- `*.aux`, `*.dvi`, `*.synctex.gz`
- LaTeX ログ（`paper.log`, `slide.log`, `guide-2e.log`, `guide-ntt.log`）

#### 問題5: git コミットが一切行われていない

全ての変更が uncommitted のまま。テーマ別コミットが必要。

---

## 2. Sonnetの提案の評価

### checkpoints/ 整理について

Sonnet は「保持/検討/削除候補」の3分類を提案したが、**具体的なディレクトリ分類リストを示さなかった**。

**Opusの判断**: checkpoints/ は .gitignore で既に除外されており、**gitコミットには影響しない**。ディスクの整理として有用だが、git履歴整理の優先度は低い。ユーザーが12GBのディスク容量を気にするなら別途対応。**今回のスコープ外**とする。

---

## 3. 残作業の実行手順（Sonnet向け）

### Step A: 未整理ファイルの移動

```bash
cd /home/menserve/Object-centric-representation

# ルートの迷子ファイルを分類
mv analyze_final_model.py experiments/analysis/
mv compare_normalized_backbones.py experiments/analysis/
mv diagnose_backbones.py experiments/analysis/
mv run_oclf_dinosaur.py debug/
mv setup_oclf.sh debug/

# src/archive/ を notebooks 配下に移動
mv src/archive/ notebooks/phase1_exploration/archive/

# __pycache__ 削除
rm -rf src/__pycache__/
```

### Step B: .gitignore にLaTeXビルド成果物を追加

`.gitignore` の末尾に以下を追加:

```gitignore
# LaTeX build artifacts
*.aux
*.dvi
*.synctex.gz
*.fls
*.fdb_latexmk
*.xbb
docs/paper/*.log
docs/paper/paper_old.tex
docs/paper/paper_draft_*.pdf
docs/paper/paper_1221.pdf
docs/paper/guide-*
docs/paper/data.tex
docs/paper/final_assignment.txt
```

### Step C: テーマ別 git コミット

**5つのコミットに分けて実行する。順序が重要。**

```bash
# コミット1: プロジェクト基盤（コアモジュール + 設定ファイル）
git add src/savi_dinosaur.py src/train_movi.py src/compute_ari.py
git add .gitignore README.md requirements.txt pyproject.toml
git commit -m "refactor: reorganize project structure and update core modules

- src/ now contains only 3 core modules: savi_dinosaur.py, train_movi.py, compute_ari.py
- Updated .gitignore to track experiment logs, exclude LaTeX artifacts
- Updated README.md with new directory structure"

# コミット2: 論文直結の実験ログと実行スクリプト
git add experiments/main/
git commit -m "feat: add main experiment logs and scripts

- 8 paper-cited training/evaluation logs (Tier 1)
- ARI evaluation results for DINOv2/DINOv1/CLIP (Table 1, 2)
- Training scripts for backbone comparison (Phase 2)"

# コミット3: サブ実験・アブレーション
git add experiments/sub/
git commit -m "feat: add sub-experiment logs and ablation scripts

- 10 supporting experiment logs (Tier 2)
- Before/after comparison logs (detach fix, ch-norm selection)
- Video mode and decoder variation experiments"

# コミット4: 分析ツール + デバッグ
git add experiments/analysis/ debug/
git commit -m "feat: add analysis tools and debug scripts

- Analysis scripts for metal/rubber comparison, backbone comparison
- Debug tools for attention maps, decoder logits, feature inspection
- Experiment report notebook"

# コミット5: 論文 + 研究記録 + ノートブック
git add docs/ notebooks/
git commit -m "docs: add paper source, research logs, and exploration notebooks

- JSAI paper LaTeX source with figures (docs/paper/)
- Research activity logs and handoff documents (docs/research_log/)
- Phase 1 exploration notebooks (notebooks/phase1_exploration/)"
```

### Step D: 検証

```bash
# コミット履歴を確認
git --no-pager log --oneline

# ワーキングツリーがクリーンか確認
git status

# リモートにはまだ push しない（ユーザー確認後）
```

---

## 4. 注意事項

1. **`git add` で削除も記録される**: `docs/METHODS.md` 等の旧パスからの削除は、コミット1で `.gitignore` と `README.md` をaddした時点で自動的に含まれる（modified/deleted として）。ただし `git add` は明示的に旧パスもステージする必要がある場合がある。`git add -A` は使わないこと（全ファイルが一括で入ってしまう）。
2. **コミット1** では旧パスの削除も含めるため、`git add` の後に `git status` でステージング内容を確認してからコミットすること。
3. **checkpoints/（12GB）** と **data/（631MB）** は .gitignore 済みなので触れない。
4. **ocl_framework/** も .gitignore 済み。submodule化は将来課題。
