# Research Log: GameAgent

## 2026-03-23 — Project Initialization

### Bootstrap
- Reviewed full codebase: 13 scripts, 4 src modules, 2 config files
- Established server connection: 4× H800 80GB at 222.223.106.147
- Found partial old results: safety agent SFT checkpoint + 670/2500 GRPO steps

### Critical Finding: Reward Saturation
Analyzed server logs from old GameRefine pipeline:
- Reward saturated to 1.0 by GRPO step ~300
- `completions/clipped_ratio` = 1.0 (all outputs hit max_tokens)
- Loss collapsed to ~0, training became vacuous
- Root cause: heuristic reward functions (keyword count, word length) are trivially gameable

### Decision: Reward Function Overhaul (P0)
Must replace heuristic rewards before ANY training:
- Correctness: word-overlap → NLI entailment (DeBERTa-v3) + semantic similarity
- Safety: 12-word blacklist → multi-level pattern detection with severity weighting
- Efficiency: word-count buckets → information density (windowed TTR + coverage)
- Creativity: unique-word ratio → sliding-window TTR + sentence variety + structure

### Bugs Fixed
1. `eval_game_performance.py`: wrong import path
2. `train_agents.py`: truncated file restored
3. `eval_benchmarks.py`: dead import removed

### Infrastructure
- Created `scripts/sync.sh` for three-way sync (local/server/GitHub)
- Verified server environment and GPU availability

## 2026-03-23 — Phase 0 Execution

### Reward Function Overhaul (Completed)
- Created `src/reward_models.py` with robust reward functions
- Added anti-reward-hacking: length penalty + repetition penalty
- Updated `game_protocol.py` to use robust rewards (backward-compatible fallback)

### Visualization Pipeline (Completed)
- Created `src/visualization.py` with 6 plot types
- Created `scripts/collect_and_visualize.py` with HTML progress report generation

### Paper Preparation (Started)
- Created `paper/outline.md` — full NeurIPS paper outline

## 2026-03-23 — Project Merge (GameRefine → GameAgent)

### Audit Results
Identified 4 GameRefine legacy components vs GameAgent unified components:
- `train_agents.py` → `train_sft_agents.py` (reward-based filtering migrated)
- `run_self_play.py` → `train_nash_dpo.py`
- `run_gamerefine.sh` → `run_all_experiments.sh --skip_grpo`
- `eval_gamerefine.py` → `eval_benchmarks.py` (majority_vote → game_protocol.py)

### Experiment Reusability Assessment
- All legacy GameRefine results: **NOT reusable** (heuristic reward hacking)
- New code components (reward_models, visualization, paper outline): **reusable**

### Code Changes
1. Merged `majority_vote_select` from `eval_gamerefine.py` into `game_protocol.py`
2. Added reward-based data filtering from `train_agents.py` into `train_sft_agents.py`
3. Marked 4 legacy scripts as DEPRECATED with pointers to replacements
4. Fixed `max_new_tokens` 512→256 in `game_protocol.py` and `train_nash_dpo.py`
5. Standardized config keys (`model.base` + `model.base_model` in both YAMLs)
6. Created `src/__init__.py`
7. Updated README project structure

### Auto-Review Loop (2 rounds)

**Round 1** (5 issues found, 5 fixed):
- R1-1: Config key inconsistency → added `base_model` alias
- R1-2: README missing `reward_models.py`, `visualization.py` → updated
- R1-3: No `src/__init__.py` → created
- R1-4: Deprecated scripts undocumented → added to README
- R1-5: Clarified `run_self_play.py` status

**Round 2** (10 issues found, all fixed):
- Unused shell variables `GRPO_EPISODES_PER_GAME`, `EVAL_SAMPLES` → removed
- Comment block wrong episode counts → corrected to match variables
- Hard-coded paths in `collect_and_visualize.py` → parameterized
- `plot_transfer_matrix` JSON schema mismatch → dual-layout support
- Misleading "resume" message in `run.sh` → clarified
- `requirements.txt` GameRefine reference → cleaned
- `run_all_experiments.sh` header comments → accurate numbers
