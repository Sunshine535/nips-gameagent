# Experiment Tracker

| Run | Milestone | Description | Status | Start | End | Notes |
|-----|-----------|-------------|--------|-------|-----|-------|
| R001 | M0 | GRPO SFT data (10 games × 500 ep) | 🟡 RUNNING | 2026-03-23 20:08 | — | 6/10 games done, ~1h remaining |
| R002 | M0 | Nash-DPO expert data (8 games × 500 ep) | ⏳ QUEUED | — | — | Starts after R001 completes |
| R003 | M1 | 4 GRPO warmup agents (parallel) | ⏳ QUEUED | — | — | After R001 |
| R004 | M1 | 4 Nash-DPO role agents | ⏳ QUEUED | — | — | After R002 |
| R005 | M2 | GRPO self-play (3 iter) | ⏳ QUEUED | — | — | After R003 |
| R006 | M2 | Nash-DPO self-play (2 iter) | ⏳ QUEUED | — | — | After R004 |
| R007 | M3 | Cross-game transfer | ⏳ QUEUED | — | — | After R005 |
| R008 | M3 | GRPO vs Nash-DPO | ⏳ QUEUED | — | — | After R005 + R006 |
| R009 | M3 | Unified benchmarks | ⏳ QUEUED | — | — | After R005 + R006 |
| R010 | M4 | Ablation: no SFT warmup | ⏳ QUEUED | — | — | After R005 |
| R011 | M4 | Ablation: fewer GRPO iters | ⏳ QUEUED | — | — | After R003 |
| R012 | M4 | Ablation: fewer Nash-DPO iters | ⏳ QUEUED | — | — | After R004 |

## Progress Log

### 2026-03-23 20:08 — Pipeline Start
- Launched `run_all_experiments.sh` on server (PID 55626)
- 4-GPU parallel data generation enabled
- SFT data generation (R001) started: 4 processes, one per GPU

### 2026-03-24 00:10 — R001 Progress Update
- 6/10 games completed: prisoners_dilemma, coordination_game, stag_hunt, chicken, public_goods, auction
- Remaining: battle_of_sexes (346/500), matching_pennies (331/500), ultimatum (175/500), negotiation (319/500)
- ETA: ~1.2 hours for R001 completion

### 2026-03-24 10:42 — R005 Progress Update: Iteration 1/3
- **Completed games**: prisoners_dilemma, coordination_game, battle_of_sexes, stag_hunt, chicken (5/10)
- **In progress**: matching_pennies (27%, ~51min remaining)
- **Remaining**: public_goods, auction, ultimatum, negotiation
- Each game: ~68-70 min, ~20.5s/episode
- Process PID 58862 running for 6h+, stable, no errors
- GPU 0 active (~50%), GPUs 1-3 models loaded but idle (sequential episode generation)

### 2026-03-24 10:42 — Pre-emptive Bug Fix
- Found: run_cross_game_transfer.py line 156 had max_seq_length=2048 in SFTConfig
- Fix: Changed to max_length=2048 (TRL 0.29.1 compatibility)
- Also fixed: train_agents.py line 157 (same issue)
- These would have caused A4 (cross-game transfer) to fail immediately after A3

### 2026-03-24 10:42 — Paper Preparation
- Created paper/main.tex (NeurIPS 2026 format, ~16KB, full structure)
- Created paper/references.bib (14 references)
- Created scripts/generate_paper_content.py (auto-generates Table 2-5, Figure 1-4 from results)
- Created scripts/monitor_pipeline.sh (quick status checker)
- Created scripts/finalize_results.sh (auto-runs after pipeline completion)

### Estimated Timeline
- Iteration 1 complete: ~2026-03-24 16:00 (5.3h remaining)
- Iteration 2 complete: ~2026-03-25 04:00
- Iteration 3 complete: ~2026-03-25 16:00
- Full pipeline (A3 + A4 + B1-B3 + C1-C2 + D1-D3): ~2026-03-26 10:00
