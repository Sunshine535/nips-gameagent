# Project: nips-gameagent

## Project goal

GameAgent: Game-Theoretic Self-Play for Strategic Reasoning and Multi-Objective Alignment in LLMs — 在多博弈场景下做 GRPO 自博弈（Track A）与四角色 Nash-DPO 跨评（Track B），统一评测 ARC、StrategyQA、BBH、GSM8K、TruthfulQA、MT-Bench。

## Key models

- `Qwen/Qwen3.5-9B` — 主实验模型（+ LoRA）

## Key datasets

- 10 个博弈场景（`configs/game_scenarios.yaml`）
- 评测: ARC-Challenge, StrategyQA, BBH, GSM8K, TruthfulQA, MT-Bench

## Repo map

- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排（Track A/B/C/D）
  - Track A: `generate_sft_data.py` → `train_sft_warmup.py` → `run_grpo_self_play.py` → `run_cross_game_transfer.py`
  - Track B: `generate_expert_data.py` → `train_sft_agents.py` → `train_nash_dpo.py`
  - 评测: `run_grpo_vs_nash_comparison.py`, `eval_benchmarks.py`, `eval_game_performance.py`, `collect_and_visualize.py`
  - `gpu_utils.sh` — GPU 分配工具
- `src/` — 核心模块
  - `game_environments.py`, `game_environments_simple.py` — 博弈环境
  - `game_protocol.py` — 博弈协议
  - `nash_dpo.py` — Nash-DPO 实现
  - `reward_models.py` — 奖励模型
  - `visualization.py` — 可视化
- `configs/` — 配置
  - `game_scenarios.yaml` — 博弈场景
  - `agent_roles.yaml` — 角色配置
- `results/` — 实验输出
- `paper/outline.md` — 论文大纲

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# 一键全流程（4–8×A100 80GB）
bash run.sh

# 后台运行
nohup bash run.sh > run.log 2>&1 &

# 强制重跑
FORCE_RERUN=1 bash run.sh

# 打包结果
bash collect_results.sh
```

## Experiment tracks

| Track | 内容 |
|-------|------|
| A | SFT warmup → GRPO 自博弈 → cross-game transfer |
| B | Expert data → 角色 SFT → Nash-DPO |
| C | GRPO vs Nash 对比 + 统一 benchmark |
| D | 消融实验（无 SFT、更少迭代等） |

## Data and outputs

- SFT 数据: `data/`
- GRPO 自博弈: `results/grpo_self_play/`
- Nash-DPO: `results/nash_dpo/`
- SFT agents: `results/sft_agents/`
- Benchmark 评估: `results/eval_benchmarks/`
- 消融: `results/ablation_*/`
- 日志: `logs/`

## Environment

- Python 3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, datasets, accelerate, trl, peft, sentence-transformers, evaluate
- 可选: flash-attn
- 不使用 wandb
- 磁盘需求: ~200GB（存储模型权重）

## Project-specific rules

- Track A 和 B 可独立运行，Track C 依赖两者结果
- 并行生成数据时使用 `CUDA_VISIBLE_DEVICES=0..3`
- `configs/game_scenarios.yaml` 定义全部 10 个博弈场景
- `configs/agent_roles.yaml` 定义 4 个角色

## Remote server

- SSH: `ssh -p 30022 wujn@root@ssh-362.default@222.223.106.147`（密码: 123456，经 JumpServer 堡垒机）
- 登录后进入: Ubuntu 24.04 容器
- GPU: 待确认
- Activate: `source .venv/bin/activate`
- Code dir: 待确认（登录后 `find / -name nips-gameagent -maxdepth 4` 查找）
- Background: `screen -dmS gameagent bash -c '...'`
- HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`
