#!/usr/bin/env python3
"""GRPO self-play — multiprocessing across GPUs for maximum throughput.

Each GPU runs in its own *process*, so Triton auto-tuning is completely
independent and there is zero lock contention.  Batch size can be pushed
to 128+ since each H800 has 80 GB.

Usage:
    python scripts/run_grpo_self_play.py --batch_size 128
"""

import argparse, json, logging, math, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_environments import create_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


GPU_GAME_MAP = {
    0: ["auction", "negotiation"],
    1: ["ultimatum", "matching_pennies"],
    2: ["public_goods", "prisoners_dilemma"],
    3: ["battle_of_sexes", "stag_hunt", "coordination_game", "chicken"],
}

NASH_EQUILIBRIA = {
    "prisoners_dilemma": {"defect": 1.0},
    "coordination_game": {"A": 0.5, "B": 0.5},
    "battle_of_sexes": {"opera": 0.5, "football": 0.5},
    "stag_hunt": {"stag": 0.5, "hare": 0.5},
    "public_goods": {"contribute": 0.5, "free_ride": 0.5},
    "ultimatum": {"accept": 0.7, "reject": 0.3},
    "auction": {},
    "negotiation": {"agree": 0.6, "propose_medium": 0.2,
                    "propose_low": 0.1, "propose_high": 0.1},
}

# ── helpers ──────────────────────────────────────────────────────────

def fmt_prompt(game_prompt: str, actions: list) -> str:
    return (
        "<|im_start|>system\nYou are a strategic decision-making agent. "
        "Analyze the situation carefully and choose the optimal action. "
        "Consider both immediate and long-term consequences.<|im_end|>\n"
        f"<|im_start|>user\n{game_prompt}\n\n"
        f"Choose exactly ONE action from: {', '.join(actions)}\n"
        "Format: ACTION: <your_choice>\nREASONING: <brief explanation>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

def parse_action(resp: str, valid: list) -> str:
    lo = resp.lower()
    if "action:" in lo:
        after = lo.split("action:")[-1].strip()
        for a in valid:
            if a.lower() in after[:50]:
                return a
    for a in valid:
        if a.lower() in lo:
            return a
    return random.choice(valid)

def player_ids(name: str, cfg: dict) -> list:
    gt = cfg.get("type", "")
    if "n_player" in gt:
        return [f"player_{i}" for i in range(cfg.get("num_players", 4))]
    if "sequential" in gt:
        return cfg.get("roles", ["party_A", "party_B"])
    return ["player_0", "player_1"]

def strategy_diversity(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-10)
                for c in counts.values() if c > 0)

def nash_distance(counts: dict, game: str) -> float:
    nash = NASH_EQUILIBRIA.get(game, {})
    if not nash:
        return 0.0
    total = sum(counts.values())
    if total == 0:
        return 0.0
    emp = {a: c / total for a, c in counts.items()}
    return sum((emp.get(a, 0) - nash.get(a, 0)) ** 2
               for a in set(list(nash) + list(emp))) ** 0.5

def load_agent(name, sft_dir, model_name, device):
    path = Path(sft_dir) / name / "final"
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device},
    )
    base.config.use_cache = True
    if len(tok) > base.config.vocab_size:
        base.resize_token_embeddings(len(tok))
    if path.exists() and (path / "adapter_config.json").exists():
        mdl = PeftModel.from_pretrained(base, str(path), is_trainable=True)
    else:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none")
        mdl = get_peft_model(base, cfg)
    return mdl, tok

# ── batched episode generation ───────────────────────────────────────

def play_batched(model, tok, game, config, n_eps, agent_name,
                 temp=0.7, bs=128):
    dev = next(model.parameters()).device
    sc = config["scenarios"][game]
    pids = player_ids(game, sc)
    all_trajs, all_states = [], []

    for bstart in range(0, n_eps, bs):
        bsz = min(bs, n_eps - bstart)
        envs = [create_environment(game, config) for _ in range(bsz)]
        states = [e.reset() for e in envs]
        trajs = [{p: [] for p in pids} for _ in range(bsz)]
        active = list(range(bsz))

        for _rnd in range(15):
            if not active:
                break
            racts = [{} for _ in range(bsz)]
            for pid in pids:
                if not active:
                    break
                prompts, aspaces = [], []
                for i in active:
                    aspaces.append(envs[i].get_action_space(pid))
                    prompts.append(fmt_prompt(
                        envs[i].get_prompt(states[i], pid), aspaces[-1]))

                tok.padding_side = "left"
                enc = tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(dev)
                try:
                    with torch.no_grad():
                        out = model.generate(
                            **enc, max_new_tokens=80, do_sample=True,
                            temperature=temp, top_p=0.9,
                            pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    for j, idx in enumerate(active):
                        a = random.choice(aspaces[j])
                        racts[idx][pid] = a
                        trajs[idx][pid].append(dict(
                            prompt=prompts[j], response=f"ACTION: {a}",
                            action=a, log_prob=-1.0,
                            round=states[idx].round_num,
                            agent_name=agent_name))
                    continue

                gs = enc["input_ids"].shape[1]
                for j, idx in enumerate(active):
                    r = tok.decode(out[j][gs:], skip_special_tokens=True)
                    a = parse_action(r, aspaces[j])
                    racts[idx][pid] = a
                    trajs[idx][pid].append(dict(
                        prompt=prompts[j], response=r, action=a,
                        log_prob=-1.0, round=states[idx].round_num,
                        agent_name=agent_name))

            new_act = []
            for i in active:
                states[i], rew = envs[i].step(racts[i])
                for pid in pids:
                    if trajs[i][pid]:
                        trajs[i][pid][-1]["reward"] = rew.get(pid, 0.0)
                if not states[i].done:
                    new_act.append(i)
            active = new_act

        all_trajs.extend(trajs)
        all_states.extend(states)
        torch.cuda.empty_cache()

    return all_trajs, all_states


def batch_log_probs(model, tok, trajs, pids, lp_bs=48):
    dev = next(model.parameters()).device
    model.eval()
    items = [(ti, p, si, s)
             for ti, td in enumerate(trajs)
             for p in pids for si, s in enumerate(td[p])]
    if not items:
        return
    tok.padding_side = "right"
    for s in range(0, len(items), lp_bs):
        batch = items[s:s + lp_bs]
        ftexts = [it[3]["prompt"] + f"ACTION: {it[3]['action']}" for it in batch]
        ptexts = [it[3]["prompt"] for it in batch]
        fenc = tok(ftexts, return_tensors="pt", padding=True,
                   truncation=True, max_length=1024).to(dev)
        penc = tok(ptexts, return_tensors="pt", padding=True,
                   truncation=True, max_length=1024)
        with torch.no_grad():
            logits = model(**fenc).logits
        for j, (ti, pid, si, _) in enumerate(batch):
            pl = int(penc["attention_mask"][j].sum().item())
            sl = int(fenc["attention_mask"][j].sum().item())
            na = sl - pl
            if na <= 0:
                continue
            al = logits[j, pl - 1:sl - 1]
            ai = fenc["input_ids"][j, pl:sl]
            lp = F.log_softmax(al, dim=-1)
            tlp = lp.gather(1, ai.unsqueeze(1)).squeeze(1)
            trajs[ti][pid][si]["log_prob"] = tlp.mean().item()
    torch.cuda.empty_cache()

# ── GRPO update ──────────────────────────────────────────────────────

def grpo_update(model, tok, trajectories, optimizer, cfg, grad_accum=8):
    dev = next(model.parameters()).device
    model.config.use_cache = False
    model.train()
    clip, kl_c, gamma = cfg["clip_range"], cfg["kl_coeff"], cfg["gamma"]
    tot_loss, n_up = 0.0, 0

    all_items = []
    for traj in trajectories:
        if len(traj) < 2:
            continue
        rew = torch.tensor([float(t["reward"]) for t in traj],
                           dtype=torch.float32, device=dev)
        ret = torch.zeros_like(rew)
        G = 0.0
        for i in reversed(range(len(rew))):
            G = rew[i] + gamma * G
            ret[i] = G
        adv = (ret - ret.mean()) / ret.std().clamp(min=1e-8)
        for sd, a in zip(traj, adv):
            all_items.append((sd, a.item()))

    random.shuffle(all_items)
    tok.padding_side = "right"

    for bs in range(0, len(all_items), grad_accum):
        batch = all_items[bs:bs + grad_accum]
        texts = [sd["prompt"] + f"ACTION: {sd['action']}" for sd, _ in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(dev)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(**enc).logits
            sl = logits[..., :-1, :].contiguous()
            labs = enc["input_ids"][..., 1:].contiguous()
            lp = F.log_softmax(sl, dim=-1)
            tlp = lp.gather(-1, labs.unsqueeze(-1)).squeeze(-1)
            mask = (labs != tok.pad_token_id).float()

        batch_loss = torch.tensor(0.0, device=dev)
        for j, (sd, adv_val) in enumerate(batch):
            nlp_j = (tlp[j] * mask[j]).sum() / mask[j].sum().clamp(min=1)
            ratio = torch.exp(nlp_j - sd["log_prob"])
            clip_r = torch.clamp(ratio, 1 - clip, 1 + clip)
            a_t = torch.tensor(adv_val, device=dev)
            batch_loss += (-torch.min(ratio * a_t, clip_r * a_t)
                           + kl_c * (sd["log_prob"] - nlp_j))

        (batch_loss / len(batch)).backward()
        tot_loss += batch_loss.item()
        n_up += len(batch)

        if (bs // grad_accum + 1) % 4 == 0 or bs + grad_accum >= len(all_items):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    model.config.use_cache = True
    return {"loss": tot_loss / max(n_up, 1), "num_updates": n_up}

# ── per-GPU worker process ──────────────────────────────────────────

def gpu_worker(rank, args_ns, config, iter_dir_str, sft_or_ckpt_dir):
    """Runs entirely inside its own process — full GPU isolation."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    agent_name = f"agent_{rank}"

    scenario_names = list(config["scenarios"].keys())
    games = [g for g in GPU_GAME_MAP.get(rank, []) if g in scenario_names]
    if not games:
        return

    log = logging.getLogger(f"GPU{rank}")
    log.info("Loading model on %s for games %s", device, games)
    model, tok = load_agent(agent_name, sft_or_ckpt_dir,
                            config["model"]["base_model"], device)
    model.eval()

    eps_per_game = args_ns.episodes_per_iter // len(scenario_names)
    all_trajs = []
    payoffs = {}

    for gn in games:
        pids = player_ids(gn, config["scenarios"][gn])
        t0 = time.time()
        trajs, states = play_batched(
            model, tok, gn, config, eps_per_game,
            agent_name, args_ns.temperature, args_ns.batch_size)
        gen_s = time.time() - t0

        t0 = time.time()
        batch_log_probs(model, tok, trajs, pids)
        lp_s = time.time() - t0

        for td in trajs:
            for p in pids:
                if td[p]:
                    all_trajs.append(td[p])
        payoffs[gn] = [sum(s.scores.values()) for s in states]
        log.info("%s: %d eps in %.1fs (gen=%.1fs lp=%.1fs)",
                 gn, len(trajs), gen_s + lp_s, gen_s, lp_s)

    iter_dir = Path(iter_dir_str)
    with open(iter_dir / f"{agent_name}_trajs.json", "w") as f:
        json.dump(all_trajs, f)
    with open(iter_dir / f"{agent_name}_payoffs.json", "w") as f:
        json.dump(payoffs, f)

    log.info("GRPO update (%d trajectory sets)...", len(all_trajs))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args_ns.learning_rate)
    stats = grpo_update(model, tok, all_trajs, optimizer, config["grpo"])
    log.info("GRPO done: loss=%.4f updates=%d", stats["loss"], stats["num_updates"])

    ckpt = iter_dir / agent_name
    model.save_pretrained(str(ckpt))
    tok.save_pretrained(str(ckpt))

    log.info("Eval (%d eps)...", args_ns.eval_episodes)
    model.eval()
    eval_res = {}
    for gn in games:
        pids_g = player_ids(gn, config["scenarios"][gn])
        ac = Counter()
        tp = defaultdict(float)
        et, es = play_batched(model, tok, gn, config,
                              args_ns.eval_episodes, agent_name,
                              temp=0.3, bs=args_ns.batch_size)
        for fs in es:
            for p in pids_g:
                tp[p] += fs.scores.get(p, 0.0)
            for h in fs.history:
                for _, act in h.get("actions", {}).items():
                    ac[act] += 1
        ne = max(args_ns.eval_episodes, 1)
        eval_res[gn] = {
            "avg_payoffs": {p: v / ne for p, v in tp.items()},
            "strategy_diversity": strategy_diversity(dict(ac)),
            "nash_distance": nash_distance(dict(ac), gn),
            "action_distribution": dict(ac),
        }
    with open(iter_dir / f"{agent_name}_eval.json", "w") as f:
        json.dump(eval_res, f, indent=2, default=str)

    log.info("Done ✓")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/game_scenarios.yaml")
    parser.add_argument("--sft_dir", default="results/sft_agents")
    parser.add_argument("--output_dir", default="results/grpo_self_play")
    parser.add_argument("--num_iterations", type=int, default=5)
    parser.add_argument("--episodes_per_iter", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    logger.info("=== GRPO Self-Play (multiprocessing, batch=%d) ===", args.batch_size)
    logger.info("GPUs=%d, Iters=%d, Eps/iter=%d", num_gpus, args.num_iterations,
                args.episodes_per_iter)

    mp.set_start_method("spawn", force=True)
    training_log = []
    sft_dir = args.sft_dir

    for iteration in range(args.num_iterations):
        logger.info("=== Iteration %d/%d ===", iteration + 1, args.num_iterations)
        iter_dir = out / f"iter_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=gpu_worker,
                           args=(rank, args, config, str(iter_dir), sft_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            logger.error("GPUs %s failed!", failed)

        iter_log = {"iteration": iteration, "payoffs": {}, "eval": {}}
        for rank in range(num_gpus):
            an = f"agent_{rank}"
            pf = iter_dir / f"{an}_payoffs.json"
            ef = iter_dir / f"{an}_eval.json"
            if pf.exists():
                with open(pf) as f:
                    iter_log["payoffs"].update(json.load(f))
            if ef.exists():
                with open(ef) as f:
                    iter_log["eval"].update(json.load(f))

        training_log.append(iter_log)
        with open(iter_dir / "iter_results.json", "w") as f:
            json.dump(iter_log, f, indent=2, default=str)

        sft_dir = str(iter_dir)

        for gn, pays in iter_log["payoffs"].items():
            ev = iter_log["eval"].get(gn, {})
            logger.info("  %s: payoff=%.2f div=%.3f nash=%.3f",
                        gn, sum(pays) / max(len(pays), 1),
                        ev.get("strategy_diversity", 0),
                        ev.get("nash_distance", 0))

    for rank in range(num_gpus):
        an = f"agent_{rank}"
        src = out / f"iter_{args.num_iterations - 1}" / an
        dst = out / "final" / an
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            for fn in src.iterdir():
                (dst / fn.name).write_bytes(fn.read_bytes())

    with open(out / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2, default=str)
    logger.info("=== GRPO Self-Play Complete ===")


if __name__ == "__main__":
    main()
