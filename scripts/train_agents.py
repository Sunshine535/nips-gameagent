#!/usr/bin/env python3
"""
Train 4 specialized agents (accuracy/safety/efficiency/creativity) via LoRA
on Qwen/Qwen3.5-9B with different reward functions.
Each agent gets a separate LoRA adapter trained with role-specific rewards.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.game_protocol import (
    AgentRole, REWARD_FUNCTIONS, compute_agent_reward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_agents")


def parse_args():
    parser = argparse.ArgumentParser(description="Train specialized RL agents")
    parser.add_argument("--config", type=str, default="configs/ag