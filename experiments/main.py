'''A main script to run attack for LLMs.'''
import time
import importlib
import numpy as np
import torch.multiprocessing as mp
from absl import app
from ml_collections import config_flags
import pdb

from llm_attacks import get_goals_and_targets, get_workers

_CONFIG = config_flags.DEFINE_config_file('config')

from configs.template import get_config as default_config
_CONFIG_small = default_config()

import gc

gc.set_threshold(0)

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)

    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)

    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

    workers, test_workers = get_workers(params)

    params_small = _CONFIG_small

    params_small.model_kwargs = [
            {"low_cpu_mem_usage": True}
        ]
    params_small.conversation_templates = ["gpt2"]
    params_small.model_paths = ["/DIR/GPT2"]
    params_small.tokenizer_paths = ["/DIR/GPT2"]
    params_small.result_prefix = "results/gpt2"
    params_small.devices = ['cuda:0']

    small_workers, _ = get_workers(params_small)

    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }


    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            goals = train_goals,
            targets = train_targets,
            workers = workers,
            small_workers=small_workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            train_goals,
            train_targets,
            workers,
            small_workers=small_workers,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}_{timestamp}.json",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    attack.run(
        n_steps=params.n_steps,
        batch_size=params.batch_size, 
        topk=params.topk,
        temp=params.temp,
        target_weight=params.target_weight,
        control_weight=params.control_weight,
        test_steps=getattr(params, 'test_steps', 1),
        anneal=params.anneal,
        incr_control=params.incr_control,
        stop_on_success=params.stop_on_success,
        verbose=params.verbose,
        filter_cand=params.filter_cand,
        allow_non_ascii=params.allow_non_ascii,
        probe_set = params.probe_set,
        filtered_set = params.filtered_set, 
    )

    for worker in workers + test_workers:
        worker.stop()

if __name__ == '__main__':
    app.run(main)