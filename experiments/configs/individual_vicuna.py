import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_vicuna'

    config.tokenizer_paths=["/DIR/Vicuna-7b-v1.3"]
    config.model_paths=["/DIR/Vicuna-7b-v1.3"]
    config.conversation_templates=['vicuna']

    return config