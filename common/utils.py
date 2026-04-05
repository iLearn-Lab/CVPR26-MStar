import json
import re
import os
import random
import numpy as np
import torch
import multiprocessing
from typing import Tuple
from statistics import mean
from torch.utils.data import Dataset

def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_model_parallel() -> Tuple[int, int]:
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    return local_rank, world_size

def read_json(file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(js_obj, file_path):
    assert str(file_path).endswith(".json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(js_obj, f, indent=4)

def read_txt(file_path):
    assert str(file_path).endswith(".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    return data

def regex_calibrate(output_text: str):
    equation_regex = r"([\d\.\%\/\*\+\-\$\s]+) = ([\d\.\$\s]+)(?=[A-Za-z,.;!?]|\b)"

    def evaluate_expression(expression):
        cleaned_expression = re.sub(r"\s+\.", ".", expression)
        cleaned_expression = re.sub(r"\.\s+", ".", cleaned_expression)
        cleaned_expression = cleaned_expression.replace(" x ", " * ").replace("$", "").replace("%", "/100")
        cleaned_expression = re.sub(r"\s+", "", cleaned_expression)
        try:
            return eval(cleaned_expression, {}, {})
        except Exception:
            return None

    def handle_units(match):
        expression, current_answer = match.groups()
        unit = re.findall(r"[\$\$\$]", current_answer)
        unit = unit[0] if unit else ""
        correct_answer = evaluate_expression(expression)
        if correct_answer is None:
            return match.group(0)
        if "." in current_answer or correct_answer % 1 != 0:
            correct_answer = f"{correct_answer:.6f}"
        else:
            correct_answer = int(correct_answer)
        return f" {expression.strip()} = {unit}{correct_answer} " if correct_answer is not None else match.group(0)

    calibrated_text = re.sub(equation_regex, handle_units, output_text)
    calibrated_text = calibrated_text.strip()

    calibrated_text = re.sub(r"(\d)([A-Za-z,.;!?])", r"\1 \2", calibrated_text)
    calibrated_text = re.sub(r"(\d)\s+(\.\d+)", r"\1\2", calibrated_text)

    return calibrated_text

def shuffleDict(d):
    keys = list(d.keys())
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    [(key, d[key]) for key in keys]
    random.shuffle(keys)
    keys = [(key, d[key]) for key in keys]

    return dict(keys)
