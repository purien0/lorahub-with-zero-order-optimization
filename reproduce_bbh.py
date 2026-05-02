from lorahub.algorithm import lorahub_inference
import os
import json
from lorahub.algorithm import lorahub_learning, lorahub_inference,lorahub_zolearning,init_global_model_and_lora,get_lora_cache
from lorahub.constant import LORA_MODULE_NAMES
import random
from random import shuffle
import argparse
import copy
import numpy as np
import torch

def evaluate_flan_results_zero_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    flan_model, flan_tokenizer , _ = init_global_model_and_lora(model_name = flan_model_name)
    
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (zero shot): ", sub_dir)
        _,task_acc = lorahub_inference(task_inputs,
                          flan_model,
                          flan_tokenizer,
                          16,
                          task_outputs)
        print("average perf:", task_acc)


def evaluate_flan_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    flan_model, flan_tokenizer , _ = init_global_model_and_lora(model_name = flan_model_name)
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (five shot): ", sub_dir)
        _,task_acc = lorahub_inference(task_inputs,
                          flan_model,
                          flan_tokenizer,
                          16,
                          task_outputs)
        print("average perf:", task_acc)


def evaluate_lorahub_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    base_model, tokenizer , _ = init_global_model_and_lora(model_name = flan_model_name)
    # 5 seeds used in our experiments
    for sub_dir in sub_dirs:
        # construct the few-shot examples for lorahub learning
        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])
            
        # random select 5 examples for each task
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        # load the zero-shot examples for evaluation
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])

        task_perf_list = []
        for seed in range(1, 6):
            random.seed(seed)

            def get_lora_module_list():
                return random.sample(LORA_MODULE_NAMES, 20)
            # get a list of modules to be used in the composition
            lora_module_list = get_lora_module_list()
            model, lora_cache = get_lora_cache(lora_module_list,copy.deepcopy(base_model))

            # perform LoRAHub learning
            module_weights, model, tokenizer = lorahub_learning(lora_module_list=lora_module_list,
                                                                example_inputs=example_inputs,
                                                                example_outputs=examples_outputs,
                                                                max_inference_step=40,
                                                                batch_size=16,model = model, tokenizer = tokenizer, cache = lora_cache)

            print("module_weights:", module_weights)

            """
            Perform inference to get predictions
            """
            _, task_acc = lorahub_inference(example_inputs=task_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=10,
                                            # can set as None if you do not have the ground truth
                                            example_outputs=task_outputs)
            task_perf_list.append(task_acc)
        avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
        print("average perf:", avg_perf, "best perf:", max_perf)

def evaluate_lorahub_zo_results_few_shot(folder, flan_model_name,args):
    sub_dirs = os.listdir(folder)
    base_model, tokenizer , _ = init_global_model_and_lora(model_name = flan_model_name)
    # 5 seeds used in our experiments
    i = 0
    for sub_dir in sub_dirs:
        i+=1
        if i<=24:
            continue
        # if i>24:
        #     continue
        # construct the few-shot examples for lorahub learning
        example_inputs, examples_outputs = [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])
            
        # random select 5 examples for each task
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # take the first 5 examples
        example_inputs, examples_outputs = example_inputs[:5], examples_outputs[:5]

        # load the zero-shot examples for evaluation
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])

        task_perf_list = []
        for seed in range(1, 6):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            def get_lora_module_list():
                return random.sample(LORA_MODULE_NAMES, 20)
            # get a list of modules to be used in the composition
            lora_module_list = get_lora_module_list()
            model, lora_cache = get_lora_cache(lora_module_list,copy.deepcopy(base_model))

            # perform LoRAHub learning
            module_weights, model, tokenizer = lorahub_zolearning(lora_module_list=lora_module_list,
                                                                example_inputs=example_inputs,
                                                                example_outputs=examples_outputs,
                                                                max_inference_step=40,seed = seed,
                                                                batch_size=16,args = args,model = model, tokenizer = tokenizer, cache = lora_cache)

            print("module_weights:", module_weights)

            """
            Perform inference to get predictions
            """
            _, task_acc = lorahub_inference(example_inputs=task_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=16,
                                            # can set as None if you do not have the ground truth
                                            example_outputs=task_outputs)
            task_perf_list.append(task_acc)
        avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
        print("average perf:", avg_perf, "best perf:", max_perf)

if __name__ == "__main__":
    if not os.path.exists("data_bbh"):
        # download dataset
        # os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        import zipfile

        with zipfile.ZipFile("data_bbh.zip", 'r') as zip_ref:
            zip_ref.extractall("data_bbh")
        # unzip
        # os.system("unzip data_bbh.zip")
    # evaluate the model
    # evaluate_flan_results_zero_shot("data_bbh", "google/flan-t5-large")
    # five shot for flan models
    # evaluate_flan_results_few_shot("data_bbh", "google/flan-t5-large")
    # five shot for lorahub models
    # evaluate_lorahub_results_few_shot("data_bbh", "google/flan-t5-large")
    # five shot for zo lorahub models

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "momentum","adam"])
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--init_scale", type=float, default=0.1)
    parser.add_argument("--clip_value", type=float, default=1.5)
    args = parser.parse_args()
    print("="*30)
    print("Experiment Config:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("="*30)

    evaluate_lorahub_zo_results_few_shot("data_bbh", "google/flan-t5-large",args)
