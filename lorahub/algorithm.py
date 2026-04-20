from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig
from functools import partial
from typing import List, Optional, Union
import copy


def load_base_model_and_lora_modules(
    lora_module_list: List[str], model_name_or_path: Optional[str] = None
):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(
            default_peft_model_id
        ).base_model_name_or_path

    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except Exception:
        raise Exception(
            f"{default_peft_model_id} is unable to load into the model {model_name_or_path}"
        )

    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except Exception:
            raise Exception(
                f"LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank)."
            )

    return peft_model, tokenizer, cache


def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(example_dataset, model, batch_size):
    """
    Get the loss of the model on the example dataset. Usually the example dataset only contains a few examples.
    """
    data_batch_size = (
        len(example_dataset)
        if batch_size is None
        else min(len(example_dataset), batch_size)
    )
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset["input"])


def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)

    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)

    return metric_val


def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict


def lorahub_inference(
    example_inputs: List[str],
    model_or_name_path: Union[AutoModelForSeq2SeqLM, str],
    tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
    batch_size: int,
    example_outputs: List[str] = None,
):
    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if (
                output.strip().lower().replace(".", "")
                == truth.strip().lower().replace(".", "")
            ):
                correct += 1
            total += 1
        return correct / total * 100

    example_predictions = []
    # load model
    if isinstance(model_or_name_path, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_or_name_path)
    else:
        model = model_or_name_path

    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path)
    else:
        tokenizer = tokenizer_or_tokenizer_path

    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=256)
        outputs = tokenizer.batch_decode(outputs.to("cpu"), skip_special_tokens=True)
        example_predictions.extend(outputs)

    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf = None

    return example_predictions, task_perf


def lorahub_learning(
    lora_module_list: List[str],
    example_inputs: List[str],
    example_outputs: List[str],
    max_inference_step: int,
    model_name_or_path=None,
    batch_size=None,
    get_loss=default_get_loss,
    get_regular=default_l1_regularization,
    seed=42,
):
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    model, tokenizer, cache = load_base_model_and_lora_modules(
        lora_module_list, model_name_or_path
    )
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    get_score_partial = partial(
        get_score,
        model=model,
        cache=cache,
        example_dataset=dataset,
        batch_size=batch_size,
        get_loss=get_loss,
        get_regular=get_regular,
    )
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(
        parametrization=instrum, budget=max_inference_step
    )
    print("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    print(f"[ori] final loss={get_score_partial(recommendation.value)}")
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return recommendation.value, model, tokenizer


def _resolve_zo_hyperparams(max_inference_step, args=None, **overrides):
    params = {
        "steps": max_inference_step,
        "eps": 0.05,
        "lr": 0.1,
        "q": 10,
        "beta": 0.9,
        "beta1": None,
        "beta2": None,
        "adam_eps": 1e-8,
        "init_scale": 0.1,
        "clip_value": 1.5,
    }

    if args is not None:
        for key in params:
            if hasattr(args, key):
                value = getattr(args, key)
                if value is not None:
                    params[key] = value

    for key, value in overrides.items():
        if value is not None:
            params[key] = value

    return params

import inspect

def filter_kwargs_for_func(func, kwargs):
    sig = inspect.signature(func)
    return {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }

# def zo_optimize_lorahub(get_score, dim, steps=100, eps=0.05, lr=1, q=10):
#     return zo_optimize_momentum(get_score, dim, steps=steps, eps=eps, lr=lr, q=q)


def zo_optimize_momentum(
    get_score,
    dim,
    steps=100,
    eps=0.05,
    lr=0.1,
    q=10,
    beta=0.9,
    beta1 = 0.9,
    beta2 = 0.99,
    init_scale=0.1,
    clip_value=1.5,
):
    import numpy as np

    weights = np.random.uniform(-init_scale, init_scale, size=dim)
    momentum = np.zeros(dim)

    for step in range(1, steps + 1):
        grad = np.zeros(dim)

        for _ in range(q):
            z = np.random.randn(dim)
            loss1 = get_score(weights + eps * z)
            loss2 = get_score(weights - eps * z)
            grad += ((loss1 - loss2) / (2 * eps)) * z

        grad /= q
        momentum = beta * momentum + (1 - beta) * grad
        weights -= lr * momentum
        weights = np.clip(weights, -clip_value, clip_value)

        loss_now = get_score(weights)
        print(f"[ZO-Momentum] step={step}, loss={loss_now:.6f}")

    return weights


def zo_optimize_adam(
    get_score,
    dim,
    steps=100,
    eps=0.05,
    lr=0.1,
    q=10,
    beta1=0.9,
    beta2=0.999,
    adam_eps=1e-8,
    init_scale=0.1,
    clip_value=1.5,
):
    import numpy as np

    weights = np.random.uniform(-init_scale, init_scale, size=dim)
    m = np.zeros(dim)
    v = np.zeros(dim)

    for step in range(1, steps + 1):
        grad = np.zeros(dim)

        for _ in range(q):
            z = np.random.randn(dim)
            loss1 = get_score(weights + eps * z)
            loss2 = get_score(weights - eps * z)
            grad += ((loss1 - loss2) / (2 * eps)) * z

        grad /= q
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)

        weights -= lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        weights = np.clip(weights, -clip_value, clip_value)

        loss_now = get_score(weights)
        print(f"[ZO-Adam] step={step}, loss={loss_now:.6f}")

    return weights


def lorahub_zolearning(
    lora_module_list: List[str],
    example_inputs: List[str],
    example_outputs: List[str],
    max_inference_step: int,
    model_name_or_path=None,
    batch_size=None,
    get_loss=default_get_loss,
    get_regular=default_l1_regularization,
    seed=42,
    args=None,
    method="base",
):
    if args is not None:
        method = args.method

    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    model, tokenizer, cache = load_base_model_and_lora_modules(
        lora_module_list, model_name_or_path
    )
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    get_score_partial = partial(
        get_score,
        model=model,
        cache=cache,
        example_dataset=dataset,
        batch_size=batch_size,
        get_loss=get_loss,
        get_regular=get_regular,
    )
    zo_kwargs = _resolve_zo_hyperparams(
        max_inference_step=max_inference_step, args=args
    )

    print("> Begin to perform gradient-free optimization ...")
    if method == "adam":
        kwargs = filter_kwargs_for_func(zo_optimize_adam, zo_kwargs)
        weights = zo_optimize_adam(
            get_score_partial,
            dim=number_of_loras,
            **kwargs,
        )
    elif method == "momentum":
        kwargs = filter_kwargs_for_func(zo_optimize_momentum, zo_kwargs)
        weights = zo_optimize_momentum(
            get_score_partial,
            dim=number_of_loras,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported ZO method: {method}")

    print(f"[ori] final loss={get_score_partial(weights)}")
    final_lora = get_final_weights(weights, lora_module_list, cache)
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return weights, model, tokenizer
