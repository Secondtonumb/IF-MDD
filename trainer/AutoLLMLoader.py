"""Llama model and tokenizer loading helpers."""

from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@contextmanager
def _disable_unused_bnb_dispatchers(model):
    """Keep PEFT from importing bitsandbytes for a non-quantized model."""

    if getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    ):
        yield
        return

    from peft.tuners.lora import model as peft_lora_model

    original_bnb = peft_lora_model.is_bnb_available
    original_bnb_4bit = peft_lora_model.is_bnb_4bit_available
    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False
    try:
        yield
    finally:
        peft_lora_model.is_bnb_available = original_bnb
        peft_lora_model.is_bnb_4bit_available = original_bnb_4bit
        

def AutoLLMLoader(
    model_name,
    use_lora=False,
    lora_config=None,
    replace_output_head=False,
    phoneme_dim=None,
    load_pretrained_weights=True,
):
    """Load a causal LM from Hub weights or a bundle-local configuration."""

    model_path = Path(str(model_name)).expanduser()
    local_only = model_path.exists()
    try:
        if load_pretrained_weights:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_name),
                device_map="auto",
                torch_dtype=torch.float16,
                local_files_only=local_only,
            )
        else:
            config = AutoConfig.from_pretrained(
                str(model_name),
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.float16,
            )

        if replace_output_head and phoneme_dim is not None:
            model.lm_head = torch.nn.Linear(
                model.config.hidden_size,
                phoneme_dim,
                bias=False,
                dtype=torch.float16,
            )
        
        if use_lora and lora_config is not None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config["r"],
                lora_alpha=lora_config["alpha"],
                lora_dropout=lora_config["dropout"],
                target_modules=lora_config["target_modules"],
            )
            with _disable_unused_bnb_dispatchers(model):
                model = get_peft_model(model, peft_config)
            for name, parameter in model.named_parameters():
                if "lora" not in name:
                    parameter.requires_grad = False
    except Exception as exc:
        raise ValueError(f"Error loading model {model_name}: {exc}") from exc
    return model


def AutoLLMTokenizer(model_name):
    """Load the matching tokenizer, enforcing local-only mode for local paths."""

    local_only = Path(str(model_name)).expanduser().exists()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_name),
            local_files_only=local_only,
        )
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        raise ValueError(f"Error loading tokenizer {model_name}: {exc}") from exc
    return tokenizer
