'''
'''

from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
import torch

def AutoLLMLoader(model_name, use_lora=False, lora_config=None):    
    try:
        # 配置4-bit量化参数
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # 加载模型

        if "Qwen" in model_name:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
        if use_lora and lora_config is not None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_config["r"],
                lora_alpha=lora_config["alpha"],
                lora_dropout=lora_config["dropout"],
                target_modules=lora_config["target_modules"],
            )
            model = get_peft_model(model, peft_config)
            # Print trainable params
            model.print_trainable_parameters()
            
            # 只训练 LoRA 参数
            for name, param in model.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
                    
    except Exception as e:
        raise ValueError(f"Error loading model {model_name}: {e}")
    return model

def AutoLLMTokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # 确保有pad_token
    except Exception as e:
        raise ValueError(f"Error loading tokenizer {model_name}: {e}")
    return tokenizer
