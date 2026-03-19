'''
'''

from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
import torch

def AutoLLMLoader(model_name, use_lora=False, lora_config=None, replace_output_head=False, phoneme_dim=None):    
    """
    Load LLaMA model with optional PEFT/LoRA and custom output head replacement.
    
    Args:
        model_name: HuggingFace model ID
        use_lora: Whether to apply LoRA
        lora_config: LoRA configuration dict
        replace_output_head: Whether to replace LLM output head with custom phoneme head
        phoneme_dim: If replace_output_head=True, dimension of custom output head (e.g., 44 for phonemes)
    
    Returns:
        model: Loaded LLaMA model (with optional LoRA and custom head)
    """
    try:
        # 配置4-bit量化参数
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        
        # 加载模型
        if "Qwen2-Audio" in model_name:
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_name,
                #quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
                # quantization_config=quantization_config,
        
        # ===== Replace LLM output head with custom phoneme head =====
        if replace_output_head and phoneme_dim is not None:
            print(f"[INFO] Replacing LLM output head (vocab_size={model.config.vocab_size}) with custom phoneme head (dim={phoneme_dim})")
            
            hidden_size = model.config.hidden_size
            
            # Replace lm_head with custom projection
            # LLaMA uses: lm_head = Linear(hidden_size, vocab_size)
            # We replace with: lm_head = Linear(hidden_size, phoneme_dim)
            
            model.lm_head = torch.nn.Linear(hidden_size, phoneme_dim, bias=False)
            
            # Convert to same dtype as model weights (float16 after quantization)
            model.lm_head = model.lm_head.to(torch.float16)
            
            print(f"[INFO] Output head replaced successfully")
            print(f"[INFO] New lm_head shape: in_features={model.lm_head.in_features}, out_features={model.lm_head.out_features}")
            print(f"[INFO] lm_head dtype: {next(model.lm_head.parameters()).dtype}")
        
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
