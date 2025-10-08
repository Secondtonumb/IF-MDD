from huggingface_hub import hf_hub_download
import importlib.util

# Customized Encoder ASR 
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")

# Dyanamic import
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Transcribe

asr_model = module.MyEncoderASR.from_hparams(source="Haopeng/CTC_for_IF-MDD", hparams_file="inference.yaml")
x = asr_model.transcribe_file("./examples/arctic_b0503.wav")
print(x)