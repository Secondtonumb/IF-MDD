from huggingface_hub import hf_hub_download
import importlib.util
from trainer.MyEncoderASR import MyEncoderASR, MyEncoderDecoderASR
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR
from trainer.MyEncoderASR import MyCTCPrefixBeamSearcher
from trainer.MyEncoderASR import plot_alignments
import torch
import numpy as np

# # Customized Encoder ASR 
path = hf_hub_download(repo_id="Haopeng/CTC_for_IF-MDD", filename="MyEncoderASR.py")

# Dyanamic import
spec = importlib.util.spec_from_file_location("MyEncoderASR", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Simple Transcribe
# asr_model = MyEncoderASR.from_hparams(source="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_for_IF-MDD", hparams_file="inference.yaml")
asr_model = MyEncoderDecoderASR.from_hparams(source="/mount/minesharedisk/kevingenghaopeng/work/IF-MDD/pretrained_models/iqra_IFMDD_Con", hparams_file="inference.yaml")

y = asr_model.transcribe_file("/mount/minesharedisk/sharedfiles/dataset/data_iqra/test/wav/00017_00094.wav")
print(y)
import pdb; pdb.set_trace()
