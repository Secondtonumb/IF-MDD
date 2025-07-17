'''
A wrapper for loading the pretrained models from huggingface,
wav2vec2, hubert, wavlm are actually inherit from wav2vec2 class,
whisper is inherit from HFTransformersInterface class
---
NOTES:
For new SSL models, we suggesting using 
encoder_type==speechbrain.lobes.models.huggingface_transformers.wav2vec2.Wav2Vec2
as the encoder_type.
'''

from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from speechbrain.lobes.models.huggingface_transformers.hubert import HuBERT
from speechbrain.lobes.models.huggingface_transformers.wavlm import WavLM
from speechbrain.lobes.models.huggingface_transformers.whisper import Whisper

def AutoSSLLoader(source, freeze, freeze_feature_extractor, save_path, encoder_type=None):
    if source == None:
        return None
    else:
        source = source.lower()
        if "wav2vec2" in source:
            return Wav2Vec2(
                source=source,
                freeze=freeze,
                freeze_feature_extractor=freeze_feature_extractor,
                save_path=save_path
            )
        elif "hubert" in source:
            return HuBERT(
                source=source,
                freeze=freeze,
                freeze_feature_extractor=freeze_feature_extractor,
                save_path=save_path
            )
        elif "wavlm" in source:
            return WavLM(
                source=source,
                freeze=freeze,
                freeze_feature_extractor=freeze_feature_extractor,
                save_path=save_path
            )
        elif "whisper" in source:
            return Whisper(
                source=source,
                freeze=freeze,
                save_path=save_path
            )
        elif encoder_type:
            # use the give encoder 
            try:
                return encoder_type(
                    source=source,
                    freeze=freeze,
                    freeze_feature_extractor=freeze_feature_extractor,
                    save_path=save_path
                )
            except:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
        else:
            raise ValueError(f"Unsupported source: {source}")
