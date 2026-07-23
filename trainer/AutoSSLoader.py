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

from speechbrain.lobes.models.huggingface_transformers.mimi import Mimi
from pathlib import Path

pretrained_models={
    "wav2vec2_base": "facebook/wav2vec2-base", # 768
    "wav2vec2_base_jp": "rinna/japanese-wav2vec2-base", # 768
    "hubert_base": "facebook/hubert-base-ls960", # 768
    "wavlm_base": "microsoft/wavlm-base", # 768
    "wavlm_base_jp": "rinna/japanese-wavlm-base", # 768
    "wavlm_base_plus": "microsoft/wavlm-base-plus", # 768
    "hubert_multilingual": "utter-project/mHuBERT-147", # 768
    "clap" : "laion/clap-htsat-fused", # 768
    "data2vec_base": "facebook/data2vec-audio-base", # 768
    
    "wav2vec2_large": "facebook/wav2vec2-large", # 1024
    "wav2vec_large_xlsr_53": "facebook/wav2vec2-large-xlsr-53", # 1024
    "wav2vec2-xls-r-1b": "facebook/wav2vec2-xls-r-1b", # 1024
    "hubert_large": "facebook/hubert-large-ls960-ft", # 1024
    "hubert_large_ll60k": "facebook/hubert-large-ll60k", # 1024
    "wavlm_large": "microsoft/wavlm-large", # 1024
    "data2vec_large": "facebook/data2vec-audio-large", #1024
    "hubert_arabic": "omarxadel/hubert-large-arabic-egyptian", # 1024
    
    "whisper_medium": "openai/whisper-medium", # 1024
    "whisper_large_v3_turbo": "openai/whisper-large-v3-turbo", # 1280

    "mimi": "kyutai/mimi" # codec, 1
}


class _ConfigOnlyWavLM(WavLM):
    """Construct WavLM from local config before SpeechBrain restores weights."""

    def _from_pretrained(
        self,
        source,
        save_path,
        cache_dir,
        device=None,
        **kwargs,
    ):
        self.model = self.auto_class.from_config(self.config)
        if device is not None:
            self.model.to(device)

def AutoSSLLoader(model_name,
                freeze,
                freeze_feature_extractor,
                save_path,
                output_all_hiddens,
                encoder_type=None,
                encoder_only=False,
                load_pretrained_weights=True):
    """
    source: str, the name of the pretrained model e.g "hubert_multilingual", "clap", "data2vec_base", etc.
    freeze: bool, whether to freeze the model
    freeze_feature_extractor: bool, whether to freeze the feature extractor
    save_path: str, the path to save the model
    encoder_type: str, the type of the encoder
    """    

    if model_name == None:
        print(f"model_name for SSL is None, return None")
        return None
    else:
        local_source = Path(str(model_name)).expanduser()
        model_id = (
            str(local_source.resolve())
            if local_source.exists()
            else pretrained_models.get(model_name, None)
        )
    
    if model_id is None:
        raise ValueError(f"Unsupported model_name: {model_name}")
    
    else:
        model_kind = model_id.lower()
        if "wav2vec2" in model_kind:
            return Wav2Vec2(
                source=model_id,
                freeze=freeze,
                freeze_feature_extractor=freeze_feature_extractor,
                save_path=save_path,
                output_all_hiddens=output_all_hiddens
            )
        elif "hubert" in model_kind:
            return HuBERT(
                source=model_id,
                freeze=freeze,
                freeze_feature_extractor=freeze_feature_extractor,
                save_path=save_path,
                output_all_hiddens=output_all_hiddens
            )
        elif "wavlm" in model_kind:
            wavlm_class = WavLM if load_pretrained_weights else _ConfigOnlyWavLM
            return wavlm_class(
                source=model_id,
                freeze=freeze,
                freeze_feature_extractor=freeze_feature_extractor,
                save_path=save_path,
                output_all_hiddens=output_all_hiddens
            )
        # TODO
        elif "whisper" in model_kind:
            return Whisper(
                source=model_id,
                freeze=freeze,
                save_path=save_path,
                encoder_only=encoder_only,
            )
        elif "mimi" in model_kind:
            return Mimi(
                source=model_id,
                freeze=freeze,
                save_path=save_path,
            )

        elif encoder_type:
            # use the give encoder 
            try:
                return encoder_type(
                    source=model_id,
                    freeze=freeze,
                    freeze_feature_extractor=freeze_feature_extractor,
                    save_path=save_path,
                    output_all_hiddens=output_all_hiddens
                )
            except:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
