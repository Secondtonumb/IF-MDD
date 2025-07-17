# Load model directly
from transformers import AutoProcessor, AutoModelForCTC
import pdb
import soundfile as sf
import torch
from transformers import pipeline
import speechbrain as sb
from mpd_eval_v3 import MpdStats
from hyperpyyaml import load_hyperpyyaml
import sys
import os
import librosa
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
# apply models on device
if device == "cuda":
    torch.cuda.set_device(0)
    print("Using GPU: ", torch.cuda.get_device_name(0))
else:
    print("Using CPU")
models= {
    "wav2vec2_lv60":"excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k",
    "wav2vec2_lv60_simple":"excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k-simplified",
    "wav2vec2_xls_r":"vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
}
model_name = "wav2vec2_lv60"  # Change this to "wav2vec2_lv60" or "wav2vec2_xls_r" as needed

timit2cmu = {
    # Vowels & diphthongs
    "aa": "aa", "ae": "ae", "ah": "ah",
    "ao": "aa", "aw": "aw", "ay": "ay",
    "eh": "eh", "er": "er", "ey": "ey",
    "ih": "ih", "iy": "iy", "ow": "ow",
    "oy": "oy", "uh": "uh", "uw": "uw",

    # Consonants
    "b": "b", "bcl": "b",
    "ch": "ch",
    "d": "d", "dcl": "d",
    "dh": "dh", "dx": "dx",
    "f": "f",
    "g": "g", "gcl": "g",
    "hh": "hh", "hv": "hh",
    "jh": "jh",
    "k": "k", "kcl": "k",
    "l": "l", "el": "l",
    "m": "m", "em": "m",
    "n": "n", "en": "n", "nx": "n",
    "ng": "ng",
    "p": "p", "pcl": "p",
    "r": "r",
    "s": "s",
    "sh": "sh",
    "t": "t", "tcl": "t",
    "th": "th",
    "v": "v",
    "w": "w",
    "y": "y",
    "z": "z",
    "zh": "zh",

    # Silences/closures / fillers
    "pau": "sil", "h#": "sil", "epi": "sil",
    "cl": "sil", "q": "sil"
    
}

def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder_save"]
    # 1. Declarations:
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        # # sample rate change to 16000, e,g, using librosa
        # sig = torch.Tensor(librosa.core.load(wav, hparams["sample_rate"])[0])
        # Use wav2vec processor to do normalization
        sig = hparams["wav2vec2"].feature_extractor(
            librosa.core.load(wav, hparams["sample_rate"])[0],
            sampling_rate=hparams["sample_rate"],
        ).input_values[0]
        sig = torch.Tensor(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("perceived_train_target", "canonical_aligned", "perceived_aligned")
    @sb.utils.data_pipeline.provides(
        "phn_list_target",
        "phn_encoded_list_target",
        "phn_encoded_target",
        "phn_list_canonical",
        "phn_encoded_list_canonical",
        "phn_encoded_canonical",
        "phn_list_perceived",
        "phn_encoded_list_perceived",
        "phn_encoded_perceived",
    )
    def text_pipeline_test(target, canonical, perceived):
        phn_list_target = target.strip().split()
        yield phn_list_target
        phn_encoded_list_target = label_encoder.encode_sequence(phn_list_target)
        yield phn_encoded_list_target
        phn_encoded_target = torch.LongTensor(phn_encoded_list_target)
        yield phn_encoded_target
        phn_list_canonical = canonical.strip().split()
        yield phn_list_canonical
        phn_encoded_list_canonical = label_encoder.encode_sequence(phn_list_canonical)
        yield phn_encoded_list_canonical
        phn_encoded_canonical = torch.LongTensor(phn_encoded_list_canonical)
        yield phn_encoded_canonical
        phn_list_perceived = perceived.strip().split()
        yield phn_list_perceived
        phn_encoded_list_perceived = label_encoder.encode_sequence(phn_list_perceived)
        yield phn_encoded_list_perceived
        phn_encoded_perceived = torch.LongTensor(phn_encoded_list_perceived)
        yield phn_encoded_perceived

    sb.dataio.dataset.add_dynamic_item([test_data], text_pipeline_test)

    # 3. Fit encoder:
    # Load the label encoder
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load(lab_enc_file)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        [test_data],
        ["id", "sig", "phn_encoded_target", "phn_encoded_canonical", "phn_encoded_perceived", "phn_list_target", "phn_list_canonical", "phn_list_perceived"],
    )

    return test_data, label_encoder

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    test_data, label_encoder = dataio_prep(hparams)
    
    if model_name == "wav2vec2_lv60":
        processor = AutoProcessor.from_pretrained("excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k")
        model = AutoModelForCTC.from_pretrained("excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k")
    
    elif model_name == "wav2vec2_lv60_simple":
        processor = AutoProcessor.from_pretrained("excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified")
        model = AutoModelForCTC.from_pretrained("excalibur12/wav2vec2-large-lv60_phoneme-timit_english_timit-4k_simplified")

    elif model_name == "wav2vec2_xls_r":
        processor = AutoProcessor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        model = AutoModelForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        pipeline = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        
    # Initialize MPD statistics collector
    stats = MpdStats()
    from mpd_eval_v3 import ErrorRateStats
    per_stats = ErrorRateStats()
    for record in tqdm(test_data):
        audio_file = record["id"]
        canonical_phonemes = record["phn_list_canonical"]
        perceived_phonemes = record["phn_list_perceived"]


        audio_input, sample_rate = sf.read(audio_file)

        inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        # Decode id into string
        predicted_ids = torch.argmax(logits, axis=-1)
        predicted_sentences = processor.batch_decode(predicted_ids)
        # print(predicted_sentences)
        # apply timit2cmu mapping
        
        predicted_phonemes = [timit2cmu.get(p, p) for p in predicted_sentences[0].split(" ")]
        
        # merge continuous sil to a single sil
        for i in range(len(predicted_phonemes) - 1, 0, -1):
            if predicted_phonemes[i] == "sil" and predicted_phonemes[i - 1] == "sil":
                predicted_phonemes.pop(i)
        predicted_phonemes = []
        for p in predicted_sentences[0].split():
            if p == "sil":
                if not predicted_phonemes or predicted_phonemes[-1] != "sil":
                    predicted_phonemes.append("sil")
            else:
                predicted_phonemes.append(p)  
        # apply timit2cmu mapping
        predicted_phonemes = [timit2cmu.get(p, p) for p in predicted_phonemes]
        
        # merge continuous sil to a single sil
        for i in range(len(predicted_phonemes) - 1, 0, -1):
            if predicted_phonemes[i] == "sil" and predicted_phonemes[i - 1] == "sil":
                predicted_phonemes.pop(i) 
        print(predicted_phonemes)
        
        # remove silences for MPD evaluation
        predicted_phonemes = [p for p in predicted_phonemes if p != "sil"]
        canonical_phonemes = [p for p in canonical_phonemes if p != "sil"]
        perceived_phonemes = [p for p in perceived_phonemes if p != "sil"]
        # print(predicted_phonemes)
                
        # Append this record's prediction and ground truth to the stats
        
        # merge continuous sil to a single sil

        
        stats.append(
            ids=[record["id"]],
            predict=[predicted_phonemes],
            canonical=[canonical_phonemes],
            perceived=[perceived_phonemes]
        )
        
        per_stats.append(
            ids=[record["id"]],
            predict=[predicted_phonemes],
            target=[perceived_phonemes]
        )
        # Use mpd_eval_v3 to compute stats
    # After processing all records, summarize and print MPD results
    summary = stats.summarize()
    
    # Write detailed WER & MPD alignment details to a file instead of stdout
    # name base on model name
    with open(f"timit_mpd_results_{model_name}.txt", "w") as f:
        stats.write_stats(f)
        print(f"MPD results written to timit_mpd_results_{model_name}.txt")
        # with open(f"timit_mpd_results_{model_name}.txt", "w") as f:
        #     stats.write_stats(f)

    with open(f"timit_per_results_{model_name}.txt", "w") as f:
        per_stats.write_stats(f)
        print(f"PER results written to timit_per_results_{model_name}.txt")