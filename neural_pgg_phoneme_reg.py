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
from ppgs_for_pr import process_utterance_with_ppg

model_name = "wav2vec2_lv60"  # Change this to "wav2vec2_lv60" or "wav2vec2_xls_r" as needed

phoneme_labels = ["aa","ae","ah","ao","aw","ay","b","ch","d","dh","eh","er","ey","f","g","hh","ih","iy","jh","k","l","m","n","ng","ow","oy","p","r","s","sh","t","th","uh","uw","v","w","y","z","zh","sil"]

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
    
    elif model_name == "wav2vec2_xls_r":
        processor = AutoProcessor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        model = AutoModelForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
        pipeline = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
    elif model_name == "ppg_based_phoneme":
        processor = None
        model = None
        pipeline = None
        
    # Initialize MPD statistics collector
    stats = MpdStats()
    from mpd_eval_v3 import ErrorRateStats
    per_stats = ErrorRateStats()
    for record in tqdm(test_data):
        audio_file = record["id"]
        canonical_phonemes = record["phn_list_canonical"]
        perceived_phonemes = record["phn_list_perceived"]


        predicted_phonemes = process_utterance_with_ppg(audio_file,  phoneme_labels, gpu=0)

        # # merge continuous sil to a single sil
        # for i in range(len(predicted_phonemes) - 1, 0, -1):
        #     if predicted_phonemes[i] == "sil" and predicted_phonemes[i - 1] == "sil":
        #         predicted_phonemes.pop(i)
        # predicted_phonemes = []
        # for p in predicted_sentences[0].split():
        #     if p == "sil":
        #         if not predicted_phonemes or predicted_phonemes[-1] != "sil":
        #             predicted_phonemes.append("sil")
        #     else:
        #         predicted_phonemes.append(p)  
        # # apply timit2cmu mapping
        # predicted_phonemes = [timit2cmu.get(p, p) for p in predicted_phonemes]
        
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
    with open("neural_pgg_phoneme_reg_mpd_results.txt", "w") as f:
        stats.write_stats(f)
        
    with open("neural_pgg_phoneme_reg_per_results.txt", "w") as f:
        per_stats.write_stats(f)
