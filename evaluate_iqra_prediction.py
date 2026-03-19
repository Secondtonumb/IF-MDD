import sys
import os
from pathlib import Path
import csv
import torch
from tqdm import tqdm

# Add .speechbrain to Python path if needed
speechbrain_path = Path.home() / ".speechbrain"
if speechbrain_path.exists():
    sys.path.insert(0, str(speechbrain_path))

# Helper to import MyEncoderASR
try:
    from trainer.MyEncoderASR import MyEncoderASR
except ImportError:
    print("Could not import MyEncoderASR from trainer.MyEncoderASR. Adding current directory to sys.path.")
    sys.path.append(str(Path.cwd()))
    try:
        from trainer.MyEncoderASR import MyEncoderASR
    except ImportError as e:
        print(f"Failed to import MyEncoderASR: {e}")
        sys.exit(1)

def main():
    # 1. Model Configuration
    # Using the path from inference_ver3.py
    # source_model_path = "/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/iqra_extra_acou_model/ottc_k7_RoPE_TTS_FT/"
    # wavlm better PER
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT/save/CKPT+221_PER_2.4347_F1_0.8968.ckpt"
    # source_model_path = "/home/kevingenghaopeng/MDD/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_is26_extra_split/save/CKPT+best_mpdf1_100_0.5619.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_ottc_confEnc_kernal_3/save/CKPT+279_PER_2.7927_F1_0.7582.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_tts/wavlm_large_None_PhnMonoSSL_crottc_confEnc_RoPE_k3/save/CKPT+085_PER_6.1534_F1_0.9104.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl/save/CKPT+014_PER_3.5983_F1_0.8591.ckpt"
    # MPL
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_1_1/save/CKPT+031_PER_3.9921_F1_0.8581.ckpt"
    # MPL 1:1
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_1_1_new/save/CKPT+031_PER_3.5088_F1_0.8772.ckpt"
    # MPL 1:1 epch 65
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_1_1_new/save/CKPT+065_PER_5.4601_F1_0.8542.ckpt"
    # MPL 1:1 epoch epoch24
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_1_1_new/save/CKPT+024_PER_3.1686_F1_0.8690.ckpt"
    # MPL 1:5 epoch75
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_1_5_new/save/CKPT+075_PER_3.5267_F1_0.8141.ckpt"
    # MPL 1:5 epoch95 QuranMB v2
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_1_5_new/save/CKPT+095_PER_4.4218_F1_0.7837.ckpt"
    # MPL 1:all epoch42
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_mpl/wavlm_large_PhnMonoSSL_TTS_FT_ctc_mpl_all/save/CKPT+042_PER_4.8693_F1_0.6703.ckpt"
    # Arabic Hubert
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_tts/hubert_arabic_None_PhnMonoSSL_CROTTC_arabic/CKPT+054_PER_5.8649_F1_0.9183.ckpt"
    # Arabic Hubert FT
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/hubert_arabic_None_PhnMonoSSL_CROTTC_TTSbasedFT_hubert_arabic/CKPT+249_PER_3.2044_F1_0.9164.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/hubert_arabic_None_PhnMonoSSL_CROTTC_TTSbasedFT_hubert_arabic/save/CKPT+265_PER_2.8643_F1_0.9270.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/hubert_arabic_None_PhnMonoSSL_CROTTC_TTSbasedFT_hubert_arabic/CKPT+283_PER_3.9742_F1_0.9338.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/hubert_arabic_None_PhnMonoSSL_CROTTC_TTSbasedFT_hubert_arabic/save/CKPT+275_PER_4.1532_F1_0.9474.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/hubert_arabic_None_PhnMonoSSL_CROTTC_TTSbasedFT_hubert_arabic/CKPT+299_PER_3.9742_F1_0.9509.ckpt"
    # IF-CROTTC
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_IF_CROTTC_IF/save/CKPT+001_PER_2.9180_F1_0.9104.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_IF_CROTTC_IF/CKPT+029_PER_3.3656_F1_0.9071.ckpt"
    
    # Kernal 3 FT
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3/save/CKPT+180_PER_3.6162_F1_0.9137.ckpt"
    source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3/save/CKPT+200_PER_4.2786_F1_0.9000.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3/save/CKPT+201_PER_4.2248_F1_0.8811.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3/save/CKPT+194_PER_3.6520_F1_0.8881.ckpt"
    # Kernal 3 FT beta prior
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3_beta_prior_hard/save/CKPT+259_PER_3.3119_F1_0.9343.ckpt"
    # source_model_path ="/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3_beta_prior_hard/save/CKPT+244_PER_3.0433_F1_0.9176.ckpt"    
    #Kernal 3 with blank insertion 
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k3_insert_blank/save/CKPT+201_PER_4.1331_F1_0.9248.ckpt"
    

    # Kernel 1 FT
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k1/save/CKPT+069_PER_4.3502_F1_0.8197.ckpt"
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra/wavlm_large_None_PhnMonoSSL_CROTTC_TTSbasedFT_k1/save/CKPT+228_PER_3.1686_F1_0.9130.ckpt"
    
    # CTC Best
    # source_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CTC_best_k3"

    # CRCTC
    # source_model_path = "/home/m64000/work/IF-MDD/pretrained_models/iqra_extra_acou_model/CRCTC"
    # Labeled 
    # source_model_path = "/home/m64000/work/IF-MDD/exp_iqra_labeled/wavlm_large_None_PhnMonoSSL_CROTTC_labeled/save/CKPT+014_PER_6.4857_F1_0.9046.ckpt"
    hparams_filename = "inference.yaml"
    
    print(f"Loading model from {source_model_path} with {hparams_filename}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    # 

    try:
        # Initialize model
        asr_model = MyEncoderASR.from_hparams(
            source=source_model_path,
            hparams_file=hparams_filename,
            run_opts={"device": device}
        )
        # asr_model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Data Configuration
    # wav_dir = Path("/home/kevingenghaopeng/MDD/IF-MDD/data_iqra_QuranMB.v2/home/kevingenghaopeng/MDD/IF-MDD/data_iqra_QuranMB.v2/wav")
    wav_dir = Path("/home/m64000/work/dataset/data_iqra/test/wav")
    # output_csv_path = "IF_CROTTC_epoch29_predictions.csv"
    # output_csv_path = "CRCTC_predictions.csv"
    # output_csv_path = "phnmonossl_crctc_k1_epoch228_iqra_test_predictions.csv"
    # output_csv_path = "phnmonossl_ctc_k3_iqra_test_predictions.csv"
    # output_csv_path = "phnmonossl_crottc_k3_beta_prior_hard_iqra_test_predictions_best_per.csv"
    # output_csv_path = "phnmonossl_crottc_k3_insert_blank_iqra_test_predictions_epoch201.csv"
    output_csv_path = "hubert_arabic_crottc_ttsbasedft_hubert_arabic_iqra_test_predictions_epoch275.csv"
    
    
    if not wav_dir.exists():
        print(f"Error: Wave directory not found at {wav_dir}")
        return

    # Get all .wav files (recursive search if needed, or just flat)
    # Assuming flat structure based on path
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    if not wav_files:
        print(f"No .wav files found in {wav_dir}")
        return
        
    print(f"Found {len(wav_files)} WAV files in {wav_dir}")

    results = []

    # 3. Batch Processing / Loop
    print("Starting transcription...")
    
    # We use transcribe_file one by one for simplicity as in inference_ver3.py
    # For speedups, we could do batching, but that requires a DataLoader setup.
    
    for wav_path in tqdm(wav_files, desc="Transcribing"):
        # try:
        file_id = wav_path.stem
            
            # Run transcription
            # transcribe_file handles loading audio and running inference
        try:
            waveform, sample_rate = asr_model.load_audio(f"{wav_path}")
        except:
            waveform = asr_model.load_audio(f"{wav_path}")
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])

        ctc_p = asr_model.encode_batch(batch, rel_length)
        import pdb; pdb.set_trace()
        # filtering ctc_p if argmax's logp < threshold, else set to blank
        threshold = 0.8  # Example threshold, adjust as needed
        max_logp, max_indices = torch.max(torch.exp(ctc_p), dim=-1)
        max_indices[max_logp < threshold] = asr_model.tokenizer.lab2ind["<blank>"]  # Set to blank index if below threshold
        # make ctc_p's < threshold position into blank's one-hot like vector, and keep the rest unchanged
        ctc_p[max_logp < threshold][0,] = -float("inf") + 
        blank_index = asr_model.tokenizer.lab2ind["<blank>"]
        from speechbrain.decoders.ctc import ctc_greedy_decode
        decoded_indices = ctc_greedy_decode(ctc_p, [1.0])
        prediction = asr_model.decode_batch(decoded_indices)
        
        results.append({
            "ID": file_id,
            "Labels": prediction
        })
            
        # except Exception as e:
        #     print(f"Error processing {wav_path.name}: {str(e)}")
        #     results.append({
        #         "ID": wav_path.stem,
        #         "Labels": "ERROR"
        #     })

    # 4. Save to CSV
    print(f"\nSaving results to {output_csv_path}")
    
    fieldnames = ["ID", "Labels"]
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Done! Processed {len(wav_files)} files.")

if __name__ == "__main__":
    main()
