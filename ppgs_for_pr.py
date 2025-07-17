import pdb
import ppgs
import json
import torchaudio
import torch
# apply default torch device as gpu

in_json = "/home/kevingenghaopeng/MDD/mpl-mdd/data/test_erj_spk_open_test_1.1.json"

phoneme_labels = ["aa","ae","ah","ao","aw","ay","b","ch","d","dh","eh","er","ey","f","g","hh","ih","iy","jh","k","l","m","n","ng","ow","oy","p","r","s","sh","t","th","uh","uw","v","w","y","z","zh","sil"]

# read json and get all valid things as a class.
#   "/common/db/ERJ_annot_ver1.1/NAR_F05/wav/NAR_F05_S3_041.wav": {
#     "wav": "/common/db/ERJ_annot_ver1.1/NAR_F05/wav/NAR_F05_S3_041.wav",
#     "duration": 4.7979375,
#     "spk_id": "NAR_F05",
#     "canonical_aligned": "sil dh ah g ah v ah m ah n t s ao t ao th ah ah z ey sh ah n ah v hh ih z s ih t ah z ah n sh ih p sil",
#     "perceived_aligned": "sil d ah g ah v er m ey n t s aa t ao th l iy z ey sh ah n ao v hh ih z s ih r ih z ih n s ih p sil",
#     "perceived_train_target": "sil d ah g ah v er m ey n t s aa t ao th l iy z ey sh ah n ao v hh ih z s ih r ih z ih n s ih p sil",
#     "wrd": "THE GOVERNMENT SOUGHT AUTHORIZATION OF HIS CITIZENSHIP\n"
#   },

class Utterance:
    def __init__(self, wav, duration, spk_id, canonical_aligned, perceived_aligned, perceived_train_target, wrd):
        self.wav = wav
        self.duration = duration
        self.spk_id = spk_id
        self.canonical_aligned = canonical_aligned
        self.perceived_aligned = perceived_aligned
        self.perceived_train_target = perceived_train_target
        self.wrd = wrd
        self.ppg_based_phoneme = None
        
    @classmethod
    def from_dict(cls, d):
        return cls(
            wav=d.get("wav"),
            duration=d.get("duration"),
            spk_id=d.get("spk_id"),
            canonical_aligned=d.get("canonical_aligned"),
            perceived_aligned=d.get("perceived_aligned"),
            perceived_train_target=d.get("perceived_train_target"),
            wrd=d.get("wrd"),
        )
    def to_dict(self):
        return {
            "wav": self.wav,
            "duration": self.duration,
            "spk_id": self.spk_id,
            "canonical_aligned": self.canonical_aligned,
            "perceived_aligned": self.perceived_aligned,
            "perceived_train_target": self.perceived_train_target,
            "wrd": self.wrd,
            "ppg_based_phoneme": self.ppg_based_phoneme,
        }
        
def process_utterance_with_ppg(audio_file, phoneme_labels, gpu=0):
    import torch
    import numpy as np

    ppgs_plot = ppgs.from_file(audio_file, gpu=gpu)
    # ppgs_plot is a array of [Batch, cluster, time]
    # We'll take the argmax over cluster axis to get the most likely cluster at each time step
    # and flatten batch if needed (assuming batch size 1)

    # If ppgs_plot is a torch tensor, convert to numpy
    if hasattr(ppgs_plot, "detach"):
        ppgs_np = ppgs_plot.detach().cpu().numpy()
    else:
        ppgs_np = np.array(ppgs_plot)

    # Remove batch dimension if present
    if ppgs_np.shape[0] == 1:
        ppgs_np = ppgs_np[0]  # [cluster, time]

    # For each time step, get the cluster with max probability
    cluster_series = np.argmax(ppgs_np, axis=0)  # [time]
    # Merge consecutive same clusters into one (discrete time series),
    # maybe it's better to 
    discrete_series = [cluster_series[0]]
    for c in cluster_series[1:]:
        if c != discrete_series[-1]:
            discrete_series.append(c)
    # convert discrete_series to phoneme labels
    discrete_series_phoneme = [phoneme_labels[i] for i in discrete_series]
    # Now discrete_series is like [1, 20, 40, 23]
    print(f"Utterance: {audio_file}")
    print(f"Discrete PPG cluster series: {discrete_series}")
    print(f"Discrete PPG phoneme series: {discrete_series_phoneme}")

    # save as string
    return discrete_series_phoneme

    
# # Load the JSON and create a list of Utterance objects
# with open(in_json, "r", encoding="utf-8") as f:
#     data = json.load(f)

# utterances = []
# for key, value in data.items():
#     if isinstance(value, dict):
#         utterances.append(Utterance.from_dict(value))

# # apply ppgs to all utterances
# for utterance in utterances:
#     # ppgs_plot = ppgs.from_audio(audio, sr)
#     process_utterance_with_ppg(utterance, phoneme_labels, gpu=0)


# # save to a new json file
# with open(f"{in_json.split('/')[-1].split('.')[0]}_ppg_based_phoneme.json", "w", encoding="utf-8") as f:
#     import pdb; pdb.set_trace()
#     utterances_dict = [utterance.to_dict() for utterance in utterances]
#     # use *.wav as key, save as a dict
#     import os
#     utterances_dict = {os.path.basename(utterance['wav']): utterance for utterance in utterances_dict}
#     json.dump(utterances_dict, f, ensure_ascii=False, indent=2)

    
    
    