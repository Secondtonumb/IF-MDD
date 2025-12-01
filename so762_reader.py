# pure_phn_id, pure_phn, pure_id, phone, start_time, duration, word_number, word_text, wrd_acc, wrd_stress, wrd_total, phn_score

import numpy as np
import json
import kaldi_io
import speechbrain as sb
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import speechbrain as sb
import json

# add ../trainer to sys.path
import sys
sys.path.append('../trainer')
from trainer.AutoSSLoader import AutoSSLLoader

class GoPDataset(Dataset):
    def __init__(self, set, am='librispeech'):
        # normalize the input to 0 mean and unit std.
        if am=='librispeech':
            dir='seq_data_librispeech_v3'
            norm_mean, norm_std = 3.203, 4.045
        elif am=='paiia':
            dir='seq_data_paiia'
            norm_mean, norm_std = -0.652, 9.737
        elif am=='paiib':
            dir='seq_data_paiib'
            norm_mean, norm_std = -0.516, 9.247
        else:
            raise ValueError('Acoustic Model Unrecognized.')
        
        # load phone mappings
        # Phone index 352
        # Pure phone 41
        self.purephone_dict = self.load_phone_mappings(mapping_mode="pure_phone")
        self.phone_dict = self.load_phone_mappings(mapping_mode="phone")
        self.index_to_pureindex = self.load_phone_mappings(mapping_mode="index_to_pureindex")
        self.phoneindex_to_purephone = self.load_phone_mappings(mapping_mode="phoneindex_to_purephone")
        
        if set == 'train':
            self.wav_scp = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/train/wav.scp"
            self.feat = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_dur_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_word_id.npy'), dtype=torch.float)
            self.alignment_ctm = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/so762_alignment/kaldi_gop_align/train/phones_int.ctm"
            self.word_ctm = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/exp/ali_train/ctm"
            self.word_csv = "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/tr_labels_word.csv"
            self.phn_ctm_nosil = "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/tr_phones_nosil.ctm"
        elif set == 'test':
            self.wav_scp = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/test/wav.scp"
            self.feat = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_dur_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/tr_word_id.npy'), dtype=torch.float)
            self.alignment_ctm = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/so762_alignment/kaldi_gop_align/test/phones_int.ctm"
            self.word_ctm = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/exp/ali_test/ctm"
            self.word_csv = "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/te_labels_word.csv"
            self.phn_ctm_nosil = "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/raw_kaldi_gop/librispeech/te_phones_nosil.ctm"
        
        # Load word CSV data and broadcast to frame level
        # Format: pure_phn_id, word_number, word, acc_score, stress_score, total_score
        # We'll map each word to its corresponding frames based on alignment_ctm timing
        
        # First, parse alignment_ctm to get frame-level phone/word assignments
        # alignment_ctm format: utt_id speaker start_time duration phone_index
        self.utt_word_scores_by_frame = {}  # key: utt_id, value: [frame_idx] -> word_scores_dict
        self.utt_phone_info = {}  # key: utt_id, value: list of {start_time, duration, phone_idx}
        
        # Parse alignment_ctm and build phone timeline for each utterance
        with open(self.alignment_ctm, 'r') as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0]
                start_time = float(parts[2])
                duration = float(parts[3])
                phone_idx = int(parts[4])
                
                if utt_id not in self.utt_phone_info:
                    self.utt_phone_info[utt_id] = []
                
                self.utt_phone_info[utt_id].append({
                    'start_time': start_time,
                    'duration': duration,
                    'phone_idx': phone_idx
                })
        
        # Parse word_csv and aggregate word scores by word_number
        with open(self.word_csv, 'r') as f:
            word_csv_lines = [line.strip() for line in f if line.strip()]
        
        # Get ordered list of utterance IDs from alignment (preserve order)
        utt_ids_ordered = []
        with open(self.alignment_ctm, 'r') as f:
            for line in f:
                utt_id = line.strip().split()[0]
                if not utt_ids_ordered or utt_ids_ordered[-1] != utt_id:
                    utt_ids_ordered.append(utt_id)
        
        # Parse word_csv by utterance and aggregate word scores
        word_idx = 0
        for utt_id in utt_ids_ordered:
            # Collect all word entries for this utterance
            word_entries_by_num = {}  # word_number -> list of scores
            
            # Read word entries for this utterance
            start_idx = word_idx
            if word_idx < len(word_csv_lines):
                while word_idx < len(word_csv_lines):
                    parts = word_csv_lines[word_idx].split(',')
                    word_number = int(parts[1])
                    word_text = parts[2]
                    acc_score = int(parts[3])
                    stress_score = int(parts[4])
                    total_score = int(parts[5])
                    
                    if word_number not in word_entries_by_num:
                        word_entries_by_num[word_number] = {
                            'word': word_text,
                            'scores': []
                        }
                    
                    word_entries_by_num[word_number]['scores'].append({
                        'accuracy': acc_score,
                        'stress': stress_score,
                        'total': total_score
                    })
                    
                    word_idx += 1
                    
                    # Check if next line starts a new utterance
                    if word_idx < len(word_csv_lines):
                        next_parts = word_csv_lines[word_idx].split(',')
                        next_word_num = int(next_parts[1])
                        if next_word_num == 0 and word_number != 0:
                            break
            
            # Aggregate scores for each word (mean across speakers)
            word_scores_agg = {}  # word_number -> aggregated scores
            for word_num in sorted(word_entries_by_num.keys()):
                entries = word_entries_by_num[word_num]['scores']
                accuracies = [e['accuracy'] for e in entries]
                stresses = [e['stress'] for e in entries]
                totals = [e['total'] for e in entries]
                
                word_scores_agg[word_num] = {
                    'word': word_entries_by_num[word_num]['word'],
                    'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0,
                    'mean_stress': sum(stresses) / len(stresses) if stresses else 0,
                    'mean_total': sum(totals) / len(totals) if totals else 0,
                    'num_speakers': len(entries)
                }
            
            # Now map words to frames based on phone timeline
            # Each phone belongs to a word; broadcast word scores to all frames of that word
            if utt_id in self.utt_phone_info:
                phones = self.utt_phone_info[utt_id]
                # Create frame-level word assignment
                # Frames are typically 10ms or 20ms; phone has start_time and duration
                # We need to assign each phone to a word_number
                # Assumption: phones are already grouped by word in the alignment
                
                # For simplicity, use word_number from phones' position
                # Group consecutive phones by word boundaries
                frame_word_scores = []  # list indexed by frame
                
                # Build a mapping from phone index to word number
                # Assume phones are in order and belong to consecutive words
                phone_to_word_num = {}
                word_num = 0
                current_word_phone_count = 0
                
                for phone_idx, phone_info in enumerate(phones):
                    phone_to_word_num[phone_idx] = word_num
                    current_word_phone_count += 1
                    
                    # Simple heuristic: update word_num after processing several phones
                    # (This depends on actual word-phone alignment; for now use word_number from csv)
                
                # For now, just store word scores indexed by phone index
                # Downstream code can handle frame-level interpolation if needed
                self.utt_word_scores_by_frame[utt_id] = word_scores_agg        
        wav_id, wav_path = [], []
        
        with open(self.wav_scp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                wav_id.append(parts[0])
                wav_path.append(" ".join(parts[1:]))
        self.wav_dict = dict(zip(wav_id, wav_path))
        # load phone alignments
        self.alignment_dict, self.alignment_phn_dict = self.load_phone_alignments(self.alignment_ctm)
        
        # flatten self.alignment_dict and word_csv_lines, 
        flattened_alignment = []
        for utt_id in self.alignment_dict:
            ali = self.alignment_dict[utt_id]
            for seg_idx in ali:
                entry = {
                    'utt_id': utt_id,
                    'seg_idx': seg_idx,
                    'phn_info': ali[seg_idx]
                }
                flattened_alignment.append(entry)
        for ali_entry, word_csv_line in zip(flattened_alignment, word_csv_lines):
            # check if ali_entry[phn_info][0] == phn_index
            assert ali_entry['phn_info'][0] == int(word_csv_line.split(',')[0]), f"Mismatch in phone index for utt {ali_entry['utt_id']} seg {ali_entry['seg_idx']}"
            parts = word_csv_line.split(',')
            phn_index = int(parts[0])
            word_number = int(parts[1])
            word_text = parts[2]
            acc_score = int(parts[3])
            stress_score = int(parts[4])
            total_score = int(parts[5])
            # Append word scores to alignment entry
            ali_entry['word_number'] = word_number
            ali_entry['word_text'] = word_text
            ali_entry['wrd_acc_score'] = acc_score
            ali_entry['wrd_stress_score'] = stress_score
            ali_entry['wrd_total_score'] = total_score
        
        # make flattened_alignment back to self.alignment_dict, gather py file id
        for ali_entry in flattened_alignment:
            utt_id = ali_entry['utt_id']
            seg_idx = ali_entry['seg_idx']
            phn_info = ali_entry['phn_info']
            word_number = ali_entry['word_number']
            word_text = ali_entry['word_text']
            acc_score = ali_entry['wrd_acc_score']
            stress_score = ali_entry['wrd_stress_score']
            total_score = ali_entry['wrd_total_score']
            
            # Append word scores to alignment entry
            self.alignment_dict[utt_id][seg_idx].append(word_number)
            self.alignment_dict[utt_id][seg_idx].append(word_text)
            self.alignment_dict[utt_id][seg_idx].append(acc_score)
            self.alignment_dict[utt_id][seg_idx].append(stress_score)
            self.alignment_dict[utt_id][seg_idx].append(total_score)
    
        
        '''
        self.alignment_dict['020220264']
        # {
            "0": [phn_id, start_time, duration, word_number, word_text, acc_score, stress_score, total_score]
            "1": [phn_id, start_time, duration, word_number, word_text, acc_score, stress_score, total_score] 
        }
        '''
                
        # normalize the GOP feature using the training set mean and std (only count the valid token features, exclude the padded tokens).
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)
        # normalize the utt_label to 0-2 (same with phn score range)
        self.utt_label = self.utt_label / 5
        # the last dim is word_id, so not normalizing
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5
        # phone related
        # [phn_id, phn_score]
        self.phn_label[:, :, 1] = self.phn_label[:, :, 1]
        
        # combine self.alignment_dict with phn_label, word label, utt_label
        # assert the length is the same
        assert len(self.alignment_dict) == self.phn_label.shape[0] == self.word_label.shape[0] == self.utt_label.shape[0]
        
        # append phn_label to alignment_dict
        self.dict_all = {}
        for i, ((utt_id, ali), phn_label, word_label, utt_label) in enumerate(zip(self.alignment_dict.items(),  self.phn_label, self.word_label, self.utt_label)):
            # unpad phn_label and word_label, only keep valid length, padded with -1
            phn_label = phn_label[phn_label[:, 0] != -1]
            word_label = word_label[word_label[:, -1] != -1]
            
            self.dict_all[utt_id] = {}
            # combine self.alignment_dict with phn_label and word_label
            self.dict_all[utt_id]["phn"] = self.alignment_dict[utt_id]
            assert len(self.alignment_dict[utt_id]) == phn_label.shape[0] == word_label.shape[0], f"Length mismatch for utt {utt_id}"
            # append phn_label and word_label to each phone entry
            for j, seg_idx in enumerate(self.alignment_dict[utt_id]):
                # phn_label: [phn_id, phn_score]
                phn_score = phn_label[j, 1].item()
                # word_label: [acc_score, stress_score, total_score, word_id]
                acc_score = word_label[j, 0].item()
                stress_score = word_label[j, 1].item()
                total_score = word_label[j, 2].item()
                # word_id = word_label[j, 3].item()
                
                # append to alignment_dict
                
                # for self.dict_all[utt_id]["phn"][seg_idx][: -3] 
                # normalize the word_label to 0-2 
                self.dict_all[utt_id]["phn"][seg_idx][-3: ] = [round(x/5, 2) for x in self.dict_all[utt_id]["phn"][seg_idx][-3: ]]

                
                self.dict_all[utt_id]["phn"][seg_idx].append(round(phn_score, 2))
                # self.dict_all[utt_id]["phn"][seg_idx].append(acc_score)
                # self.dict_all[utt_id]["phn"][seg_idx].append(stress_score)
                # self.dict_all[utt_id]["phn"][seg_idx].append(total_score)
                # self.dict_all[utt_id]["phn"][seg_idx].append(word_id)
            
            # pure_phn_id, pure_phn, pure_id, phone, start_time, duration, word_number, word_text, wrd_acc, wrd_stress, wrd_total, phn_score
            
            self.dict_all[utt_id]["utt_label"] = [round(x, 3) for x in utt_label.tolist()] # round to .3f
        
        # return self.dict_all
            
    def dump_dataset(self, output_file, format='json'):
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(self.dict_all, f, indent=4)
    
    def load_phone_alignments(self, alignment_ctm):
        with open(self.alignment_ctm, 'r') as f:
            """
            ID, dummy_spk, start_time, duration, phn (ind)
            # 000030012 1 0.000 0.550 1
            # 000030012 1 0.550 0.170 227
            # 000030012 1 0.720 0.140 17
            # 000030024 1 0.000 0.570 1
            # 000030024 1 0.570 0.120 219
            # 000030024 1 0.690 0.150 129
            # 000030024 1 0.840 0.180 288
            ---
            return a dict:     
            where sil and spn segments are removed
            assume 1 represents SIL or SPN       
            
            result:
            {
                "000030012": {
                    "0": [227, 0.550, 0.170],
                    "1": [17, 0.720, 0.140],
                    ...
                },                
                "000030024": {
                    "0": [219, 0.570, 0.120],
                    "1": [129, 0.690, 0.150],
                    "2": [288, 0.840, 0.180],
                }
            }
            """
            
            # group the alignment by utt_id, store in a dict
            self.alignment_dict = {}
            self.alignment_phn_dict = {}
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0]
                # gather by same utt_id
                start = float(parts[2])
                dur = float(parts[3])
                phn_ind = int(parts[4])
                
                # get pure phone name
                phn_pure = self.phoneindex_to_purephone.get(phn_ind, None)
                
                # remove SIL and SPN segments
                if phn_pure == "SIL" or phn_pure == "SPN":
                    continue
                
                # initialize utt_id entry if not exists
                if utt_id not in self.alignment_dict:
                    self.alignment_dict[utt_id] = {}
                    self.alignment_phn_dict[utt_id] = []
                
                # calculate the segment index for this utterance
                seg_idx = len(self.alignment_dict[utt_id])
                
                # store [phn_ind, start_time, duration] with format: "seg_idx": [phn_ind, start, dur]
                # use pure_id for alignment_phn_dict
                # pdb.set_trace()
                pure_id = self.index_to_pureindex.get(phn_ind, None)
                pure_phone = self.purephone_dict.get(pure_id, None)
                phone = self.phone_dict.get(phn_ind, None)
                
                # self.alignment_dict[utt_id][str(seg_idx)] = [pure_id, start, dur]
                self.alignment_dict[utt_id][str(seg_idx)] = [pure_id, pure_phone, phn_ind, phone, start, dur]
                
                # self.alignment_dict[utt_id][str(seg_idx)] = [phn_ind, start, dur]
                self.alignment_phn_dict[utt_id].append(phn_pure)
        return self.alignment_dict, self.alignment_phn_dict
    
    def load_phone_mappings(self, mapping_mode="phoneindex_to_purephone"):
        """_summary_

        Args:
            mapping_mode (str, optional): _description_. Defaults to "phoneindex_to_purephone".
            pure_phone: 41 phones
            phone: 352 phones
            index_to_pureindex: mapping from phone index to pure phone index, 352->41
            phoneindex_to_purephone: mapping from phone index to pure phone, 352->41
            
        Returns:
            _type_: _description_
        """
        phones_pure = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/lang_nosp/phones-pure.txt"
        phones = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/lang_nosp/phones.txt"
        phone_to_phone_pure = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/lang_nosp/phone-to-pure-phone.int"
        # load as dict, key: phone, value: index
        self.phone2index = {}
        with open(phones, "r") as f:
            for line in f:
                parts = line.strip().split()
                phone = parts[0]
                index = int(parts[1])
                self.phone2index[phone] = index
        self.index2phone = {v: k for k, v in self.phone2index.items()}
        
        # load phone to pure phone mapping
        self.phone_to_pure_dict = {}
        with open(phone_to_phone_pure, "r") as f:
            for line in f:
                parts = line.strip().split()
                phone_ind = int(parts[0])
                pure_phone_ind = int(parts[1])
                self.phone_to_pure_dict[phone_ind] = pure_phone_ind
        # 
        self.pure_phone2index = {}
        with open(phones_pure, "r") as f:
            for line in f:
                parts = line.strip().split()
                phone = parts[0]
                index = int(parts[1])
                self.pure_phone2index[phone] = index
        self.index2pure_phone = {v: k for k, v in self.pure_phone2index.items()}
        
        # given phone index, make a dict return pure phone
        self.phoneindex_to_purephone = {}
        for phone_ind in self.phone_to_pure_dict:
            pure_phone_ind = self.phone_to_pure_dict[phone_ind]
            pure_phone = self.index2pure_phone[pure_phone_ind]
            self.phoneindex_to_purephone[phone_ind] = pure_phone
        
        
        if mapping_mode == "pure_phone":
            return self.index2pure_phone
        if mapping_mode == "phone":
            return self.index2phone
        if mapping_mode == "index_to_pureindex":
            return self.phone_to_pure_dict
        if mapping_mode == "phoneindex_to_purephone":
            return self.phoneindex_to_purephone
  
    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat
 
    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        # feat, phn_label, phn_id, utt_label, word_label
        #[word_id, phn_id]
        return self.feat[idx, :], self.feat_energy[idx, :], self.feat_dur[idx, :], self.phn_label[idx, :, 1], self.phn_label[idx, :, 0], self.utt_label[idx, :], self.word_label[idx, :], self.word_id[idx,:]

class GoPDataset_ver2(GoPDataset):
    '''
    Extended GoPDataset with on-the-fly SSL feature extraction.
    This dataset integrates AutoSSLLoader to extract SSL features dynamically.
    '''
    def __init__(self, set, am='librispeech', model_name="wavlm_large", sample_rate=16000, 
                 freeze=True, freeze_feature_extractor=True, save_path=None, 
                 output_all_hiddens=False, encoder_type=None):
        """
        Initialize the dataset with SSL feature extractor.
        
        Args:
            set: 'train' or 'test'
            am: acoustic model type ('librispeech', 'paiia', 'paiib')
            model_name: SSL model name (e.g., 'wavlm_large', 'hubert_large')
            sample_rate: target sample rate for audio processing
            freeze: whether to freeze the SSL model
            freeze_feature_extractor: whether to freeze the feature extractor
            save_path: path to save the model
            output_all_hiddens: whether to output all hidden states
            encoder_type: custom encoder type if needed
        """
        super().__init__(set, am)
        
        # Initialize SSL feature extractor
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.ssl_model = AutoSSLLoader(
            model_name=model_name,
            freeze=freeze,
            freeze_feature_extractor=freeze_feature_extractor,
            save_path=save_path if save_path else "/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models",
            output_all_hiddens=output_all_hiddens,
            encoder_type=encoder_type
        )
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.ssl_model is not None:
            self.ssl_model = self.ssl_model.to(self.device)
            if freeze:
                self.ssl_model.eval()
        
        # Cache for SSL features (optional, can be memory intensive)
        self.ssl_feature_cache = {}
        
    def audio_pipeline(self, wav_path):
        """
        Load and process audio file to extract waveform.
        
        Args:
            wav_path: path to the audio file (can be a command string for Kaldi)
        
        Returns:
            sig: processed waveform tensor
        """
        import torchaudio
        
        # Handle Kaldi pipe format (e.g., "sox ... |" or "ffmpeg ... |")
        if '|' in wav_path:
            # For Kaldi pipe commands, use kaldi_io or execute the command
            import subprocess
            import io
            # Execute the command and read from stdout
            try:
                process = subprocess.Popen(
                    wav_path.rstrip('|').strip(),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout, stderr = process.communicate()
                
                # Load from bytes
                audio_bytes = io.BytesIO(stdout)
                waveform, sr = torchaudio.load(audio_bytes)
            except Exception as e:
                raise RuntimeError(f"Failed to load audio from pipe command: {wav_path}\nError: {e}")
        else:
            # Load waveform directly from file
            waveform, sr = torchaudio.load(wav_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Return as 1D tensor
        sig = waveform.squeeze(0)
        return sig
    
    def extract_ssl_features(self, waveform):
        """
        Extract SSL features from waveform.
        
        Args:
            waveform: 1D tensor of audio samples
        
        Returns:
            ssl_features: SSL features tensor
        """
        if self.ssl_model is None:
            return None
        
        # Ensure waveform is on the correct device and has batch dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]
        
        waveform = waveform.to(self.device)
        
        # Extract features
        with torch.no_grad():
            ssl_features = self.ssl_model(waveform)
            # ssl_features shape: [batch, time, feature_dim]
        
        return ssl_features.squeeze(0)  # Remove batch dimension [time, feature_dim]
    
    def __getitem__(self, idx):
        """
        Get item with SSL features extracted on-the-fly.
        
        Returns:
            utt_id: utterance ID
            utt_info: utterance information dict (phone alignments, labels, etc.)
            feat: GOP features
            feat_energy: energy features
            feat_dur: duration features
            wav_path: path to audio file
            ssl_features: SSL features extracted from audio
        """
        utt_id = list(self.dict_all.keys())[idx]
        wav_path = self.wav_dict[utt_id]
        
        # Check cache first
        if utt_id in self.ssl_feature_cache:
            ssl_features = self.ssl_feature_cache[utt_id]
        else:
            # Extract SSL features on-the-fly
            waveform = self.audio_pipeline(wav_path)
            ssl_features = self.extract_ssl_features(waveform)
            
            # Optionally cache (comment out if memory is limited)
            # self.ssl_feature_cache[utt_id] = ssl_features
        
        return (
            utt_id, 
            self.dict_all[utt_id], 
            self.feat[idx, :], 
            self.feat_energy[idx, :], 
            self.feat_dur[idx, :], 
            wav_path,
            ssl_features
        )
    
    def dump_ssl_features(self, output_file, format='npz', mode="all"):
        """
        Extract and save SSL features for all utterances.
        
        Args:
            output_file: path to save the features (file path for mode='all', directory path for mode='per_utt')
            format: 'npz' (numpy archive) or 'pt' (pytorch) or 'h5' (hdf5)
            mode: 'all' to save all features in one file, 'per_utt' to save each utterance separately
        """
        import os
        from tqdm import tqdm
        
        print(f"Extracting SSL features for {len(self)} utterances using {self.model_name}...")
        
        ssl_features_dict = {}
        
        for idx in tqdm(range(len(self))):
            utt_id = list(self.dict_all.keys())[idx]
            wav_path = self.wav_dict[utt_id]
            
            # Extract features
            waveform = self.audio_pipeline(wav_path)
            ssl_features = self.extract_ssl_features(waveform)
            
            # Convert to numpy for storage
            ssl_features_dict[utt_id] = ssl_features.cpu().numpy()
        
        # if mode == "all", save all utterances in one file
        # if mode == "per_utt", create a folder and save each utterance in separate file
        if mode == "per_utt":
            os.makedirs(output_file, exist_ok=True)
            if format == "npz":
                for utt_id in ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.npy")
                    np.save(utt_file, ssl_features_dict[utt_id])
                print(f"SSL features saved to directory {output_file} in npz format (per utterance files)")
            elif format == "pt":
                for utt_id in ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.pt")
                    torch.save(torch.from_numpy(ssl_features_dict[utt_id]), utt_file)
                print(f"SSL features saved to directory {output_file} in pytorch format (per utterance files)")
            elif format == "h5":
                import h5py
                for utt_id in ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.h5")
                    with h5py.File(utt_file, 'w') as hf:
                        hf.create_dataset('ssl_features', data=ssl_features_dict[utt_id], compression='gzip')
                print(f"SSL features saved to directory {output_file} in hdf5 format (per utterance files)")
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pt', or 'h5'")
        if mode == "all":
            # mode == "all": save all features in one file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save based on format
            if format == 'npz':
                np.savez_compressed(output_file, **ssl_features_dict)
                print(f"SSL features saved to {output_file} (compressed npz format)")
            elif format == 'pt':
                # Convert back to tensors for pytorch format
                ssl_features_tensor_dict = {k: torch.from_numpy(v) for k, v in ssl_features_dict.items()}
                torch.save(ssl_features_tensor_dict, output_file)
                print(f"SSL features saved to {output_file} (pytorch format)")
            elif format == 'h5':
                import h5py
                with h5py.File(output_file, 'w') as hf:
                    for utt_id, features in ssl_features_dict.items():
                        hf.create_dataset(utt_id, data=features, compression='gzip')
                print(f"SSL features saved to {output_file} (hdf5 format)")
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pt', or 'h5'")
        
        return ssl_features_dict
    
    def load_ssl_features(self, input_file, format='npz'):
        """
        Load pre-extracted SSL features into cache.
        
        Args:
            input_file: path to the saved features
            format: 'npz' (numpy archive) or 'pt' (pytorch) or 'h5' (hdf5)
        """
        print(f"Loading SSL features from {input_file}...")
        
        if format == 'npz':
            data = np.load(input_file)
            for utt_id in data.files:
                self.ssl_feature_cache[utt_id] = torch.from_numpy(data[utt_id])
        elif format == 'pt':
            data = torch.load(input_file)
            self.ssl_feature_cache = data
        elif format == 'h5':
            import h5py
            with h5py.File(input_file, 'r') as hf:
                for utt_id in hf.keys():
                    self.ssl_feature_cache[utt_id] = torch.from_numpy(hf[utt_id][:])
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pt', or 'h5'")
        
        print(f"Loaded SSL features for {len(self.ssl_feature_cache)} utterances")
    
    def frame_feat_to_phn_feat(self, ssl_features, phn_info, mode="mean", feat_frame_shift=0.02):
        """
        Convert frame-level SSL features to phone-level features.
        
        Args:
            ssl_features: tensor of shape [time, feature_dim]
            phn_info: dict of phone segments with start_time and duration
            mode: aggregation mode ('mean', 'max', 'sum')
        
        Returns:
            phn_features: list of phone-level SSL features
        """
        phn_features = []
        frame_shift = 0.02  # assuming 10ms frame shift for SSL
        # import pdb; pdb.set_trace()
        for idx, phn in phn_info.items():
            start_time = phn[4]
            duration = phn[5]
            start_frame = int(start_time / frame_shift)
            end_frame = int((start_time + duration) / frame_shift)
            
            if start_frame >= ssl_features.shape[0]:
                phn_feat = torch.zeros(ssl_features.shape[1])
            else:
                end_frame = min(end_frame, ssl_features.shape[0])
                phn_frames = ssl_features[start_frame:end_frame, :]
                
                if mode == "mean":
                    phn_feat = phn_frames.mean(dim=0)
                elif mode == "max":
                    phn_feat, _ = phn_frames.max(dim=0)
                elif mode == "sum":
                    phn_feat = phn_frames.sum(dim=0)
                else:
                    raise ValueError(f"Unsupported mode: {mode}. Use 'mean', 'max', or 'sum'")
            
            phn_features.append(phn_feat)
        
        return torch.stack(phn_features)  # shape: [num_phones, feature_dim]
        
    def generate_phn_ssl_features(self, mode="mean"):
        """
        Generate GOP-like SSL features aligned with phones for all utterances.
        
        Args:
            mode: aggregation mode ('mean', 'max', 'sum')
        
        Returns:
            phn_ssl_features_dict: dict of utterance ID to phone-level SSL features
        """
        phn_ssl_features_dict = {}
        
        for idx in range(len(self)):
            utt_id = list(self.dict_all.keys())[idx]
            wav_path = self.wav_dict[utt_id]
            
            # Extract SSL features
            waveform = self.audio_pipeline(wav_path)
            ssl_features = self.extract_ssl_features(waveform)
            
            # Get phone info
            phn_info = self.dict_all[utt_id]['phn']
            # Convert to phone-level features
            phn_ssl_features = self.frame_feat_to_phn_feat(ssl_features, phn_info, mode=mode)
            
            # append to dict
            phn_ssl_features_dict[utt_id] = phn_ssl_features
            
            # append to phn_ssl to self.dict_all
            self.dict_all[utt_id]['phn_ssl_features'] = phn_ssl_features
        
        return phn_ssl_features_dict
    
    def dump_phn_ssl_features(self, output_file, phn_ssl_features_dict, format='npz', mode="all"):
        """
        Save phone-level SSL features to disk.
        
        Args:
            output_file: path to save the features
            phn_ssl_features_dict: dict of utterance ID to phone-level SSL features
            format: 'npz' (numpy archive) or 'pt' (pytorch) or 'h5' (hdf5)
            mode: 'all' to save all features in one file, 'per_utt' to save each utterance separately
              "gopt_like" is the feature similar to 
              https://github.com/YuanGongND/gopt
              where Time dimension (phone level) is padded to 50 frames
        """
        import os
        
        if mode == "per_utt":
            os.makedirs(output_file, exist_ok=True)
            if format == "npz":
                for utt_id in phn_ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.npy")
                    np.save(utt_file, phn_ssl_features_dict[utt_id].cpu().numpy())
                print(f"Phone-level SSL features saved to directory {output_file} (per utterance files)")
            if format == "npy":
                for utt_id in phn_ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.npy")
                    np.save(utt_file, phn_ssl_features_dict[utt_id].cpu().numpy())
                print(f"Phone-level SSL features saved to directory {output_file} (per utterance files)")
            elif format == "pt":
                for utt_id in phn_ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.pt")
                    torch.save(phn_ssl_features_dict[utt_id], utt_file)
                print(f"Phone-level SSL features saved to directory {output_file} (per utterance files)")
            elif format == "h5":
                import h5py
                for utt_id in phn_ssl_features_dict:
                    utt_file = os.path.join(output_file, f"{utt_id}.h5")
                    with h5py.File(utt_file, 'w') as hf:
                        hf.create_dataset('phn_ssl_features', data=phn_ssl_features_dict[utt_id].cpu().numpy(), compression='gzip')
                print(f"Phone-level SSL features saved to directory {output_file} (per utterance files)")
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pt', or 'h5'")
        
        if mode == "all":
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            if format == 'npz':
                np.savez_compressed(output_file, **{k: v.cpu().numpy() for k, v in phn_ssl_features_dict.items()})
                print(f"Phone-level SSL features saved to {output_file} (compressed npz format)")
            elif format == 'pt':
                torch.save(phn_ssl_features_dict, output_file)
                print(f"Phone-level SSL features saved to {output_file} (pytorch format)")
            elif format == 'h5':
                import h5py
                with h5py.File(output_file, 'w') as hf:
                    for utt_id, features in phn_ssl_features_dict.items():
                        hf.create_dataset(utt_id, data=features.cpu().numpy(), compression='gzip')
                print(f"Phone-level SSL features saved to {output_file} (hdf5 format)")
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pt', or 'h5'")
        if mode == "gopt_like":
            # default return as numpy without utt_id
            # os.makedirs(os.path.dirname(output_file), exist_ok=True)
            max_phn_len = 50  # pad to 50 phones
            feature_dim = list(phn_ssl_features_dict.values())[0].shape[1]
            gopt_like_dict = {}
            for utt_id, features in phn_ssl_features_dict.items():
                padded_feat = torch.zeros((max_phn_len, feature_dim))
                length = min(features.shape[0], max_phn_len)
                padded_feat[:length, :] = features[:length, :]
                gopt_like_dict[utt_id] = padded_feat
            
            if format == 'dict_npz':
                if not output_file.endswith('.npz'):
                    output_file += '.npz'
                np.savez_compressed(output_file, **{k: v.cpu().numpy() for k, v in gopt_like_dict.items()})
                print(f"GOPt-like phone-level SSL features saved to {output_file} (compressed npz format)")
            if format == 'npz':
                # save value in order without utt_id
                # stack as one array
                if not output_file.endswith('.npz'):
                    output_file += '.npz'
                stacked_array = np.stack([v.cpu().numpy() for k, v in gopt_like_dict.items()])
                # np.savez_compressed(output_file, *[v.cpu().numpy() for k, v in gopt_like_dict.items()])
                np.savez_compressed(output_file, features=stacked_array)
                print(f"GOPt-like phone-level SSL features saved to {output_file} (compressed npz format)")
            if format == "npy":
                # save value in order without utt_id
                if not output_file.endswith('.npy'):
                    output_file += '.npy'
                stacked_array = np.stack([v.cpu().numpy() for k, v in gopt_like_dict.items()])
                np.save(output_file, stacked_array)
                print(f"GOPt-like phone-level SSL features saved to {output_file} (npy format)")
            elif format == 'pt':
                torch.save(gopt_like_dict, output_file)
                print(f"GOPt-like phone-level SSL features saved to {output_file} (pytorch format)")
            elif format == 'h5':
                import h5py
                with h5py.File(output_file, 'w') as hf:
                    for utt_id, features in gopt_like_dict.items():
                        hf.create_dataset(utt_id, data=features.cpu().numpy(), compression='gzip')
                print(f"GOPt-like phone-level SSL features saved to {output_file} (hdf5 format)")
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'npz', 'pt', or 'h5'")
    
    def clear_cache(self):
        """Clear SSL feature cache to free memory."""
        self.ssl_feature_cache.clear()


# Example usage:
if __name__ == "__main__":
    import argparse
    import os
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract SSL features from SpeechOcean762 dataset')
    parser.add_argument('--am', type=str, default='librispeech', 
                        choices=['librispeech', 'paiia', 'paiib'],
                        help='Acoustic model type')
    parser.add_argument('--ssl_model', type=str, default='wavlm_large',
                        help='SSL model name')
    parser.add_argument('--data_root', type=str, 
                        default='/home/kevingenghaopeng/MDD/IF-MDD/data_so762',
                        help='Root directory for data storage')
    parser.add_argument('--skip_basic', action='store_true',
                        help='Skip basic GoPDataset JSON creation')
    parser.add_argument('--skip_frame', action='store_true',
                        help='Skip frame-level SSL feature extraction')
    parser.add_argument('--skip_phone', action='store_true',
                        help='Skip phone-level SSL feature extraction')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extraction even if files exist')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        choices=['train', 'test'],
                        help='Dataset splits to process')
    parser.add_argument('--phn_ssl_format', type=str, default='npy',
                        choices=['npy', 'npz', 'pt', 'h5', 'dict_npz'],
                        help='Format to save phone-level SSL features')
    
    args = parser.parse_args()
    
    am = args.am
    ssl_model_name = args.ssl_model
    data_root = args.data_root
    
    # Ensure data root exists
    os.makedirs(data_root, exist_ok=True)
    
    print("="*80)
    print(f"Configuration:")
    print(f"  Acoustic Model: {am}")
    print(f"  SSL Model: {ssl_model_name}")
    print(f"  Data Root: {data_root}")
    print(f"  Splits: {args.splits}")
    print(f"  Force re-extraction: {args.force}")
    print("="*80)
    
    # ===== Example 1: Basic GoPDataset (no SSL features) =====
    if not args.skip_basic:
        print("\n" + "="*80)
        print("Creating basic GoPDataset...")
        print("="*80)
        for split in args.splits:
            split_prefix = 'tr' if split == 'train' else 'te'
            json_file = os.path.join(data_root, f'gop_librispeech_{split}.json')
            
            if os.path.exists(json_file) and not args.force:
                print(f"\n[{split.upper()}] JSON file already exists: {json_file}")
                print(f"  Skipping... (use --force to re-create)")
                continue
            
            print(f"\n[{split.upper()}] Processing {split} split...")
            
            # Create basic dataset and dump to JSON
            dataset = GoPDataset(split, am=am)
            dataset.dump_dataset(json_file, format='json')
            print(f"  Basic dataset saved: {json_file}")
    
    # ===== Example 2: GoPDataset_ver2 with SSL features (on-the-fly extraction) =====
    print("\n" + "="*80)
    print(f"GoPDataset_ver2: Extracting SSL features using {ssl_model_name}")
    print("="*80)
    
    for split in args.splits:
        split_prefix = 'tr' if split == 'train' else 'te'
        print(f"\n{'='*80}")
        print(f">>> Processing {split.upper()} split <<<")
        print(f"{'='*80}")
        
        # Define output paths
        frame_dir = os.path.join(data_root, f'{split_prefix}_frm_{ssl_model_name}_per_utt')
        phone_npz = os.path.join(data_root, f'{split_prefix}_{ssl_model_name}_phn.{args.phn_ssl_format}')
        phone_dir = os.path.join(data_root, f'{split_prefix}_{ssl_model_name}_phn_per_utt')
        
        # Check if all outputs already exist
        all_exist = (os.path.exists(frame_dir) and 
                     os.path.exists(phone_npz) and 
                     os.path.exists(phone_dir))
        
        if all_exist and not args.force:
            print(f"\n[{split.upper()}] All SSL features already exist:")
            print(f"  Frame-level: {frame_dir}")
            print(f"  Phone-level (npz): {phone_npz}")
            print(f"  Phone-level (h5): {phone_dir}")
            print(f"  Skipping... (use --force to re-extract)")
            continue
        
        # Create SSL dataset
        print(f"\n[{split.upper()}] Creating SSL dataset...")
        dataset_ssl = GoPDataset_ver2(split, am=am, model_name=ssl_model_name, sample_rate=16000)
        
        # Test single item (only for first split)
        if split == 'train':
            print("\nTesting single item from SSL dataset...")
            utt_id, utt_info, feat, feat_energy, feat_dur, wav_path, ssl_features = dataset_ssl[0]
            print(f"  Utterance ID: {utt_id}")
            print(f"  SSL features shape: {ssl_features.shape}")
            print(f"  GOP features shape: {feat.shape}")
        
        # ===== Extract and save frame-level SSL features =====
        if not args.skip_frame:
            if os.path.exists(frame_dir) and not args.force:
                print(f"\n[{split.upper()}] Frame-level SSL features already exist: {frame_dir}")
                print(f"  Skipping frame extraction...")
            else:
                print(f"\n[{split.upper()}] Extracting frame-level SSL features...")
                dataset_ssl.dump_ssl_features(
                    frame_dir,
                    format='h5',
                    mode='per_utt'
                )
        
        # ===== Generate and save phone-level SSL features =====
        if not args.skip_phone:
            # Check if we need to generate phone features
            need_phone_features = (not os.path.exists(phone_npz) or 
                                   not os.path.exists(phone_dir) or 
                                   args.force)
            
            if need_phone_features:
                print(f"\n[{split.upper()}] Generating phone-level SSL features...")
                phn_ssl_dict = dataset_ssl.generate_phn_ssl_features(mode="mean")
                
                # Save in GOPt-like format (padded numpy array)
                if not os.path.exists(phone_npz) or args.force:
                    print(f"  Saving GOPt-like format...")
                    dataset_ssl.dump_phn_ssl_features(
                        phone_npz,
                        phn_ssl_dict,
                        format=args.phn_ssl_format,
                        mode='gopt_like'
                    )
                else:
                    print(f"  GOPt-like format already exists: {phone_npz}")
                
                # Save in per-utterance HDF5 format
                if not os.path.exists(phone_dir) or args.force:
                    print(f"  Saving per-utterance HDF5 format...")
                    dataset_ssl.dump_phn_ssl_features(
                        phone_dir,
                        phn_ssl_dict,
                        format=args.phn_ssl_format,
                        mode='per_utt'
                    )
                else:
                    print(f"  Per-utterance HDF5 already exists: {phone_dir}")
            else:
                print(f"\n[{split.upper()}] Phone-level SSL features already exist:")
                print(f"  GOPt-like format: {phone_npz}")
                print(f"  Per-utterance HDF5: {phone_dir}")
                print(f"  Skipping phone extraction...")
        
        print(f"\n[{split.upper()}] Complete! Dataset size: {len(dataset_ssl)}")
    
    print("\n" + "="*80)
    print("All processing complete!")
    print("="*80)
