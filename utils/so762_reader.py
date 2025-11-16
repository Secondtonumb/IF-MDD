import numpy as np
import json
import kaldi_io
import speechbrain as sb
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import speechbrain as sb

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
            self.word_csv = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/so762_alignment/kaldi_gop_align/train/word_score.csv"
        elif set == 'test':
            self.wav_scp = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/test/wav.scp"
            self.feat = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_feat.npy'), dtype=torch.float)
            self.feat_energy = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_energy_feat.npy'), dtype=torch.float)
            self.feat_dur = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_dur_feat.npy'), dtype=torch.float)
            self.phn_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_label_phn.npy'), dtype=torch.float)
            self.utt_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_label_word.npy'), dtype=torch.float)
            self.word_id = torch.tensor(np.load('data_so762/hiertfr/'+dir+'/te_word_id.npy'), dtype=torch.float)
            self.alignment_ctm = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/so762_alignment/kaldi_gop_align/test/phones_int.ctm"
            self.word_ctm = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/exp/ali_test/ctm"
            self.word_csv = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/so762_alignment/kaldi_gop_align/test/word_score.csv"
        
        wav_id, wav_path = [], []
        with open(self.wav_scp, 'r') as f:
            for line in f:
                parts = line.strip().split()
                wav_id.append(parts[0])
                wav_path.append(" ".join(parts[1:]))
        self.wav_dict = dict(zip(wav_id, wav_path))
        
        # load phone alignments
        self.alignment_dict, self.alignment_phn_dict = self.load_phone_alignments(self.alignment_ctm)
                
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
        for i, ((utt_id, ali), phn_scores, word_scores, utt_score) in enumerate(zip(self.alignment_dict.items(), self.phn_label, self.word_label, self.utt_label)):
            # phn_scores remove masked, masked value = -1
            phn_scores = phn_scores[phn_scores[:,0] != -1]
            phn_scores = phn_scores[:, 1]
            # collateion check
            assert len(ali) == len(phn_scores), f"Length mismatch for utt_id {utt_id}: alignment length {len(ali)} vs phn_scores length {len(phn_scores)}"
            # append phn_scores to ali
            """
            ali[0] Phone index, ali[1] start_time, ali[2] duration
            # return:
            {
                "utt_id": 
                {   phone_info:
                        "0": {
                        "phone_index", phone index,
                        "start_time": start time,
                        "duration: duration,
                        "phn_score": phn_score
                        },
                        "1": {
                        "phone_index", phone index,
                        "start_time": start time,
                        "duration: duration,
                        "phn_score": phn_score
                        },
                },
            """
            
            pdb.set_trace()
            
    
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
                self.alignment_dict[utt_id][str(seg_idx)] = [pure_id, start, dur]
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


# train_set_dict = {
#     "wav_scp": "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/train/wav.scp",
#     "feat": "data_so762/hiertfr/seq_data_librispeech_v3/tr_feat.npy",
#     "enerey": "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/hiertfr/seq_data_librispeech_v3/tr_energy_feat.npy",
#     "dur": "data_so762/hiertfr/seq_data_librispeech_v3/tr_dur_feat.npy",
#     "tr_label_phn": "data_so762/hiertfr/seq_data_librispeech_v3/tr_label_phn.npy",
#     "tr_label_word": "data_so762/hiertfr/seq_data_librispeech_v3/tr_label_word.npy",
#     "tr_label_utt": "data_so762/hiertfr/seq_data_librispeech_v3/tr_label_utt.npy"
# }

# test_set_dict  = {
#     "wav_scp": "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/test/wav.scp",
#     "feat": "data_so762/hiertfr/seq_data_librispeech_v3/te_feat.npy",
#     "enerey": "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/hiertfr/seq_data_librispeech_v3/te_energy_feat.npy",
#     "dur": "/home/kevingenghaopeng/MDD/IF-MDD/data_so762/hiertfr/seq_data_librispeech_v3/te_dur_feat.npy",
#     "te_label_phn": "data_so762/hiertfr/seq_data_librispeech_v3/te_label_phn.npy",
#     "te_label_word": "data_so762/hiertfr/seq_data_librispeech_v3/te_label_word.npy",
#     "te_label_utt": "data_so762/hiertfr/seq_data_librispeech_v3/te_label_utt.npy"
# }

# phone_tokenizer_41 = "/home/kevingenghaopeng/Tools/kaldi/egs/gop_speechocean762/s5/data/lang_nosp/phones-pure.txt"
# # load as dict, key: phone, value: index
# phone2index_41 = {}
# with open(phone_tokenizer_41, "r") as f:
#     for line in f:
#         parts = line.strip().split()
#         phone = parts[0]
#         index = int(parts[1])
#         phone2index_41[phone] = index
# index2phone_41 = {v: k for k, v in phone2index_41.items()}

# phone_encoder = sb.dataio.encoder.CTCTextEncoder()
# phone_encoder.update_from_iterable(index2phone_41)

import pdb; pdb.set_trace()
# merge wav.scp into speechbrain style dataset

# def merge_wav_scp_with_features(wav_scp, set_dict, prefix=""):
#     dataset = {}
#     # load wav.scp
#     wav_dict = {}
#     # for key, path in kaldi_io.read_wav_scp(wav_scp):
#         # wav_dict[key] = path
#     # file = np.loadtxt(wav_scp, dtype=str, delimiter="  ", comments=None, usecols=1)

#     for line in open(wav_scp, "r"):
#         parts = line.strip().split()
#         key = parts[0]
#         path = " ".join(parts[1:])
#         wav_dict[key] = path
#     import pdb; pdb.set_trace()
#     # load features
#     feat_dict = {}
#     enerey_dict = {}
#     dur_dict = {}
#     label_phn_dict = {}
#     label_word_dict = {}
#     label_utt_dict = {}
    
#     # T_p: phoneme sequence length, padded to max length in the set
#     feat_array = np.load(set_dict["feat"], allow_pickle=True) # [2500, T_p, feat_dim]
#     enerey_array = np.load(set_dict["enerey"], allow_pickle=True) # [2500, T_p, 7]
#     dur_array = np.load(set_dict["dur"], allow_pickle=True) # [2500, T_p, 1]
#     label_phn_array = np.load(set_dict[f"{prefix}label_phn"], allow_pickle=True) # [2500, 50, 2]
#     label_word_array = np.load(set_dict[f"{prefix}label_word"], allow_pickle=True)
#     label_utt_array = np.load(set_dict[f"{prefix}label_utt"], allow_pickle=True)
    
#     # get phoneme duration from label_phn_array:
#     # label_phn_array[i],  padded matrix of shape (T_p, 2)
#     # each row: [phoneme_index, phoneme_score], padded with -1
#     # get each row's duration and store in phn_duration [2500]

#     phn_len_list = []
#     phn_list = []
#     import pdb; pdb.set_trace()
#     for i in range(label_phn_array.shape[0]):
#         phn_index = label_phn_array[i][:, 0]
#         phn_index = phn_index[phn_index != -1]  # remove padding
#         if len(phn_index) == 0:
#             duration = 0
#         else:
#             duration = len(phn_index) 
#         phn_list.append(phone_encoder.decode_ndim(phn_index.astype(int)).split())
#         phn_len_list.append(duration)
#     import pdb; pdb.set_trace()

#     # label word score:
#     # label_word_array[i]  #[T_P, 4]
#     # each row: [accuracy_score, stress_score, total_score, word_position_of_current_phoneme], padded with -1
#     # get each row's word score and store in word_score_list [2500]
#     word_score_list = []
#     for i in range(label_word_array.shape[0]):
#         word_scores = label_word_array[i][:, 2]
#         word_scores = word_scores[word_scores != -1]  # remove padding
#         if len(word_scores) == 0:
#             avg_score = 0.0
#         else:
#             avg_score = float(np.mean(word_scores))
#         word_score_list.append(avg_score)
#     import pdb; pdb.set_trace()
    
#     # label_utt_array:  # [2500, 5]
#     # each row: [accuracy_score, completeness, fluency, prosody, total_score]
#     utt_score_list = []
#     for i in range(label_utt_array.shape[0]):
#         utt_scores = label_utt_array[i]
#         utt_scores = utt_scores[utt_scores != -1]  # remove padding
#     import pdb; pdb.set_trace()
    
#     # merge, for list, asuume the order is the same as wav.scp
#     for i, key in enumerate(wav_dict.keys()):
#         phn_len = phn_len_list[i]
#         feat_dict[key] = feat_array[i]
#         enerey_dict[key] = enerey_array[i]
#         dur_dict[key] = dur_array[i]
#         label_phn_dict[key] = label_phn_array[i]
#         label_word_dict[key] = label_word_array[i]
#         label_utt_dict[key] = label_utt_array[i]
#     for key in wav_dict:
#         dataset[key] = {
#             "wav": wav_dict[key],
#             "phn_len": phn_len,
#             "feat": feat_dict[key][: phn_len, :],
#             "enerey": enerey_dict[key][: phn_len, :],
#             "dur": dur_dict[key][: phn_len, :],
#             "label_phn": label_phn_dict[key],
#             "label_word": label_word_dict[key],
#             "label_utt": label_utt_dict[key]
#         }
#     pdb.set_trace()
#     return dataset

# train_data = merge_wav_scp_with_features(train_set_dict["wav_scp"], train_set_dict, prefix="tr_")
# am='librispeech'
am = 'librispeech'
tr_dataset = GoPDataset('train', am=am)
tr_dataloader = DataLoader(tr_dataset, batch_size=1, shuffle=True)
te_dataset = GoPDataset('test', am=am)
te_dataloader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

import pdb; pdb.set_trace()