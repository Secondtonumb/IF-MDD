import os
import sys
import json
import argparse
import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.edit_distance import wer_details_for_batch
from speechbrain.dataio.wer import print_alignments, _print_alignment
from speechbrain.utils.metric_stats import MetricStats, ErrorRateStats

# assume "err" phns are
EDIT_SYMBOLS = {
    "eq": "=",  # when tokens are equal
    "ins": "I",
    "del": "D",
    "sub": "S",
}

class MpdStats(MetricStats):
    """Compute MDD eval metrics, adapted from speechbrain.utils.metric_stats.MetricStats
    see speechbrain.utils.metric_stats.MetricStats
    """

    def __init__(self, merge_tokens=False, split_tokens=False, space_token="_"):
        self.clear()
        self.merge_tokens = merge_tokens
        self.split_tokens = split_tokens
        self.space_token = space_token

    def append(
        self,
        ids,
        predict,
        canonical,
        perceived,
        predict_len=None,
        canonical_len=None,
        perceived_len=None,
        ind2lab=None,
    ):
        self.ids.extend(ids)
        
        if predict_len is not None:
            predict = undo_padding(predict, predict_len)

        if canonical_len is not None:
            canonical = undo_padding(canonical, canonical_len)
        if perceived_len is not None:
            perceived = undo_padding(perceived, perceived_len)

        if ind2lab is not None:
            predict = ind2lab(predict)
            canonical = ind2lab(canonical)
            perceived = ind2lab(perceived)

        if self.merge_tokens:
            predict = merge_char(predict, space=self.space_token)
            target = merge_char(target, space=self.space_token)

        if self.split_tokens:
            predict = split_word(predict, space=self.space_token)
            target = split_word(target, space=self.space_token)
        
        ## remove parallel sil in cano and perc
        canonical, perceived = rm_parallel_sil_batch(canonical, perceived)
        assert len(canonical) == len(perceived)  # make sure cano and perc are aligned

        ## remove all sil in hyp
        predict = [[x for x in y if x!= "sil"] for y in predict]


        alignments = [extract_alignment(c, p) for c, p in zip(canonical, perceived)]
        perc_hyp_alignments = [extract_alignment(p, h) for p, h in zip(perceived, predict)]
        wer_details = wer_details_for_batch(ids=ids,
                                           refs=[[s for s in c if s != "sil"] for c in canonical],
                                           hyps=predict,
                                           compute_alignments=True)
        
        ## let's be clear about the two alignments' names, rename the keys
        wer_details_perc = wer_details_for_batch(ids=ids,
                                            refs=[[s for s in p if s != "sil"] for p in perceived],
                                            hyps=predict,
                                            compute_alignments=True)
        for a, p, det, det_2 in zip(alignments, perceived, wer_details, wer_details_perc):
            det["alignment_cano2hyp"] = det.pop("alignment")
            det["canonical"] = det.pop("ref_tokens")
            det["hypothesis"] = det.pop("hyp_tokens")
            det.update({"alignment_cano2perc": a})
            det.update({"perceived": [s for s in p if s != "sil"]})
            # import pdb; pdb.set_trace()
            det["alignment_perc2hyp"] = det_2.pop("alignment")

            ## canonical and hypothesis alignment with <eps>
            cano_align_cano_hyp = []
            hyp_align_cano_hyp = []
            op_cano_hyp = []
            for (x, y, z) in det["alignment_cano2hyp"]:
                cano_align_cano_hyp.append(det["canonical"][y] if y is not None else "<eps>")
                hyp_align_cano_hyp.append(det["hypothesis"][z] if z is not None else "<eps>")
                op_cano_hyp.append(x)
            det["cano_align_cano_hyp"] = cano_align_cano_hyp
            det["hyp_align_cano_hyp"] = hyp_align_cano_hyp
            det["op_cano_hyp"] = op_cano_hyp
            ## canonical vs perceived  alignment with <eps>
            cano_align_cano_perc = []
            perc_align_cano_perc = []
            op_cano_perc = []
            for (x, y, z) in det["alignment_cano2perc"]:
                cano_align_cano_perc.append(det["canonical"][y] if y is not None else "<eps>")
                perc_align_cano_perc.append(det["perceived"][z] if z is not None else "<eps>")
                op_cano_perc.append(x)
            det["cano_align_cano_perc"] = cano_align_cano_perc
            det["perc_align_cano_perc"] = perc_align_cano_perc
            det["op_cano_perc"] = op_cano_perc
            ## perceived vs hypothesis alignment with <eps>
            perc_align_perc_hyp = []
            hyp_align_perc_hyp = []
            op_perc_hyp = []
            for (x, y, z) in det["alignment_perc2hyp"]:
                perc_align_perc_hyp.append(det["perceived"][y] if y is not None else "<eps>")
                hyp_align_perc_hyp.append(det["hypothesis"][z] if z is not None else "<eps>")
                op_perc_hyp.append(x)
            det["perc_align_perc_hyp"] = perc_align_perc_hyp
            det["hyp_align_perc_hyp"] = hyp_align_perc_hyp
            det["op_perc_hyp"] = op_perc_hyp

        # import pdb; pdb.set_trace()
        self.scores.extend(wer_details)

    def summarize(self, field=None):
        """Summarize the error_rate and return relevant statistics.
        * See MetricStats.summarize()
        """
        # self.summary = wer_summary(self.scores)
        self.summary = mpd_summary(self.scores)

        # Add additional, more generic key
        self.summary["mpd_f1"] = self.summary["f1"]

        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream):
        """Write all relevant info (e.g., error rate alignments) to file.
        * See MetricStats.write_stats()
        """
        if not self.summary:
            self.summarize()

        print_mpd_details(self.scores, self.summary, filestream)

def mpd_eval_on_dataset(in_json, mpd_file=sys.stdout, per_file=None, with_sil=True):

    if per_file:
        error_rate_stats = ErrorRateStats()
    total_wer_details = []

    for wav_id, wav_data in in_json.items():
        cano_phns = wav_data["canonical_phn"].split()
        perc_phns = wav_data["phn"].split()
        cano_phns, perc_phns = rm_parallel_sil(cano_phns, perc_phns)
        assert len(cano_phns) == len(perc_phns)

        alignment = extract_alignment(cano_phns, perc_phns)


        hyp = [s for s in wav_data["hyp"].split() if s!= "sil"]
        # hyp = wav_data["hyp"].split()
        wer_details = wer_details_for_batch(ids=[wav_id],
                                           refs=cano_phns,
                                           hyps=[hyp],
                                           compute_alignments=True)[0]
        ## let's be clear about the two alignments' names, rename the keys
        wer_details["alignment_cano2hyp"] = wer_details.pop("alignment")
        wer_details["canonical"] = wer_details.pop("ref_tokens")
        wer_details["hypothesis"] = wer_details.pop("hyp_tokens")
        wer_details.update({"alignment_cano2perc": alignment})
        wer_details.update({"perceived": [s for s in perc_phns if s != "sil"]})
        wer_details.update({"wav_id": wav_id})
        
        
        total_wer_details.append(wer_details)


        if per_file:
            error_rate_stats.append(ids=[wav_id],
                                    target=[cano_phns],
                                    predict=[hyp])

    if per_file:
        error_rate_stats.write_stats(per_file)

    mpd_stats = mpd_summary(total_wer_details)
    
    print_mpd_details(total_wer_details, mpd_stats, mpd_file)

def mpd_stats_new(arr):
    # del detection 
    ref_seq = arr[0].split(" ")
    ref_seq3 = arr[6].split(" ")
    op =  arr[2].split(" ")
    op3 = arr[8].split(" ")
    flag = 0
    for i in range( len(ref_seq) ):
        if(ref_seq[i] == "<eps>"):
            continue
        while(ref_seq3[flag] == "<eps>"):
            flag+=1  
        if( ref_seq[i]  == ref_seq3[flag] and ref_seq[i]!="<eps>" ):
            if( op[i] == "D"  and op3[flag] == "D" ):
                del_del+=1
            elif( op[i] == "D" and op3[flag] != "D" and op3[flag] != "="):
                del_del1+=1  
            elif( op[i] == "D" and op3[flag] != "D" and op3[flag] == "="):
                del_nodel+=1
            flag+=1  
            
    ## cor ins sub detection 
    ref_seq = arr[0].split(" ")
    human_seq = arr[1].split(" ")
    op =  arr[2].split(" ")
    human_seq2 = arr[3].split(" ")
    our_seq2 = arr[4].split(" ")
    op2 = arr[5].split(" ")
    flag = 0 
    import pdb; pdb.set_trace()
    for i in range( len(human_seq) ):
        if(human_seq[i] == "<eps>"):
            continue
        while(human_seq2[flag] == "<eps>"):
            flag+=1
        if( human_seq[i]  == human_seq2[flag] and human_seq[i]!="<eps>" ):
            if( op[i] == "="  and op2[flag] == "=" ):
                cor_cor+=1
            elif( op[i] == "=" and op2[flag] != "="):
                cor_nocor+=1


            if( op[i] == "S" and op2[flag] == "=" ):
                sub_sub+=1
            elif( op[i] == "S"  and op2[flag] !="=" and ref_seq[i] != our_seq2[flag]):
                sub_sub1+=1
            elif( op[i] == "S"  and op2[flag] !="=" and ref_seq[i] == our_seq2[flag]):
                sub_nosub+=1

            if(op[i] == "I" and op2[flag] == "=" ):
                ins_ins+=1
            elif( op[i] == "I" and op2[flag]!="=" and op2[flag]!="D"):
                ins_ins1+=1
            elif( op[i] == "I" and op2[flag]!="=" and op2[flag]=="D"):
                ins_noins+=1

            flag+=1

    sum1 = cor_cor + cor_nocor + sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel  
    print("sum:",sum1)
    TR = sub_sub + ins_ins + del_del + sub_sub1 + ins_ins1 + del_del1
    FR = cor_nocor
    FA = sub_nosub + ins_noins + del_nodel
    TA = cor_cor 
    recall = TR/(TR+FA)
    precision = TR/(TR+FR)
    print("Recall: %.4f" %(recall))
    print("Precision: %.4f" %(precision))
    print("f1:%.4f" % ( 2*precision*recall/(recall+precision)  ))

    print("TA: %.4f %d" %(cor_cor/(cor_cor+cor_nocor), TA))
    print("FR: %.4f %d" %(cor_nocor/(cor_cor+cor_nocor), FR))
    err_count = sub_sub+sub_sub1+sub_nosub+ins_ins+ins_ins1+ins_noins+del_del+del_del1+del_nodel
    false_accept = sub_nosub + ins_noins + del_nodel
    Correct_Diag = sub_sub + ins_ins + del_del
    Error_Diag =  sub_sub1 + ins_ins1 + del_del1
    print("FA: %.4f %d" %(false_accept/err_count, false_accept))
    print("Correct Diag: %.4f %d" %(Correct_Diag/(Correct_Diag+Error_Diag), Correct_Diag))
    print("Error Diag: %.4f %d" %(Error_Diag/(Correct_Diag+Error_Diag), Error_Diag))
    FAR = 1-recall
    FRR = cor_nocor/(cor_nocor+cor_cor)
    DER = Error_Diag / (Error_Diag + Correct_Diag)
    print("FAR: %.4f" %(FAR))
    print("FRR: %.4f" %(FRR))
    print("DER: %.4f" %(DER))

def mpd_summary(total_wer_details):

    total_ta, total_fr, total_fa, total_tr, total_cor_diag, total_err_diag = 0, 0, 0, 0, 0, 0
    total_ins, total_del, total_sub, total_eq = 0, 0, 0, 0
    cor_cor = 0
    cor_cor1 = 0 
    cor_nocor = 0

    sub_sub = 0
    sub_sub1 = 0
    sub_nosub = 0

    ins_ins = 0
    ins_ins1 = 0
    ins_noins =0

    del_del = 0
    del_del1 = 0
    del_nodel =0
    # 
    for det in total_wer_details:

        total_ins += len([a for a in det["alignment_cano2perc"] if a[0] == "I"])
        total_del += len([a for a in det["alignment_cano2perc"] if a[0] == "D"])
        total_sub += len([a for a in det["alignment_cano2perc"] if a[0] == "S"])
        total_eq += len([a for a in det["alignment_cano2perc"] if a[0] == "="])
        
        ## OLD metrics
        ta, fr, fa, tr, cor_diag, err_diag = mpd_stats(det["alignment_cano2perc"],
                                                       det["alignment_cano2hyp"],
                                                       det["canonical"],
                                                       det["perceived"],
                                                       det["hypothesis"])
        assert tr == (cor_diag + err_diag)
        det.update({
                      "ta": ta,
                      "fr": fr,
                      "fa": fa,
                      "tr": tr,
                      "cor_diag": cor_diag,
                      "err_diag": err_diag,
                    })

        total_ta += ta
        total_fr += fr
        total_fa += fa
        total_tr += tr
        total_cor_diag += cor_diag
        total_err_diag += err_diag
        
        ## New metrics
        # ref human
        cano_align_cano_perc = det['cano_align_cano_perc']
        perc_align_cano_perc = det['perc_align_cano_perc']
        op_cano_perc = det['op_cano_perc']
        assert len(cano_align_cano_perc) == len(perc_align_cano_perc) == len(op_cano_perc)
        # human hyp
        perc_align_perc_hyp = det['perc_align_perc_hyp']
        hyp_align_perc_hyp = det['hyp_align_perc_hyp']
        op_perc_hyp = det['op_perc_hyp']
        assert len(perc_align_perc_hyp) == len(hyp_align_perc_hyp) == len(op_perc_hyp)
        # ref hyp
        cano_align_cano_hyp = det['cano_align_cano_hyp']
        hyp_align_cano_hyp = det['hyp_align_cano_hyp']
        op_cano_hyp = det['op_cano_hyp']
        assert len(cano_align_cano_hyp) == len(hyp_align_cano_hyp) == len(op_cano_hyp)
        
        arr = [
            cano_align_cano_perc,
            perc_align_cano_perc,
            op_cano_perc,
            perc_align_perc_hyp,
            hyp_align_perc_hyp,
            op_perc_hyp,
            cano_align_cano_hyp,
            hyp_align_cano_hyp,
            op_cano_hyp
        ]
        # apply
        # del detection 
        ref_seq = arr[0]
        ref_seq3 = arr[6]
        op =  arr[2]
        op3 = arr[8]
        flag = 0
        for i in range( len(ref_seq) ):
            if(ref_seq[i] == "<eps>"):
                continue
            while(ref_seq3[flag] == "<eps>"):
                flag+=1  
            if( ref_seq[i]  == ref_seq3[flag] and ref_seq[i]!="<eps>" ):
                if( op[i] == "D"  and op3[flag] == "D" ):
                    del_del+=1
                elif( op[i] == "D" and op3[flag] != "D" and op3[flag] != "="):
                    del_del1+=1  
                elif( op[i] == "D" and op3[flag] != "D" and op3[flag] == "="):
                    del_nodel+=1
                flag+=1  
                
        ## cor ins sub detection 
        ref_seq = arr[0]
        human_seq = arr[1]
        op =  arr[2]
        human_seq2 = arr[3]
        our_seq2 = arr[4]
        op2 = arr[5]
        flag = 0 
        for i in range( len(human_seq) ):
            if(human_seq[i] == "<eps>"):
                continue
            while(human_seq2[flag] == "<eps>"):
                flag+=1
            if( human_seq[i]  == human_seq2[flag] and human_seq[i]!="<eps>"):
                if( op[i] == "="  and op2[flag] == "=" ):
                    cor_cor+=1
                elif( op[i] == "=" and op2[flag] != "="):
                    cor_nocor+=1

                if( op[i] == "S" and op2[flag] == "=" ):
                    sub_sub+=1
                elif( op[i] == "S"  and op2[flag] !="=" and ref_seq[i] != our_seq2[flag]):
                    sub_sub1+=1
                elif( op[i] == "S"  and op2[flag] !="=" and ref_seq[i] == our_seq2[flag]):
                    sub_nosub+=1

                if(op[i] == "I" and op2[flag] == "=" ):
                    ins_ins+=1
                elif( op[i] == "I" and op2[flag]!="=" and op2[flag]!="D"):
                    ins_ins1+=1
                elif( op[i] == "I" and op2[flag]!="=" and op2[flag]=="D"):
                    ins_noins+=1

                flag+=1
    # OLD metrics
    precision_old = 1.0*total_tr / (total_fr + total_tr)
    try:
        recall_old = 1.0*total_tr / (total_fa + total_tr)
    except:
        recall_old = 0.0
    try:
        f1_old = 2.0 * precision_old * recall_old / (precision_old + recall_old)
    except:
        f1_old = 0.0
    # New metrics
    
    sum1 = cor_cor + cor_nocor + sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel
    TR = sub_sub + ins_ins + del_del + sub_sub1 + ins_ins1 + del_del1
    FR = cor_nocor
    FA = sub_nosub + ins_noins + del_nodel
    TA = cor_cor 
    
    TOTAL_EQ = cor_cor + sub_sub + ins_ins + del_del
    TOTAL_SUB = sub_sub + sub_sub1 + sub_nosub
    TOTAL_INS = ins_ins + ins_ins1 + ins_noins
    TOTAL_DEL = del_del + del_del1 + del_nodel
    
    RECALL = TR/(TR+FA)
    PRECISION = TR/(TR+FR)
    F1 = 2*PRECISION*RECALL/(RECALL+PRECISION) if (RECALL + PRECISION) > 0 else 0.0
    # print("Recall: %.4f" %(recall_new))
    # print("Precision: %.4f" %(precision_new))
    # print("f1:%.4f" % (F1_new))

    # print("TA: %.4f %d" %(cor_cor/(cor_cor+cor_nocor), TA))
    # print("FR: %.4f %d" %(cor_nocor/(cor_cor+cor_nocor), FR))

    Correct_Diag = sub_sub + ins_ins + del_del
    Error_Diag =  sub_sub1 + ins_ins1 + del_del1
    # print("FA: %.4f %d" %(false_accept/err_count, false_accept))
    # print("Correct Diag: %.4f %d" %(Correct_Diag/(Correct_Diag+Error_Diag), Correct_Diag))
    # print("Error Diag: %.4f %d" %(Error_Diag/(Correct_Diag+Error_Diag), Error_Diag))
    FAR = 1-RECALL
    FRR = FR / (FR + TA) if (FR + TA) > 0 else 0.0
    DER = Error_Diag / (Error_Diag + Correct_Diag)
    # print("FAR: %.4f" %(FAR))
    # print("FRR: %.4f" %(FRR))
    # print("DER: %.4f" %(DER))

    return {
        "total_eq": total_eq,
        "total_sub": total_sub,
        "total_del": total_del,
        "total_ins": total_ins,
        "ta": total_ta,
        "fr": total_fr,
        "fa": total_fa,
        "tr": total_tr,
        "cor_diag": total_cor_diag,
        "err_diag": total_err_diag,
        "precision_old": precision_old,
        "recall_old": recall_old,
        "f1_old": 2.0 * precision_old * recall_old / (precision_old + recall_old),

        "SUM": sum1,
        "TOTAL_EQ": TOTAL_EQ,
        "TOTAL_SUB": TOTAL_SUB,
        "TOTAL_INS": TOTAL_INS,
        "TOTAL_DEL": TOTAL_DEL,
        "TR": TR,
        "FR": FR,
        "FA": FA,
        "TA": TA,
        "Correct_Diag" : Correct_Diag,
        "Error_Diag": Error_Diag,
        "FAR": FAR,
        "FRR": FRR,
        "DER": DER,
        "f1": F1,
        "precision": PRECISION,
        "recall": RECALL,
    }
    

def print_mpd_details(wer_details, mpd_stats, mpd_file):
    
    # OLD metrics

    print("In original annotation: \nTotal Eq: {}, Total Sub: {}, Total Del: {}, Total Ins: {}".format(\
            mpd_stats["total_eq"], mpd_stats["total_sub"], mpd_stats["total_del"], mpd_stats["total_ins"]), file=mpd_file)
    
    # Calculate percentages
    ta_fr_sum = mpd_stats["ta"] + mpd_stats["fr"]
    fa_tr_sum = mpd_stats["fa"] + mpd_stats["tr"]
    cd_ed_sum = mpd_stats["cor_diag"] + mpd_stats["err_diag"]

    ta_pct = mpd_stats["ta"] / ta_fr_sum if ta_fr_sum > 0 else 0.0
    fr_pct = mpd_stats["fr"] / ta_fr_sum if ta_fr_sum > 0 else 0.0
    fa_pct = mpd_stats["fa"] / fa_tr_sum if fa_tr_sum > 0 else 0.0
    tr_pct = mpd_stats["tr"] / fa_tr_sum if fa_tr_sum > 0 else 0.0
    cor_diag_pct = mpd_stats["cor_diag"] / cd_ed_sum if cd_ed_sum > 0 else 0.0
    err_diag_pct = mpd_stats["err_diag"] / cd_ed_sum if cd_ed_sum > 0 else 0.0


    print("Overall MPD results: \nTrue Accept: {}, False Rejection: {}, False Accept: {}, True Rejection: {}, Corr Diag: {}, Err Diag: {}".format(
        mpd_stats["ta"], mpd_stats["fr"], mpd_stats["fa"], mpd_stats["tr"], mpd_stats["cor_diag"], mpd_stats["err_diag"]), file=mpd_file)
    print("Percentages/Counts: TA%/TA: {:.2f}/{}  FR%/FR: {:.2f}/{}  FA%/FA: {:.2f}/{}  TR%/TR: {:.2f}/{}  Cor_diag%/Cor_diag: {:.2f}/{}  Err_diag%/Err_diag: {:.2f}/{}".format(
        ta_pct * 100, mpd_stats["ta"],
        fr_pct * 100, mpd_stats["fr"],
        fa_pct * 100, mpd_stats["fa"],
        tr_pct * 100, mpd_stats["tr"],
        cor_diag_pct * 100, mpd_stats["cor_diag"],
        err_diag_pct * 100, mpd_stats["err_diag"]), file=mpd_file)
    print("Precision: {}, Recall: {}, F1: {}".format(mpd_stats["precision_old"], mpd_stats["recall_old"], mpd_stats["f1_old"]), file=mpd_file)
    
    # New metrics
    print("NEW MPD stats: \nTotal Eq: {}, Total Sub: {}, Total Del: {}, Total Ins: {}".format(\
        mpd_stats["TOTAL_EQ"], mpd_stats["TOTAL_SUB"], mpd_stats["TOTAL_DEL"], mpd_stats["TOTAL_INS"]), file=mpd_file)
    
    print("New MPD results: \nTrue Accept: {}, False Rejection: {}, False Accept: {}, True Rejection: {}, Correct Diag: {}, Error Diag: {}".format(
        mpd_stats["TA"], mpd_stats["FR"], mpd_stats["FA"], mpd_stats["TR"], mpd_stats["Correct_Diag"], mpd_stats["Error_Diag"]), file=mpd_file)
    
    print("Percentages/Counts: TA%/TA: {:.2f}/{}  FR%/FR: {:.2f}/{}  FA%/FA: {:.2f}/{}  TR%/TR: {:.2f}/{}  Correct_diag%/Correct_diag: {:.2f}/{}  Error_diag%/Error_diag: {:.2f}/{}".format(
        mpd_stats["TA"] / (mpd_stats["TA"] + mpd_stats["FR"]) * 100 if (mpd_stats["TA"] + mpd_stats["FR"]) > 0 else 0.0, mpd_stats["TA"],
        mpd_stats["FR"] / (mpd_stats["TA"] + mpd_stats["FR"]) * 100 if (mpd_stats["TA"] + mpd_stats["FR"]) > 0 else 0.0, mpd_stats["FR"],
        mpd_stats["FA"] / (mpd_stats["FA"] + mpd_stats["TR"]) * 100 if (mpd_stats["FA"] + mpd_stats["TR"]) > 0 else 0.0, mpd_stats["FA"],
        mpd_stats["TR"] / (mpd_stats["FA"] + mpd_stats["TR"]) * 100 if (mpd_stats["FA"] + mpd_stats["TR"]) > 0 else 0.0, mpd_stats["TR"],
        mpd_stats["Correct_Diag"] / (mpd_stats["Correct_Diag"] + mpd_stats["Error_Diag"]) * 100 if (mpd_stats["Correct_Diag"] + mpd_stats["Error_Diag"]) > 0 else 0.0, mpd_stats["Correct_Diag"],
        mpd_stats["Error_Diag"] / (mpd_stats["Correct_Diag"] + mpd_stats["Error_Diag"]) * 100 if (mpd_stats["Correct_Diag"] + mpd_stats["Error_Diag"]) > 0 else 0.0, mpd_stats["Error_Diag"]), file=mpd_file)
    
    print("New Precision: {}, New Recall: {}, New F1: {}".format(mpd_stats["precision"], mpd_stats["recall"], mpd_stats["f1"]), file=mpd_file)

    # sort wer_details by key
    wer_details = sorted(wer_details, key=lambda x: x["key"])
    for det in wer_details:
        print("="*80, file=mpd_file)
        print(det["key"], file=mpd_file)
        print("Human annotation: Canonical vs Perceived:", file=mpd_file)
        _print_alignment(alignment=det["alignment_cano2perc"],
                         a=det["canonical"],
                         b=det["perceived"],
                         file=mpd_file)

        print("Model Prediction: Canonical vs Hypothesis:", file=mpd_file)
        _print_alignment(alignment=det["alignment_cano2hyp"],
                         a=det["canonical"],
                         b=det["hypothesis"],
                         file=mpd_file)
        print("True Accept: {}, False Rejection: {}, False Accept: {}, True Reject: {}, Corr Diag: {}, Err Diag: {}".format(\
                det["ta"], det["fr"], det["fa"], det["tr"], det["cor_diag"], det["err_diag"]), file=mpd_file)


def mpd_stats(align_c2p, align_c2h, c, p, h):
    """
    schema: [(operator, idx_i(None), idx_j(None))]
    c: canonical
    p: perceived
    h: hypothesis
    """
    cnt = 0
    ta, fr, fa, tr, cor_diag, err_diag = 0, 0, 0, 0, 0, 0
    import pdb; pdb.set_trace()
    # cano_len = 1 + max(x[1] for x in align_c2p)
    assert max(x[1] for x in align_c2p if x[1] is not None) ==  max(x[1] for x in align_c2h if x[1] is not None)

    i, j = 0, 0
    while i < len(align_c2p) and j < len(align_c2h):
        ## sub and del cases
        if align_c2p[i][1] is not None and \
           align_c2h[j][1] is not None and \
           align_c2p[i][1] == align_c2h[j][1]:
            assert align_c2p[i][0] != EDIT_SYMBOLS["ins"]
            assert align_c2h[j][0] != EDIT_SYMBOLS["ins"]
            if align_c2p[i][0] == EDIT_SYMBOLS["eq"]:
                ## canonical cases
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    ta += 1
                else:
                    fr += 1
            elif align_c2p[i][0] != EDIT_SYMBOLS["eq"]:
                ## mispronunciation cases
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    fa += 1
                else:
                    tr += 1
                    if align_c2p[i][0] != align_c2h[j][0]:
                        err_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["del"] and align_c2h[j][0] == EDIT_SYMBOLS["del"]:
                        cor_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["sub"] and align_c2h[j][0] == EDIT_SYMBOLS["sub"]:
                        if p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                            cor_diag += 1
                        else:
                            err_diag += 1
            i += 1
            j += 1
        ## ins cases
        elif align_c2p[i][1] is None and \
             align_c2h[j][1] is not None:
            fa += 1
            i += 1
        elif align_c2p[i][1] is not None and  \
             align_c2h[j][1] is None:
            fr += 1
            j += 1
        elif align_c2p[i][1] is None and align_c2h[j][1] is None:
            tr += 1
            if p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                cor_diag += 1
            else:
                err_diag += 1
            i += 1
            j += 1
    if i == len(align_c2p) and j != len(align_c2h):
        fr += len(align_c2h[j:])
    if i != len(align_c2p) and j == len(align_c2h):
        fa += len(align_c2p[j:])

    return ta, fr, fa, tr, cor_diag, err_diag


def mpd_stats(align_c2p, align_c2h, c, p, h):
    """
    schema: [(operator, idx_i(None), idx_j(None))]
    c: canonical
    p: perceived
    h: hypothesis
    """
    cnt = 0
    ta, fr, fa, tr, cor_diag, err_diag = 0, 0, 0, 0, 0, 0
    # cano_len = 1 + max(x[1] for x in align_c2p)
    assert max(x[1] for x in align_c2p if x[1] is not None) ==  max(x[1] for x in align_c2h if x[1] is not None)

    i, j = 0, 0
    while i < len(align_c2p) and j < len(align_c2h):
        ## sub and del cases
        if align_c2p[i][1] is not None and \
           align_c2h[j][1] is not None and \
           align_c2p[i][1] == align_c2h[j][1]:
            assert align_c2p[i][0] != EDIT_SYMBOLS["ins"]
            assert align_c2h[j][0] != EDIT_SYMBOLS["ins"]
            if align_c2p[i][0] == EDIT_SYMBOLS["eq"]:
                ## canonical cases
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    ta += 1
                else:
                    fr += 1
            elif align_c2p[i][0] != EDIT_SYMBOLS["eq"]:
                ## mispronunciation cases
                if align_c2h[j][0] == EDIT_SYMBOLS["eq"]:
                    fa += 1
                else:
                    tr += 1
                    if align_c2p[i][0] != align_c2h[j][0]:
                        err_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["del"] and align_c2h[j][0] == EDIT_SYMBOLS["del"]:
                        cor_diag += 1
                    elif align_c2p[i][0] == EDIT_SYMBOLS["sub"] and align_c2h[j][0] == EDIT_SYMBOLS["sub"]:
                        if p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                            cor_diag += 1
                        else:
                            err_diag += 1
            i += 1
            j += 1
        ## ins cases
        elif align_c2p[i][1] is None and \
             align_c2h[j][1] is not None:
            fa += 1
            i += 1
        elif align_c2p[i][1] is not None and  \
             align_c2h[j][1] is None:
            fr += 1
            j += 1
        elif align_c2p[i][1] is None and align_c2h[j][1] is None:
            tr += 1
            if p[align_c2p[i][2]] == h[align_c2h[j][2]]:
                cor_diag += 1
            else:
                err_diag += 1
            i += 1
            j += 1
    if i == len(align_c2p) and j != len(align_c2h):
        fr += len(align_c2h[j:])
    if i != len(align_c2p) and j == len(align_c2h):
        fa += len(align_c2p[j:])

    return ta, fr, fa, tr, cor_diag, err_diag

def extract_alignment(a, b, gap_token="sil"):
    """
    a, b are two aligned lists (i.e. same length)
    gap_token is the artificial token placeholder used in L2Arctic annotation. In this case is a `sil` token
    """
    alignment = []
    idx_a, idx_b = 0, 0
    for str_a, str_b in zip(a, b):
        if str_a == gap_token and str_b != gap_token:
            alignment.append((EDIT_SYMBOLS["ins"], None, idx_b))
            idx_b += 1
        elif str_a != gap_token and str_b == gap_token:
            alignment.append((EDIT_SYMBOLS["del"], idx_a, None))
            idx_a += 1
        elif str_a != gap_token and str_b != gap_token and str_a != str_b:
            alignment.append((EDIT_SYMBOLS["sub"], idx_a, idx_b))
            idx_a += 1
            idx_b += 1
        else:
            alignment.append((EDIT_SYMBOLS["eq"], idx_a, idx_b))
            idx_a += 1
            idx_b += 1
    return alignment

def rm_parallel_sil_batch(canos, percs):
    canos_out, percs_out = [], []
    assert len(canos) == len(percs)  ## batch size
    for cano, perc in zip(canos, percs):
        cano, perc = rm_parallel_sil(cano, perc)
        canos_out.append(cano)
        percs_out.append(perc)
    return canos_out, percs_out

def rm_parallel_sil(canos, percs):
    canos_out, percs_out = [], []
    if len(canos) == len(percs):
        for i in range(len(canos)):
            if canos[i] == "sil" and percs[i] == "sil":
                continue
            canos_out.append(canos[i])
            percs_out.append(percs[i])
    else:
        ## aligned
        for cano, perc in zip(canos, percs):
            if (cano==perc and cano=="sil"):
                continue
            canos_out.append(cano)
            percs_out.append(perc)
    assert len(canos_out) == len(percs_out)  # make sure cano and perc are aligned
    return canos_out, percs_out


def main(args):
    with open(args.json_path, "r") as f:
        json_data = json.load(f)
    per_file = open(args.per_file, "w")
    mpd_file = open(args.mpd_file, "w")
    mpd_eval_on_dataset(json_data, mpd_file, per_file)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", type=str)
    p.add_argument("--per_file", type=str, default=None)
    p.add_argument("--mpd_file", type=str, default=None)
    args = p.parse_args()

    main(args)
