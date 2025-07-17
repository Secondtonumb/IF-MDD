import json
import argparse
import random

def split_list(in_list, ratio=0.1):
    small = random.sample(in_list, int(ratio*len(in_list)))
    big = [x for x in in_list if x not in small]
    return big, small

def split_by_speaker(in_json, ratio=0.1):
    # spks = set(in_json[wav_id]["spk_id"] for wav_id in in_json)
    out_train = {}
    out_dev = {}
    dev = random.sample(list(in_json.keys()), int(ratio * len(in_json)))
    train = [i for i in in_json if i not in dev]
            
    
    for i in train:
        out_train.update({i: in_json[i]})
    for i in dev:
        out_dev.update({i: in_json[i]})
    return out_train, out_dev

def main(args):
    with open(args.in_json, "r") as f:
        in_data = json.load(f)
    import pdb; pdb.set_trace()
    out_train, out_dev = split_by_speaker(in_data, ratio=args.dev_ratio)

    with open(args.out_json_train, "w") as f:
        json.dump(out_train, f, indent=2)
    with open(args.out_json_dev, "w") as f:
        json.dump(out_dev, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", type=str)
    p.add_argument("--dev_ratio", type=float, default=0.1)
    p.add_argument("--out_json_train", type=str)
    p.add_argument("--out_json_dev", type=str)
    args = p.parse_args()

    main(args)



