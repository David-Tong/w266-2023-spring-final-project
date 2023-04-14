"""
Use Skim and Read mode to predict for SQuAD version 2.0.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
from collections import OrderedDict

def score(args):
    # skim mode
    null_files = args.input_null_files.split(",")

    cof = [1] * len(null_files) 
    all_scores = OrderedDict()
    idx = 0
    for file in null_files:
        with open('input/' + file, 'r') as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(cof[idx] * score)
        idx += 1
    output_scores = {}
    for (key, scores) in all_scores.items():
        mean_score = 0.0
        for score in scores:
            mean_score += score
        mean_score /= float(len(scores))
        output_scores[key] = mean_score

    # read mode
    nbest_files = args.input_nbest_files.split(",")

    best_cof = [1] * len(nbest_files)
    all_nbest = OrderedDict()    
    idx = 0
    for file in nbest_files:
        with open('input/' + file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, entries) in input_data.items():
                if key not in all_nbest:
                    all_nbest[key] = collections.defaultdict(float)
                for entry in entries:
                    all_nbest[key][entry["text"]] += best_cof[idx] * entry["probability"]
        idx += 1
    output_predictions = {}
    for (key, entry_map) in all_nbest.items():
        sorted_texts = sorted(
            entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
        best_text = sorted_texts[0]
        output_predictions[key] = best_text

    # need to fine tune it
    threshold = args.threshold

    for qid in output_predictions.keys():
        if output_scores[qid] >= threshold:
            output_predictions[qid] = ""

    output_prediction_file = "output/predictions.json"
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_predictions, indent=4) + "\n")

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--input_null_files', type=str, default="null_odds.json")
    parser.add_argument('--input_nbest_files', type=str, default="nbest_predictions.json")
    parser.add_argument('--threshold', default=0, type=float)
    parser.add_argument("--predict_file", default="data/dev-v2.0.json")
    args = parser.parse_args()
    score(args)

if __name__ == "__main__":
    main()
