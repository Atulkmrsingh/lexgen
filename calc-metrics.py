import evaluate
import argparse
import numpy as np

chrf = evaluate.load('chrf')
parser = argparse.ArgumentParser(description='Evaluation of Dictionary Predictions')

parser.add_argument(
    '--ref', type=str, help='Reference GT'
)
parser.add_argument(
    '--predictions', type=str, help='Predictions'
)

args = parser.parse_args()
refs, preds = [], []
with open(args.ref) as f:    
    for line in f.readlines():
        refs.append(line.strip())

with open(args.predictions) as f:    
    for line in f.readlines():
        preds.append(line.strip())
chrf_mean = []
refs=refs[290:366]
preds=preds[290:366]
for r, p in zip(refs, preds):
    refs_arr = [i.strip() for i in r.split(',')]
    chrf_score = chrf.compute(references=[refs_arr], predictions=[p], word_order=1, char_order=4)['score']
    chrf_mean.append(chrf_score)
print('Chrf score is ', np.mean(chrf_mean))
with open("score.txt", "a") as file:
    file.write(str(np.mean(chrf_mean)) + "\t")

# Calculate Precision@1
precision1_scores = []
for r, p in zip(refs, preds):
    refs_arr = [i.strip() for i in r.split(',')]
    precision1_score = 1.0 if p in refs_arr else 0.0
    precision1_scores.append(precision1_score)

print('Precision@1 score is ', np.mean(precision1_scores))
with open("score.txt", "a") as file:
    file.write(str(np.mean(precision1_scores)) + "\t")

# import evaluate
# import argparse
# import numpy as np
# from rapidfuzz.distance import Levenshtein

# chrf = evaluate.load('chrf')
# parser = argparse.ArgumentParser(description='Evaluation of Dictionary Predictions')

# parser.add_argument(
#     '--ref', type=str, help='Reference GT'
# )
# parser.add_argument(
#     '--predictions', type=str, help='Predictions'
# )

# args = parser.parse_args()
# refs, preds = [], []
# with open(args.ref) as f:    
#     for line in f.readlines():
#         refs.append(line.strip())

# with open(args.predictions) as f:    
#     for line in f.readlines():
#         preds.append(line.strip())
# refs=refs[:366]
# preds=preds[:366]
# chrf_mean = []
# for r, p in zip(refs, preds):
#     refs_arr = [i.strip() for i in r.split(',')]
#     chrf_score = chrf.compute(references=[refs_arr], predictions=[p], word_order=1, char_order=4)['score']
#     chrf_mean.append(chrf_score)

# print('Chrf score is ', np.mean(chrf_mean))
# with open("score.txt", "a") as file:
#     file.write(str(np.mean(chrf_mean)) + "\t")

# precision1_scores = []
# for r, p in zip(refs, preds):
#     refs_arr = [i.strip() for i in r.split(',')]
    
#     precision1_score = 0.0
#     for ref in refs_arr:
#         if Levenshtein.distance(p, ref) <= 1: 
#             precision1_score = 1.0
#             break
#     precision1_scores.append(precision1_score)

# print('Precision@1 score with one-character tolerance is ', np.mean(precision1_scores))
# with open("score.txt", "a") as file:
#     file.write(str(np.mean(precision1_scores)) + "\t")
