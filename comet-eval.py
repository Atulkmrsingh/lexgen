# python comet-eval.py  --src input.txt --ref ref_fname.txt --predictions output.txt
from comet import download_model, load_from_checkpoint
import argparse
import numpy as np

model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)


parser = argparse.ArgumentParser(description='Evaluation of Dictionary Predictions')
parser.add_argument(
    "--src", type= str, help = 'Source to calculate bleurt'
)
parser.add_argument(
    '--ref', type=str, help = 'Reference GT'
)
parser.add_argument(
    '--predictions', type=str, help='Predictions'
)

args = parser.parse_args()
refs, preds, src  = [], [], []
with open(args.ref) as f:    
    for line in f.readlines():
        refs.append(line.strip())

with open(args.predictions) as f:    
    for line in f.readlines():
        preds.append(line.strip())

with open(args.src) as f:    
    for line in f.readlines():
        src.append(line.strip())

chrf_mean = []
data = []
for s, r, p in zip(src, refs, preds):
    refs_arr = [i.strip() for i in r.split(',')]
    # print('refs_arra ', refs_arr)
    max_ps = []
    for rfs in refs_arr:
        data.append({'src': s, 'mt':p, 'ref':rfs})
score = model.predict(data, batch_size =128, gpus = 1)
print('Final mean COMET score is ', score['system_score'])
with open("score.txt", "a") as file:
    file.write(str(score['system_score'])+"	")
