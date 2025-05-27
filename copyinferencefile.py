#python calc-metrics.py --ref ref_fname.txt --predictions output.txt
import evaluate
import argparse
import numpy as np

chrf = evaluate.load('chrf')
ter = evaluate.load('ter')
parser = argparse.ArgumentParser(description='Evaluation of Dictionary Predictions')
# parser.add_argument(
#     "--src", type= str, help = 'Source to calculate bleurt'
# )
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

# with open(args.src) as f:    
#     for line in f.readlines():
#         src.append(line.strip())


domains=[0,66,130,184,286,362,456]
for i in range(1,len(domains)):
    chrf_mean = []
    for  r, p in zip( refs[domains[i-1]:domains[i]], preds[domains[i-1]:domains[i]]):
        refs_arr = [i.strip() for i in r.split(',')]
        chrf_score = chrf.compute(references = [refs_arr], predictions =[p], word_order=1, char_order=4)['score']
        chrf_mean.append(chrf_score)
    print('Chrf score is ', np.mean(chrf_mean))
    with open("score.txt", "a") as file:
        file.write(str(np.mean(chrf_mean))+"	")
    
    ter_mean = []
    for  r, p in zip( refs[domains[i-1]:domains[i]], preds[domains[i-1]:domains[i]]):
        refs_arr = [i.strip() for i in r.split(',')]
        ter_score = ter.compute(references = [refs_arr], predictions =[p])['score']
        ter_mean.append(ter_score)
    print('Ter score is ', np.mean(ter_mean))
    with open("score.txt", "a") as file:
        file.write(str(np.mean(ter_mean))+"	")
with open("score.txt", "a") as file:
    file.write("\n")

