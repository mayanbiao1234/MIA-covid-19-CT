from sklearn.metrics import f1_score
import numpy as np


List_true = []
with open(r"/root/data/covid_prediction_zhongzhuan/True_label.txt","r") as f:  ## do not touch
    for line in f.readlines():
        line = line.strip('\n')
        List_true.append(int(line))
List_true = np.array(List_true)

List_preb = []
with open(r"/root/Desktop/vote/lowest_2of_3_8.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_preb.append(int(line))
List_preb = np.array(List_preb)
print("F1:{:.4f}".format(f1_score(List_true, List_preb, average='macro')))
