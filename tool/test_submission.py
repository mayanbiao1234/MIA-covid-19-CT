import numpy as np
import json
import pandas as pd
import csv

all_path = '/root/data/covid19_test/test_result/3_36_0.8463/'
json_path = '/root/data/covid19/test_json/'

List_subset1 = []
with open(all_path + "subset1.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset1.append(int(line))
List_subset1 = np.array(List_subset1)

List_subset2 = []
with open(all_path + "subset2.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset2.append(int(line))
List_subset2 = np.array(List_subset2)

List_subset3 = []
with open(all_path + "subset3.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset3.append(int(line))
List_subset3 = np.array(List_subset3)

List_subset4 = []
with open(all_path + "subset4.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset4.append(int(line))
List_subset4 = np.array(List_subset4)

List_subset5 = []
with open(all_path + "subset5.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset5.append(int(line))
List_subset5 = np.array(List_subset5)

List_subset6 = []
with open(all_path + "subset6.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset6.append(int(line))
List_subset6 = np.array(List_subset6)

List_subset7 = []
with open(all_path + "subset7.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset7.append(int(line))
List_subset7 = np.array(List_subset7)

List_subset8 = []
with open(all_path + "subset8.txt","r") as f:
    for line in f.readlines():
        line = line.strip('\n')
        List_subset8.append(int(line))
List_subset8 = np.array(List_subset8)

def json_read_file(path):
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
        List_oldname = []
        for i in range(len(data)):
            List_oldname.append(data[i]['old_id'])
    List_oldname = np.array(List_oldname)
    print('finish!')
    return List_oldname

subset1_oldid = json_read_file(json_path + 'subset1.json')
subset2_oldid = json_read_file(json_path + 'subset2.json')
subset3_oldid = json_read_file(json_path + 'subset3.json')
subset4_oldid = json_read_file(json_path + 'subset4.json')
subset5_oldid = json_read_file(json_path + 'subset5.json')
subset6_oldid = json_read_file(json_path + 'subset6.json')
subset7_oldid = json_read_file(json_path + 'subset7.json')
subset8_oldid = json_read_file(json_path + 'subset8.json')

List_all = np.hstack([List_subset1, List_subset2, List_subset3, List_subset4, List_subset5, List_subset6, List_subset7, List_subset8])
Json_all = np.hstack([subset1_oldid, subset2_oldid, subset3_oldid, subset4_oldid, subset5_oldid, subset6_oldid, subset7_oldid, subset8_oldid])

sub_covid = []
sub_noncovid = []
if len(List_all) == len(Json_all):
    for i in range(len(List_all)):
        if List_all[i] == 1:
            sub_noncovid.append(Json_all[i])
        if List_all[i] == 0:
            sub_covid.append(Json_all[i])
# print(sub_noncovid)
# print(sub_covid)

if len(List_all) == (len(sub_noncovid) + len(sub_covid)):
    print(len(sub_noncovid))
    print(len(sub_covid))
    sub_noncovid = ",".join(sub_noncovid)
    sub_covid = ",".join(sub_covid)
    print(sub_noncovid)
    with open('/root/data/covid19_test/test_submision/3_36_0.8463/non-covid.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([sub_noncovid])
    with open('/root/data/covid19_test/test_submision/3_36_0.8463/covid.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([sub_covid])
    # sub_noncovid.to_csv('/root/data/covid19_test/test_submision/3_36_0.8463/non-covid.csv', index=False)
    # sub_covid.to_csv('/root/data/covid19_test/test_submision/3_36_0.8463/covid.csv', index=False)
else:
    print('wrong!')






