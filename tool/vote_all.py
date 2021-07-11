## 投票，依据逻辑投票, 这里先不管非数字问题，先将数字问题依据逻辑投票

import numpy as np
import json
from collections import Counter

# 读取数据
List1 = []
List2 = []
List3 = []
List4 = []
# List5 = []
# List6 = []
# List7 = []
# List8 = []
# List9 = []


with open(r"/root/data/covid_prediction_zhongzhuan/3.8252.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        List1.append(int(line))
with open(r"/root/data/covid_prediction_zhongzhuan/3.8262.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        List2.append(int(line))
with open(r"/root/data/covid_prediction_zhongzhuan/8.8197.txt", "r") as f:
     for line in f.readlines():
         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
         List3.append(int(line))
with open(r"/root/data/covid_prediction_zhongzhuan/8.8198.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # 去掉列表中每一个元素的换行符
        List4.append(int(line))
# with open(r"/root/data/covid_prediction_zhongzhuan/3.8319.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#         List5.append(int(line))
# with open(r"/root/data/covid_prediction_zhongzhuan/8.8244.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#         List6.append(int(line))
# with open(r"/root/data/covid_prediction_zhongzhuan/3.8318.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#         List7.append(int(line))
# with open(r"/root/data/covid_prediction_zhongzhuan/3.8315.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#         List8.append(int(line))
# with open(r"/root/data/covid_prediction_zhongzhuan/7.8082.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
#         List9.append(int(line))


answer = []
for i in range(0, 374):
    # print(i)
    v_flag = 0
    list_ = []
    list_.append(List1[i])
    list_.append(List2[i])
    list_.append(List3[i])
    list_.append(List4[i])
    # list_.append(List5[i])
    # list_.append(List6[i])
    # list_.append(List7[i])
    # list_.append(List8[i])
    # list_.append(List9[i])

    votes = Counter(list_)
    print(votes)

    # print(votes.most_common(2))

    # print("第%d行和第%d行的结果" % (final_list[i*2]+1,final_list[i*2+1]+1))
    if len(votes.most_common(3)) == 3:
        if votes.most_common(3)[0][1] == votes.most_common(3)[1][1] == votes.most_common(3)[2][1]:
            v_flag = 1
            print("第%d行的结果投出来【3】平票啦，需要抉择:%s,%s,%s" % (i + 1, votes.most_common(3)[0][0], votes.most_common(3)[1][0],
            votes.most_common(3)[2][0]))

    if len(votes.most_common(2)) == 2:
        if votes.most_common(2)[0][1] == votes.most_common(2)[1][1]:
            v_flag = 2
            print("第%d行结果投出来【2】平票啦，需要抉择:%s,%s" % (i + 1, votes.most_common(2)[0][0], votes.most_common(2)[1][0]))
    if v_flag == 0:
        print(votes.most_common(1)[0][0])
        # print(votes.most_common(1)[0][0])
        answer.append(votes.most_common(1)[0][0])
    else:
        print(votes.most_common(1)[0][0])
        answer.append(votes.most_common(1)[0][0])
    #print("")
#         answer.append('0')
#         answer.append('0')
# print(answer)


with open(r"/root/Desktop/vote/lowest_2of_3_8.txt", "w") as f:
    for i in answer:
        f.write(str(i))  # 自带文件关闭功能，不需要再写f.close()
        f.write('\n')