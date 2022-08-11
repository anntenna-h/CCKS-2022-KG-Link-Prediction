import pickle
import os
import torch
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='model choices')
parser.add_argument('--model', help='designate model to generate rusult from')
args = parser.parse_args()

path = '../scores/'
dirc = os.listdir(path)



if args.model == 'ensemble':
    score_sum = torch.zeros([17957, 6676])
    score_tot = {}
    sfm = torch.nn.Softmax(dim=1)
    weights = {'score_0.5583.pickle':0.1, 
               'score_complEx.pickle':0.15, 
               'score_0.5545.pickle':0.1, 
               'score_flowing.pickle':0.1, 
               'score_0.5580.pickle':0.1, 
               'score_vocal.pickle':0.1,
               'score_lite_bumbling.pickle':0.08, 
               'score_fast.pickle':0.1, 
               'score_lite_polar.pickle':0.07, 
               'score_woven.pickle':0.1}

    for i in range(len(dirc)):
        file = open(path + dirc[i], 'rb')
        print(path + dirc[i])
        score_cur = pickle.load(file).cpu()
        score_tot[i] = score_cur
        file.close()
        if dirc[i] in ['score_complEx.pickle', 'score_rank_1500.pickle', 'score_lite_bumbling.pickle', 'score_lite_polar.pickle']:
            score_cur = sfm(score_cur)
        score_sum += score_cur * weights[dirc[i]]
else:
    try:
        file = open(path + 'score_' + args.model + '.pickle', 'rb')
        score_sum = pickle.load(file).cpu()
        file.close()
    except Exception as e: print(e)


def save_results(file_name, data_path = "../data/OpenBG-IMG/"):
    if not os.path.exists('../results/'):
        os.mkdir('../results/')
        
    for i in range(score_sum.size()[0]):
        score_one = score_sum[i]
        res = [str(i.item()) for i in torch.topk(score_one, k = 10, largest = True).indices] 
        with open('../results/' + file_name + '.txt','a+') as fp:
            fp.write(f"{' '.join(res)}\n") 

    with open(os.path.join(data_path, 'lhs_id'), 'r') as fp:
        lhs2id = fp.readlines()
        lhs2id = [re.findall(r'(.+?)\t(.+?)\n',i)[0] for i in lhs2id]
        lhs2id = {a[1]:a[0] for a in lhs2id}

    with open(os.path.join(data_path, 'rhs_id'), 'r') as fp:
        rhs2id = fp.readlines()
        rhs2id = [re.findall(r'(.+?)\t(.+?)\n',i)[0] for i in rhs2id]
        rhs2id = {a[1]:a[0] for a in rhs2id}

    with open(os.path.join(data_path, 'rel_id'), 'r') as fp:
        relation2id = fp.readlines()
        relation2id = [re.findall(r'(.+?)\t(.+?)\n',i)[0] for i in relation2id]
        rel2id = {a[1]:a[0] for a in relation2id}

    with open('../results/' + file_name + '.txt', 'r') as fp:
        res = fp.readlines()

    with open('../data/test_final.tsv', 'r') as test:
        title = test.readlines()

    new_res = []
    for item, resline in zip(title, res):
        item = item[:-1]
        new_resline = []
        resline = resline[:-1].split(' ')
        for idx, li in enumerate(resline):
            new_resline.append(rhs2id[li])
        new_resline = item + '\t' + '\t'.join(new_resline) + '\n'
        new_res.append(new_resline)

    # print(new_res)
    with open('../results/' + file_name + '.tsv', 'w') as fp:
        fp.writelines(new_res)
        
save_results(f'result_{args.model}', data_path = "../data/OpenBG-IMG/")