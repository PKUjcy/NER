from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect


def cut_data(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            offset = np.random.randint(max_offset)
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 100:
        offset = 100 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data

class WeightRandom(object):
    def __init__(self, dic):
        weights = [w for _,w in dic.items()]
        self.goods = [x for (i,(x,_)) in enumerate(dic.items())]
        self.total = sum(weights)
        self.acc = list(self.accumulate(weights))
    def accumulate(self, weights):#累和.如accumulate([10,40,50])->[10,50,100]
        cur = 0
        for w in weights:
            cur = cur+w
            yield cur
    def __call__(self):
        return self.goods[bisect.bisect_right(self.acc , random.uniform(0, self.total))]



class dataloader():
    def __init__(self, out_length = 600):
        self.out_length = out_length
        self.indir = 'data/youtube_cqt_npy/'
        self.dic = np.load('hpcp/dict_cqt_id_list5+.npy').item()
        self.dic_num = np.load('hpcp/dict_cqt_id_num5+.npy').item()
        
        self.wr = WeightRandom(self.dic_num)
        
    def get(self, num=10):
        # 返回num个数据
        label_list = []
        data_list = []
        for i in range(num):
            index = self.wr() #随机选取10个id
            #print(index)
            path_list = random.sample(self.dic[index],5)
            #path_list = random.sample(self.all_list[index],5)
            for path in path_list:
                data = np.load(self.indir+path+'.npy')
                data = data.T
                data = cut_data(data, self.out_length)
                #data = data[:,0:-13]
                data_list.append(data)
            label = torch.Tensor([i])
            label_list = label_list + [label,label,label,label,label]
            #path_list += random.sample(self.dic_train[index],5) #每个id随机抽取5首歌
        #data_list, label_list = load_2dfm(path_list)
        #print(path_list)
        return data_list, label_list





class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        #print(dist[0])
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(sorted(dist[i][mask[i]])[-3])
            dist_an.append(sorted(dist[i][mask[i] == 0])[5])
        dist_ap = torch.stack((dist_ap))
        dist_an = torch.stack((dist_an))
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() 
        return loss, prec
    