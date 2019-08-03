# -*- coding: utf-8 -*-

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Function
from collections import OrderedDict
import math
import numpy as np
from .basic_module import BasicModule
import matplotlib.pyplot as plt
from numba import jit
from .FCN import *
from .TCN import *
def softmin(zs, gamma):
    max_zs = torch.max(zs, dim=1, keepdim=True)[0]
    return ((max_zs + torch.log(torch.sum(torch.exp(zs - max_zs), dim=1, keepdim=True))) * -gamma).squeeze()

class DTWLoss(Function):
    @staticmethod
    def forward(ctx, D, Ma, gamma):
        N, M = D.shape[1:]
        ctx.gamma = gamma
        r = torch.zeros(D.shape).cuda()
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    r[:, i, j] = D[:, i, j]
                else:
                    zs = []
                    if i > 0 and Ma[i - 1, j]:
                        zs.append(r[:, i - 1, j])
                    if j > 0 and Ma[i, j - 1]:
                        zs.append(r[:, i, j - 1])
                    if i > 0 and j > 0 and Ma[i - 1, j - 1]:
                        zs.append(r[:, i - 1, j - 1])
                    zs = torch.stack(zs, dim=1) * -1 / gamma
                    r[:, i, j] = D[:, i, j] + softmin(zs, gamma)
        ctx.save_for_backward(D, Ma, r)
        return r[:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        D, Ma, r= ctx.saved_tensors
        gamma = ctx.gamma
        N, M = D.shape[1: ]
        e = torch.zeros(D.shape).cuda()
        e[:, -1, -1] = 1
        for i in range(N - 1, -1, -1):
            for j in range(M - 1, -1, -1):
                if j == M - 1 and i == N - 1:
                    continue
                if i + 1 < N and Ma[i + 1, j]:
                    e[:, i, j] += e[:, i + 1, j] * torch.exp((r[:, i + 1, j] - r[:, i, j] - D[:, i + 1, j]) / gamma)
                if j + 1 < M and Ma[i, j + 1]:
                    e[:, i, j] += e[:, i, j + 1] * torch.exp((r[:, i, j + 1] - r[:, i, j] - D[:, i, j + 1]) / gamma)
                if i + 1 < N and j + 1 < M and Ma[i + 1, j + 1]:
                    e[:, i, j] += e[:, i + 1, j + 1] * torch.exp((r[:, i + 1, j + 1] - r[:, i, j] - D[:, i + 1, j + 1]) / gamma)
        # print((grad_output.unsqueeze(1).unsqueeze(2) * e).shape, e.shape)
        return grad_output.unsqueeze(1).unsqueeze(2) * e, None, None

class SoftDTW_Model(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.dtwloss = DTWLoss().apply
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.gamma = params['gamma']
        self.C = params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)

    def metric(self, seqa, seqp, debug=False):
        T1, T2, C, gamma = seqa.shape[1], seqp.shape[1], self.C, self.gamma

        Ma = torch.ones(T1, T2)
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))
        
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap 
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap = (d_ap * align_ap).view(-1, T1, T2)

        d = self.dtwloss(d_ap, Ma, gamma)
        return d if debug == False else (d, d_ap, d_ap)

    def forward(self, seqa, seqp, seqn):
        model, gamma = self.model, self.gamma
        xa, seqa, _ = model(seqa)
        xp, seqp, _ = model(seqp)
        xn, seqn, _ = model(seqn)

        p_ap, p_an = torch.sigmoid(self.metric(seqa, seqp)), torch.sigmoid(-1 * self.metric(seqa, seqn))
        return torch.cat((p_ap, p_an), dim=0)



"""
_____________________________________Soft_Dtw_____________________________________

"""

@jit(nopython = True)
def compute_softdtw(D, gamma):
  N = D.shape[1]
  M = D.shape[2]
  batch_size = D.shape[0]
  R = np.zeros((batch_size,N + 2, M + 2)) + 1e8
  R[:,0, 0] = 0
  for b in range(batch_size):
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[b,i - 1, j - 1] / gamma
            r1 = -R[b,i - 1, j] / gamma
            r2 = -R[b,i, j - 1] / gamma
            rmax = max(max(r0, r1), r2)
            rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
            softmin = - gamma * (np.log(rsum) + rmax)
            R[b,i, j] = D[b,i - 1, j - 1] + softmin
  return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
  N = D_.shape[1]
  M = D_.shape[2]
  batch_size = D_.shape[0]
  D = np.zeros(( batch_size,N + 2, M + 2))
  E = np.zeros(( batch_size,N + 2, M + 2))
  D[:,1:N + 1, 1:M + 1] = D_
  E[:,-1, -1] = 1
  R[:,:, -1] = -1e8
  R[:,-1, :] = -1e8
  R[:,-1, -1] = R[-2, -2]
  for b in range(batch_size):
    for j in range(M, 0, -1):
        for i in range(N, 0, -1):
            a0 = (R[b,i + 1, j] - R[b,i, j] - D[b,i + 1, j]) / gamma
            b0 = (R[b,i, j + 1] - R[b,i, j] - D[b,i, j + 1]) / gamma
            c0 = (R[b,i + 1, j + 1] - R[b,i, j] - D[b,i + 1, j + 1]) / gamma
            a = np.exp(a0)
            t = np.exp(b0)
            c = np.exp(c0)
            E[b,i, j] = E[b,i + 1, j] * a + E[b,i, j + 1] * t + E[b,i + 1, j + 1] * c
  return E[:,1:N + 1, 1:M + 1]

class _SoftDTW(Function):
  @staticmethod
  def forward(ctx, D, gamma):
    dev = D.device
    dtype = D.dtype
    gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
    D_ = D.detach().cpu().numpy()
    g_ = gamma.item()
    softdtw = compute_softdtw(D_, g_)

    R = torch.Tensor(softdtw).to(dev).type(dtype)
    ctx.save_for_backward(D, R, gamma)
    
    return R[:,-2, -2]

  @staticmethod
  def backward(ctx, grad_output):
    dev = grad_output.device
    dtype = grad_output.dtype
    D, R, gamma = ctx.saved_tensors
    D_ = D.detach().cpu().numpy()
    R_ = R.detach().cpu().numpy()
    g_ = gamma.item()
    print(D_.shape,R_.shape,g_)
    r_ = compute_softdtw_backward(D_, R_, g_)
    
    E = torch.Tensor(r_).to(dev).type(dtype)
    return grad_output * E, None

        ## Added

        # def calc_distance_matrices(xb, yb):
        #     batch_size = xb.size(0)
        #     n = xb.size(1)
        #     m = yb.size(1)
        #     D = torch.zeros(batch_size, n, m)
        #     for i in range(batch_size):
        #         D[i] = calc_distance_matrix(xb[i], yb[i])
        #     return D
           

class SoftDTW(torch.nn.Module):
  def __init__(self, gamma=1.0, normalize=False):
    super(SoftDTW, self).__init__()
    self.normalize = normalize
    self.gamma=gamma
    self.func_dtw = _SoftDTW.apply

  def calc_distance_matrix(self, seqa, seqp):
    # n = x.size(0)
    # m = y.size(0)
    # d = x.size(1)
    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)
    # dist = torch.pow(x - y, 2).sum(2)
    T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]


    seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
    seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
    
                                                    
    d_ap = seqa - seqp
    d_ap = d_ap * d_ap
    d_ap = d_ap.sum(dim=1, keepdim=True)
    d_ap_s = d_ap
    d_ap = d_ap.view(-1, T1, T2)
        
    return d_ap

  def forward(self, x, y):
    if self.normalize:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      D_xx = self.calc_distance_matrix(x, x)
      out_xx = self.func_dtw(D_xx, self.gamma)
      D_yy = self.calc_distance_matrix(y, y)
      out_yy = self.func_dtw(D_yy, self.gamma)
      return out_xy - 1/2 * (out_xx + out_yy) # distance
    else:
      D_xy = self.calc_distance_matrix(x, y)
      out_xy = self.func_dtw(D_xy, self.gamma)
      return out_xy # discrepancy
class NeuralDTW_SoftDtw(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.dtwloss = SoftDTW(gamma=1.0, normalize=True)

        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqb1, seqb2, seqb3 = self.model(seqb)
        p_a1 = self.dtwloss(seqa1,seqb2).unsqueeze(1)
      
        p_a2 = self.dtwloss(seqa1,seqb2).unsqueeze(1)
        p_a3 = self.dtwloss(seqa1,seqb2).unsqueeze(1)


        p_a = torch.cat((p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

"""
_____________________________________NeuralDTW_____________________________________

"""
class NeuralDTW(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqb1, seqb2, seqb3 = self.model(seqb)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)


        p_a = torch.cat((p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

class CQTSPPNet_blstm_seq_withouttpp(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 3), stride=(1, 2), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        # self.spp = SPP([8,4,2,1])
        self.T, self.C = params['T'], params['C']
        # self.fc0 = nn.Linear(self.C, 300)
        # self.fc1 = nn.Linear(300, 10000)
        #self.lstm = nn.LSTM(512, 256, bidirectional=True)
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1]
        seq = x
        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        seq2, _ = self.lstm2(seq1)
        seq3, _ = self.lstm3(seq2)
        seq1 = seq1.permute(1, 0, 2) #N, 16, C
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)

        return seq1,seq2,seq3

class CQTSPPNet_blstm_seq4_withouttpp(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.T, self.C = params['T'], params['C']
        self.fc0 = nn.Linear(self.C, 300)
        self.fc1 = nn.Linear(300, 10000)
        self.lstm = nn.LSTM(128, 64, bidirectional=True)

    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        seq = x

        N, C, H, W = x.size()
        maxpool = nn.AdaptiveMaxPool2d((H, 1))
        x = maxpool(x).view(-1, self.C) # N, C * 16

        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        seq, _ = self.lstm(seq)
        seq = seq.permute(1, 0, 2) #N, 16, C

        x = self.fc0(x)
        fea = x
        x = self.fc1(x)
        return x, seq, fea

"""
_____________________________________NeuralDTW_WithOutSF_____________________________
这里希望测试只加一层LSTM的效果
"""
class NeuralDTW_WithOutSF(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.T, self.C = params['T'], params['C']
        self.batch_size = params['batch_size']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        #align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))

        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
    
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)
        d_ap = d_ap.view(-1, T1, T2)
        #align_ap = align_ap.view(-1, T1, T2)
        

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap* Ma.cuda(torch.cuda.current_device())
        
        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap)

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqb1, seqb2, seqb3 = self.model(seqb)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        # p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)


        # p_a = torch.cat((p_a1,p_a2,p_a3),1)
        # p_a = torch.sigmoid(self.fc3(p_a))
        return p_a1

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

"""
_____________________________________ NeuralDTW_Milti_Metix_____________________________
这里希望测试对于不同的LSTM分别使用不同的网络计算Metrix的效果
"""

class NeuralDTW_Milti_Metix(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc1_metric2 = nn.Linear(self.C,self.C)
        self.fc2_metric2 = nn.Linear(self.C, 1)
        self.fc1_metric3 = nn.Linear(self.C,self.C)
        self.fc2_metric3 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def metric2(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2_metric2(torch.sigmoid(self.fc1_metric2(seqa) + self.fc1_metric2(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def metric3(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2_metric3(torch.sigmoid(self.fc1_metric3(seqa) + self.fc1_metric3(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)
    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqb1, seqb2, seqb3 = self.model(seqb)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric2(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric3(seqa3 , seqb3).unsqueeze(1)


        p_a = torch.cat((p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

"""
_____________________________________NeuralDTW_LSTMRes_____________________________
这里希望测试对于不同的LSTM输出到Neural里的结果相加的效果：
将第一层的结果加到第二层再输入到Metrix
将第二层的结果加到第三层再输入到Metrix
        seqa2 = seqa2+seqa1
        seqa3 = seqa3+seqa2
"""
class NeuralDTW_LSTMRes(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqa2 = seqa2+seqa1
        seqa3 = seqa3+seqa2
        seqb1, seqb2, seqb3 = self.model(seqb)
        seqb2 = seqb2+seqb1
        seqb3 = seqb2+seqb3
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

"""
_____________________________________NeuralDTW_withOutLSTM_____________________________
经过实验发现多层的LSTM并不能将不同元素之间的时序信息记录，且增加LSTM后发现两个序列不同元素之间的
差异更加模糊，因此希望测试去掉LSTM的效果
结论：训练20000组效果不好，而后可测试其他效果
"""
class CQTSPPNet_blstm_seq_withouttpp_res(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 3), stride=(1, 2), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.T, self.C = params['T'], params['C']
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
    def forward(self, x):
        
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1]
        seq = x
        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        seq2, _ = self.lstm2(seq1)
        seq3, _ = self.lstm3(seq2)
        seq  = seq.permute(1, 0, 2)
        seq1 = seq1.permute(1, 0, 2) #N, 16, C
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)

        return seq,seq1,seq2,seq3


class NeuralDTW_withOutLSTM(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)

        seqa2 = seqa2+seqa1
        seqa3 = seqa3+seqa2
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        seqb2 = seqb2+seqb1
        seqb3 = seqb2+seqb3
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        # p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        # p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        # p_a = torch.cat((p_a1,p_a2,p_a3),1)
        # p_a = torch.sigmoid(self.fc3(p_a))
        return p_a0

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

"""
_____________________________________NeuralDTW_Milti_Metix_res_____________________________
在NeuralDTW_Milti_Metix修改：原因是因为在NeuralDTW_Milti_Metix对第二层LSTM的输出计算Metrix的结果非常小。
经过可视化发现第二层LSTM的输出是非常小的结果，此外发现即使经过多层LSTM，对于信息的提取没有达到想要的结果，
输出变化也不大。因此借鉴ResNet，在第一层LSTM输出结果后将第一层输出的结果与第一层的输入相加，作为第二层LSTM的输入。
将第二层的结果与第二层的输入相加作为第三层的输入。
best model path: 0623_02:32:16.pth
coversong80 result:map:0.7005135062030867 top1:0.083125 7.84375
"""
class NeuralDTW_Milti_Metix_res(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttp_res_1(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc1_metric2 = nn.Linear(self.C,self.C)
        self.fc2_metric2 = nn.Linear(self.C, 1)
        self.fc1_metric3 = nn.Linear(self.C,self.C)
        self.fc2_metric3 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def metric2(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2_metric2(F.relu(self.fc1_metric2(seqa) + self.fc1_metric2(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def metric3(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2_metric3(F.relu(self.fc1_metric3(seqa) + self.fc1_metric3(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)
    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqb1, seqb2, seqb3 = self.model(seqb)
    
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric2(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric3(seqa3 , seqb3).unsqueeze(1)


        p_a = torch.cat((p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

class CQTSPPNet_blstm_seq_withouttp_res_1(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 3), stride=(1, 2), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.T, self.C = params['T'], params['C']
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
    def forward(self, x):
        
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1]
        seq = x
        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        seq1_ = seq1 + seq
        seq2, _ = self.lstm2(seq1_)
        seq2_ = seq2 + seq
        seq3, _ = self.lstm3(seq2_)
        seq  = seq.permute(1, 0, 2)
        seq1 = seq1.permute(1, 0, 2) #N, 1  6, C
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)
        
        return seq1,seq2,seq3


"""
_____________________________________NeuralDTW_withMultiInformation_____________________________
测试使用四层LSTM信息输入的效果
基于原来的Neural的模型，如果效果好可以使用
"""


class NeuralDTW_withMultiInformation(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(4,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)


"""
_____________________________________NeuralDTW_with3_seq_____________________________
在Mertic计算相似度函数的时候输入三个邻近的段作为中间段的相似性匹配结果
    cat(seqa(i-1),seqa(i),seqa(i+1))
    cat(seqb(i-1),seqb(i),seqb(i+1))
"""
class NeuralDTW_with3_seq(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C*3, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(4,1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma
        seqa_re,seqp_re = seqa,seqp
        seqa
        seqa_np,seqp_np = seqa.data.cpu().numpy(), seqp.data.cpu().numpy()

        seqa_front = np.insert(seqa_np,0,np.zeros(self.C),1)
        seqa_front = np.delete(seqa_front,T1,1)
        seqa_front = torch.from_numpy(seqa_front).cuda()
        seqa_behind = np.delete(seqa_np,0,1)
        seqa_behind = np.insert(seqa_behind,T1-1,np.zeros(self.C),1)
        seqa_behind =torch.from_numpy(seqa_behind).cuda()
        seqa = torch.cat((seqa,seqa_behind),2).cuda()
        seqa = torch.cat((seqa_front,seqa),2).cuda()

        seqp_front = np.insert(seqp_np,0,np.zeros(self.C),1)
        seqp_front = np.delete(seqp_front,T2,1)
        seqp_front = torch.from_numpy(seqp_front).cuda()
        seqp_behind = np.delete(seqp_np,0,1)
        seqp_behind = np.insert(seqp_behind,T2-1,np.zeros(self.C),1)
        seqp_behind =torch.from_numpy(seqp_behind).cuda()
        seqp = torch.cat((seqp,seqp_behind),2).cuda()
        seqp = torch.cat((seqp_front,seqp),2).cuda()
        
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C*3), seqp.view(-1, C*3)
        seqa_re, seqp_re = seqa_re.repeat(1, 1, T2), seqp_re.repeat(1, T1, 1)
        seqa_re, seqp_re = seqa_re.view(-1, C), seqp_re.view(-1, C)
    
                
        align_ap = torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(seqa) + self.fc1(seqp))))
                                                    
        d_ap = seqa_re - seqp_re
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = (d_ap * align_ap).view(-1, T1, T2)
        align_ap = align_ap.view(-1, T1, T2)
        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            d_ap = d_ap * Ma

        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        p_a = torch.sigmoid(self.fc3(p_a))
        return p_a

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

"""
_____________________________________NeuralDTW_FCN_Mask_____________________________
在Mertic计算相似度函数的时候使用全卷积网络将相似度矩阵作为输入，得到与相似度矩阵同样大小的输出
在于相似度矩阵相乘，得到最终的结果，然后将对角线上的值相加，
"""
class NeuralDTW_FCN_Mask(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp(params)
        self.T, self.C = params['T'], params['C']
        self.fc1 = nn.Linear(self.C, self.C)
        self.fc2 = nn.Linear(self.C, 1)
        self.fc3 = nn.Linear(3,1)
        self.fcn_model = FCN16s(pretrained_net=VGGNet(requires_grad=True, show_params=False), n_class=1)
        self.Ma = None
        if params['mask'] != -1:
            Ma = torch.zeros((1, self.T, self.T)).cuda()
            for i in range(self.T):
                for j in range(i - params['mask'], i + params['mask'] + 1):
                    if j >= 0 and j < self.T:
                        Ma[0, i, j] = 1
            Ma /= torch.sum(Ma)
            self.Ma = Ma

    def metric(self, seqa, seqp, debug=False):
        # return a similarity
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        Ma = self.Ma

        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        
    
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap = d_ap.view(-1, 1,T1, T2)
        align_ap = self.fcn_model(d_ap)
       
        d_ap = d_ap*align_ap

        d_ap_s = d_ap_s.view(-1, T1, T2)
        # align_ap = torch.sigmoid(self.fc2(F.relu(self.fc1(seqa.unsqueeze(1)) + self.fc1(seqp.unsqueeze(0)))))
        # d_ap = seqa.unsqueeze(1) - seqp.unsqueeze(0)
        # d_ap = d_ap * d_ap
        # d_ap = d_ap.sum(dim=1, keepdim=True)
        # d_ap = (d_ap * align_ap).view(-1, T1, T2)

        if Ma is not None:
            # print(d_ap.shape, Ma.shape)
            b_d_ap = d_ap
            #d_ap = d_ap * Ma
        d_ap = d_ap.squeeze(1)
        align_ap =align_ap.squeeze(1)
        b_d_ap = b_d_ap.squeeze()
        s_ap = torch.exp(-1 * torch.sum(d_ap, dim=[1, 2]))
        
        return s_ap if debug == False else (s_ap, d_ap, b_d_ap,align_ap,d_ap_s)

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3 = self.model(seqa)
        seqb1, seqb2, seqb3 = self.model(seqb)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        # p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        # p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)


        # p_a = torch.cat((p_a1,p_a2,p_a3),1)
        # p_a = torch.sigmoid(self.fc3(p_a))
        return p_a1

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
"""
_____________________________________NeuralDTW_CNN_Mask_____________________________
在Mertic计算相似度函数的时候使用VGG卷积网络将相似度矩阵作为输入，对相似度矩阵图像进行分类得到最
终的结果
train_loss: 0.009155398036803429
                         0.918582975857165 0.1924 2.228
350it [00:00, 487.55it/s]0.8445498201282954 0.09 4.24375
model name 0701_07:20:37.pth
train_loss: 0.01003534658458084
                         0.8939061249925756 0.1888 2.624
350it [00:00, 397.09it/s]0.8682273521160873 0.09125 3.575
model name 0630_07:59:56.pth
train_loss: 0.010032127343475198
                         0.9082044381743016 0.1908 2.352
350it [00:00, 377.23it/s]0.8580150749602462 0.090625 3.8
"""


class NeuralDTW_CNN_Mask(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.fc =nn.Linear(6400,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil
    
    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
class NeuralDTW_CNN_Mask_300(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg4')
        self.fc =nn.Linear(2304,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil
    
    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

class NeuralDTW_CNN_Mask_VGG16(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False, model='vgg16')
        self.fc =nn.Linear(6400,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))
        
        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

class CQTSPPNet_blstm_seq_VGG(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                stride=(1, 1), bias=False)),
            ('norm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3), stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.T, self.C = params['T'], params['C']
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
    def forward(self, x):
        
        x = self.features(x)  # [N, 256, 1, 87]
        x = self.conv(x)  # [N, 256, 1, 85]
        seq = x
        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        seq2, _ = self.lstm2(seq1)
        seq3, _ = self.lstm3(seq2)
        seq  = seq.permute(1, 0, 2)
        seq1 = seq1.permute(1, 0, 2) #N, 16, T=85
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)

        return seq,seq1,seq2,seq3
class NeuralDTW_CNN_Mask_VGG_T85(BasicModule):
    """
    存在梯度消失问题
    考虑使用ResNet  
    """
    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_VGG(params)
        self.T, self.C = params['T'], params['C']
        self.resnet= ResNet(4,1)

    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        
        samil   = self.resnet(p_a)
        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

class CQTSPPNet_seq_dilation(BasicModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation = (1,1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation = (1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),

            #('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

                
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation = (1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None)))
        ]))
    def forward(self, x):
        x = self.features(x)  # [N, 512, 1, 86]
       
        x = x.squeeze(dim=2).permute(0, 2, 1)
        return x
"""
NeuralDTW_CNN_Mask_dilation:
    train_loss: 0.008146948917397142
    Youtube350: 0.9482116911103371 0.1944 1.956
    CoverSong80:0.8785660490623716 0.09125 4.66875
    0704_06:40:41.pth
"""
class NeuralDTW_CNN_Mask_dilation(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation()
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg4')
        self.fc =nn.Linear(4096,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        p_a = self.metric(seqa , seqb).unsqueeze(1)
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)


class CQTSPPNet_seq_dilation1(BasicModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation = (1,1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation = (1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),

            ('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

                
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation = (1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None)))
        ]))
    def forward(self, x):
        x = self.features(x)  # [N, 512, 1, 38]
        x = x.squeeze(dim=2).permute(0, 2, 1)
        return x
class NeuralDTW_CNN_Mask_dilation1(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation()
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.fc =nn.Linear(2048,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa = self.model(seqa)
        seqb = self.model(seqb)
        p_a = self.metric(seqa , seqb).unsqueeze(1)
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x4'].view(VGG_out['x4'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)

class CQTSPPNet_blstm_seq_withouttpp_res2(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 3), stride=(1, 2), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.T, self.C = params['T'], params['C']
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        
    def forward(self, x):
        
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1]
        seq = x
        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        
        seq  = seq.permute(1, 0, 2)
        seq1 = seq1.permute(1, 0, 2) #N, 16, C
        
        return seq,seq1
class NeuralDTW_CNN_Mask2lstm(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res2(params)
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.fc =nn.Linear(6400,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1 = self.model(seqa)
        seqb0,seqb1 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1),1)
        
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil
    
    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)


class CQTSPPNet_blstm_seq_withouttpp_spp(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 3), stride=(1, 2), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        self.T, self.C = params['T'], params['C']
        self.SPPooling = SpatialPyramidPooling2d(9)
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
    def forward(self, x):
        
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1]
        seq = x.squeeze(2)
        seq = seq.permute(2, 0, 1) #16, N, C
        
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        seq2, _ = self.lstm2(seq1)
        seq3, _ = self.lstm3(seq2)
        seq  = seq.permute(1, 0, 2)
        seq1 = seq1.permute(1, 0, 2) #N, 16, C
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)

        return seq,seq1,seq2,seq3
class CQTSPPNet_blstm_seq_withouttpp_adaptive(BasicModule):
    def __init__(self, params):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(36, 40),
                                stride=(1, 1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(12, 3),
                                stride=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 2), bias=False)),
            ('norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1, None))),
        ]))
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(1, 3), stride=(1, 2), bias=False)),
            ('norm0', nn.BatchNorm2d(512)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool1',nn.AdaptiveMaxPool2d((None, 20)))
        ]))
        self.T, self.C = params['T'], params['C']
        
        self.lstm1 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, bidirectional=True)
    def forward(self, x):
        
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x = self.conv(x)  # [N, 256, 1, W - 75 +1 - 3 + 1] 
         
        seq = x
        seq = seq.squeeze(dim=2).permute(2, 0, 1) #16, N, C
        #seq, _ = self.lstm(seq)
        seq1, _ = self.lstm1(seq)
        seq2, _ = self.lstm2(seq1)
        seq3, _ = self.lstm3(seq2)
        seq  = seq.permute(1, 0, 2)
        seq1 = seq1.permute(1, 0, 2) #N, 16, C
        seq2 = seq2.permute(1, 0, 2)
        seq3 = seq3.permute(1, 0, 2)

        return seq,seq1,seq2,seq3        
class NeuralDTW_CNN_Mask_spp(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_adaptive(params)
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.fc =nn.Linear(1024,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))
        
        return samil
    
    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:,:,:, 0]
        x2 = x[:,:,:, 1]
        x3 = x[:,:,:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=2)
        return x
class NeuralDTW_CNN_Mask_inhance(BasicModule):

    def __init__(self, params):
        super().__init__()
        self.model = CQTSPPNet_blstm_seq_withouttpp_res(params)
        self.T, self.C = params['T'], params['C']
        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.fc =nn.Linear(6400,1)
        inhance_kernel= np.array([[ 0, 0, 0, 0,-1],
                               [ 0, 0, 0,-1, 0],
                               [ 0, 0, 4, 0, 0],
                               [ 0,-1, 0, 0, 0],
                               [-1, 0, 0, 0, 0]])
        self.inhancemetrix = torch.autograd.Variable\
                    (torch.from_numpy(inhance_kernel)\
                    .unsqueeze(0).unsqueeze(1).float(),requires_grad=False )
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], self.C
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)

        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        d_ap_s = F.conv2d(d_ap_s.unsqueeze(1),self.inhancemetrix.cuda())
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa0,seqa1, seqa2, seqa3 = self.model(seqa)
        seqb0,seqb1, seqb2, seqb3 = self.model(seqb)
        p_a0 = self.metric(seqa0 , seqb0)
        p_a1 = self.metric(seqa1 , seqb1)
        p_a2 = self.metric(seqa2 , seqb2)
        p_a3 = self.metric(seqa3 , seqb3)
        p_a = torch.cat((p_a0,p_a1,p_a2,p_a3),1)
        
        VGG_out = self.VGG_Conv(p_a)
        
        VGG_out = VGG_out['x3'].view(VGG_out['x3'].shape[0],-1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil
    
    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
class CQTSPPNet_seq_dilation_SPP(BasicModule):
    def __init__(self):
        super().__init__()
        self.features1 = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(12, 3),
                                dilation = (1,1), bias=False)),
            ('norm0', nn.BatchNorm2d(32)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(13, 3),
                                dilation = (1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True))
            ]))
            #('maxpool0',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),

        self.features2 = nn.Sequential(OrderedDict([        
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(13, 3),
                                dilation = (1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),

            ('maxpool1',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),
        ]))
        self.features3 = nn.Sequential(OrderedDict([    
            ('conv4', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)),
            ('relu5', nn.ReLU(inplace=True)),

            ('maxpool2',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),
        ]))
        self.features4 = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)),
            ('relu7', nn.ReLU(inplace=True)),
            ('maxpool3',nn.MaxPool2d(kernel_size = (1,2),stride=(1,2))),
        ]))
        self.features5 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                dilation = (1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3),
                                dilation = (1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),
            ('pool0', nn.AdaptiveMaxPool2d((1,None)))
        ]))
    def forward(self, x):
        x1= self.features1(x)  # [N, 512, 1, 86]
        x2= self.features2(x1)
        x3= self.features3(x2)
        x4= self.features4(x3)
        x5= self.features5(x4)
        
        x1 = nn.AdaptiveMaxPool2d((1,None))(x1).squeeze(dim=2).permute(0, 2, 1)
        x2 = nn.AdaptiveMaxPool2d((1,None))(x2).squeeze(dim=2).permute(0, 2, 1)
        x3 = nn.AdaptiveMaxPool2d((1,None))(x3).squeeze(dim=2).permute(0, 2, 1)
        x4 = nn.AdaptiveMaxPool2d((1,None))(x4).squeeze(dim=2).permute(0, 2, 1)
        x5 = x5.squeeze(dim=2).permute(0, 2, 1)
        return x2,x3,x4,x5
class NeuralDTW_CNN_Mask_dilation_SPP(BasicModule):
    def __init__(self, params ):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP()
       
        self.VGG_Conv1 = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.VGG_Conv2 = VGGNet(requires_grad=True, show_params=False,model='vgg11')
        self.VGG_Conv3 = VGGNet(requires_grad=True, show_params=False,model='vgg4')
        self.fc =nn.Linear(18944,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3, seqa4, = self.model(seqa)
        seqb1, seqb2,seqb3,seqb4 = self.model(seqb)
        #p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4 , seqb4).unsqueeze(1)
        #p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])  
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0],-1)
        VGG_out2 = self.VGG_Conv2(p_a3) 
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0],-1)
        VGG_out3 = self.VGG_Conv3(p_a4) 
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0],-1)
        VGG_out = torch.cat((VGG_out1,VGG_out2,VGG_out3),1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
class NeuralDTW_CNN_Mask_dilation_SPP2(BasicModule):
    """
    0.0065394770698876045
    Youtube350:
                         0.9423899344521117 0.1928 2.008
    CoverSong80:
                        0.8930383739411555 0.09125 5.43125
    model name 0709_00:31:23.pth

    train_loss: 0.007253749475885513
    Youtube350:
                         0.9580713600534396 0.194 2.132
    CoverSong80:
                         0.8858273129993087 0.093125 3.9
    model name 0707_18:23:53.pth
    """
    def __init__(self, params ):
        super().__init__()
        self.model = CQTSPPNet_seq_dilation_SPP()
       
        self.VGG_Conv1 = VGGNet(requires_grad=True, in_channels =1,show_params=False,model='vgg11')
        
        self.fc =nn.Linear(18944,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqp.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)  
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return  d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self,seqa,seqb):
        seqa1, seqa2, seqa3, seqa4, = self.model(seqa)
        seqb1, seqb2,seqb3,seqb4 = self.model(seqb)
        #p_a1 = self.metric(seqa1 , seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2 , seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3 , seqb3).unsqueeze(1)
        p_a4 = self.metric(seqa4 , seqb4).unsqueeze(1)
        #p_a = torch.cat((p_a2,p_a3,p_a4),3)
        # torch.Size([1, 1, 84, 400])
        # torch.Size([1, 194, 64])
        # torch.Size([1, 94, 128])  
        # torch.Size([1, 44, 256])
        # torch.Size([1, 38, 512])
        
        VGG_out1 = self.VGG_Conv1(p_a2)
        VGG_out1 = VGG_out1['x4'].view(VGG_out1['x4'].shape[0],-1)
        VGG_out2 = self.VGG_Conv1(p_a3) 
        VGG_out2 = VGG_out2['x4'].view(VGG_out2['x4'].shape[0],-1)
        VGG_out3 = self.VGG_Conv1(p_a4) 
        VGG_out3 = VGG_out3['x3'].view(VGG_out3['x3'].shape[0],-1)
        VGG_out = torch.cat((VGG_out1,VGG_out2,VGG_out3),1)
        samil   = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
"____________________________________NeuralDTW_TCN__________________________________"


class CQTSPPNet_TCN(BasicModule):
    def __init__(self):
        super().__init__()
        self.features =TemporalConvNet(num_inputs=84, num_channels=[128,256,256,512], kernel_size=2, dropout=0.2)

    def forward(self, x):
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        seq = x.permute(0, 2, 1)
        # seq1 = seq1.permute(1, 0, 2)  # N, 16, C
        # seq2 = seq2.permute(1, 0, 2)
        # seq3 = seq3.permute(1, 0, 2)
        return seq


class NeuralDTW_TCN(BasicModule):

    def __init__(self , params ):
        super().__init__()
        self.model =CQTSPPNet_TCN()

        self.VGG_Conv = VGGNet(requires_grad=True, show_params=False, model='vgg11')
        self.fc = nn.Linear(73728, 1)

    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqa.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        # seqa0, seqa1, seqa2, seqa3 = self.model(seqa)
        # seqb0, seqb1, seqb2, seqb3 = self.model(seqb)
        # p_a0 = self.metric(seqa0, seqb0).unsqueeze(1)
        # p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        # p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        # p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        # p_a = torch.cat((p_a0, p_a1, p_a2, p_a3), 1)
       # print(seqa.shape)
        seqa = self.model(seqa.squeeze(1))
        seqb = self.model(seqb.squeeze(1))
        p_a = self.metric(seqa,seqb).unsqueeze(1)
        VGG_out = self.VGG_Conv(p_a)
        VGG_out = VGG_out['x5'].view(VGG_out['x5'].shape[0], -1)
        samil = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        seqa,seqp,seqn = seqa.squeeze(1),seqp.squeeze(1),seqn.squeeze(1)
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
"____________________________________NeuralDTW_TCN_SPP__________________________________"
class CQTSPPNet_TCN_SPP(BasicModule):
    def __init__(self):
        super().__init__()
        self.features =TemporalConvNet_SPP(num_inputs=84, num_channels=[128,256,256,512], kernel_size=2, dropout=0.2)

    def forward(self, x):
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x1,x2,x3 = x[2],x[3],x[4]
        seq1 = x1.permute(0, 2, 1)
        seq2 = x2.permute(0, 2, 1)
        seq3 = x3.permute(0, 2, 1)
        # seq1 = seq1.permute(1, 0, 2)  # N, 16, C
        # seq2 = seq2.permute(1, 0, 2)
        # seq3 = seq3.permute(1, 0, 2)
        return seq1,seq2,seq3


class NeuralDTW_TCN_SPP(BasicModule):

    def __init__(self  , params  ):
        super().__init__()
        self.model =CQTSPPNet_TCN_SPP()

        self.VGG_Conv = VGGNet(requires_grad=True,in_channels=3 ,show_params=False, model='vgg11')
        self.fc = nn.Linear(73728, 1)

    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqa.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3 = self.model(seqa.squeeze(1))
        seqb1, seqb2, seqb3 = self.model(seqb.squeeze(1) )
        # p_a0 = self.metric(seqa0, seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        p_a = torch.cat((p_a1, p_a2, p_a3), 1)
       # print(seqa.shape)
       #  seqa1, = self.model(seqa.squeeze(1))
       #  seqb = self.model(seqb.squeeze(1))
        #p_a = self.metric(seqa,seqb).unsqueeze(1)
        VGG_out = self.VGG_Conv(p_a)
        VGG_out = VGG_out['x5'].view(VGG_out['x5'].shape[0], -1)
        samil = torch.sigmoid(self.fc(VGG_out))

        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        seqa,seqp,seqn = seqa.squeeze(1),seqp.squeeze(1),seqn.squeeze(1)
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
class NeuralDTW_TCN_SPP_2(BasicModule):

    def __init__(self  , params  ):
        super().__init__()
        self.model =CQTSPPNet_TCN_SPP()

        self.VGG_Conv1 = VGGNet(requires_grad=True,in_channels=1 ,show_params=False, model='vgg11')
        self.VGG_Conv2 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.VGG_Conv3 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.fc1 = nn.Linear(73728, 1)
        self.fc2 = nn.Linear(73728, 1)
        self.fc3 = nn.Linear(73728, 1)
        self.f=nn.Linear(3,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqa.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3 = self.model(seqa.squeeze(1))
        seqb1, seqb2, seqb3 = self.model(seqb.squeeze(1) )
        # p_a0 = self.metric(seqa0, seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        #p_a = torch.cat((p_a1, p_a2, p_a3), 1)
       # print(seqa.shape)
       #  seqa1, = self.model(seqa.squeeze(1))
       #  seqb = self.model(seqb.squeeze(1))
        #p_a = self.metric(seqa,seqb).unsqueeze(1)
        VGG_out1 = self.VGG_Conv1(p_a1)
        VGG_out1 = VGG_out1['x5'].view(VGG_out1['x5'].shape[0], -1)
        VGG_out2 = self.VGG_Conv2(p_a2)
        VGG_out2 = VGG_out2['x5'].view(VGG_out2['x5'].shape[0], -1)
        VGG_out3 = self.VGG_Conv3(p_a3)
        VGG_out3 = VGG_out3['x5'].view(VGG_out3['x5'].shape[0], -1)

        samil1 = torch.sigmoid(self.fc1(VGG_out1))
        samil2 = torch.sigmoid(self.fc2(VGG_out2))
        samil3 = torch.sigmoid(self.fc3(VGG_out3))
        samil = torch.sigmoid(self.f(torch.cat((samil1, samil2, samil3), 1)))
        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        seqa,seqp,seqn = seqa.squeeze(1),seqp.squeeze(1),seqn.squeeze(1)
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
class CQTSPPNet_TCN_SPP_2(BasicModule):
    def __init__(self):
        super().__init__()
        self.features =TemporalConvNet_SPP(num_inputs=84, num_channels=[128,256,256,512,512], kernel_size=2, dropout=0.2)

    def forward(self, x):
        x = self.features(x)  # [N, 128, 1, W - 75 + 1]
        x1,x2,x3 = x[3],x[4],x[5]
        seq1 = x1.permute(0, 2, 1)
        seq2 = x2.permute(0, 2, 1)
        seq3 = x3.permute(0, 2, 1)
        # seq1 = seq1.permute(1, 0, 2)  # N, 16, C
        # seq2 = seq2.permute(1, 0, 2)
        # seq3 = seq3.permute(1, 0, 2)
        return seq1,seq2,seq3
class NeuralDTW_TCN_SPP_3(BasicModule):

    def __init__(self  , params  ):
        super().__init__()
        self.model =CQTSPPNet_TCN_SPP_2()

        self.VGG_Conv1 = VGGNet(requires_grad=True,in_channels=1 ,show_params=False, model='vgg11')
        self.VGG_Conv2 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.VGG_Conv3 = VGGNet(requires_grad=True, in_channels=1, show_params=False, model='vgg11')
        self.fc1 = nn.Linear(73728, 1)
        self.fc2 = nn.Linear(73728, 1)
        self.fc3 = nn.Linear(73728, 1)
        self.f=nn.Linear(3,1)
    def metric(self, seqa, seqp, debug=False):
        T1, T2, C = seqa.shape[1], seqp.shape[1], seqa.shape[2]
        seqa, seqp = seqa.repeat(1, 1, T2), seqp.repeat(1, T1, 1)
        seqa, seqp = seqa.view(-1, C), seqp.view(-1, C)
        d_ap = seqa - seqp
        d_ap = d_ap * d_ap
        d_ap = d_ap.sum(dim=1, keepdim=True)
        d_ap_s = d_ap
        d_ap_s = d_ap_s.view(-1, T1, T2)
        return d_ap_s if debug == False else d_ap_s

    def multi_compute_s(self, seqa, seqb):
        seqa1, seqa2, seqa3 = self.model(seqa.squeeze(1))
        seqb1, seqb2, seqb3 = self.model(seqb.squeeze(1) )
        # p_a0 = self.metric(seqa0, seqb0).unsqueeze(1)
        p_a1 = self.metric(seqa1, seqb1).unsqueeze(1)
        p_a2 = self.metric(seqa2, seqb2).unsqueeze(1)
        p_a3 = self.metric(seqa3, seqb3).unsqueeze(1)
        #p_a = torch.cat((p_a1, p_a2, p_a3), 1)
       # print(seqa.shape)
       #  seqa1, = self.model(seqa.squeeze(1))
       #  seqb = self.model(seqb.squeeze(1))
        #p_a = self.metric(seqa,seqb).unsqueeze(1)
        VGG_out1 = self.VGG_Conv1(p_a1)
        VGG_out1 = VGG_out1['x5'].view(VGG_out1['x5'].shape[0], -1)
        VGG_out2 = self.VGG_Conv2(p_a2)
        VGG_out2 = VGG_out2['x5'].view(VGG_out2['x5'].shape[0], -1)
        VGG_out3 = self.VGG_Conv3(p_a3)
        VGG_out3 = VGG_out3['x5'].view(VGG_out3['x5'].shape[0], -1)

        samil1 = torch.sigmoid(self.fc1(VGG_out1))
        samil2 = torch.sigmoid(self.fc2(VGG_out2))
        samil3 = torch.sigmoid(self.fc3(VGG_out3))
        samil = torch.sigmoid(self.f(torch.cat((samil1, samil2, samil3), 1)))
        return samil

    def forward(self, seqa, seqp, seqn):
        model = self.model
        seqa,seqp,seqn = seqa.squeeze(1),seqp.squeeze(1),seqn.squeeze(1)
        p_ap, p_an = self.multi_compute_s(seqa, seqp), self.multi_compute_s(seqa, seqn)
        return torch.cat((p_ap, p_an), dim=0)
if __name__=='__main__':
    # print(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(13, 3),
    #                             dilation = (1, 1), bias=False)(torch.rand([1,1,84,400])).shape)
    # y = CQTSPPNet_seq_dilation()
    # print(y(torch.rand([1,1,84,400])).shape)
    # s = NeuralDTW_CNN_Mask_dilation_SPP()
    # s(torch.randn([1,1,84,400]),torch.randn([1,1,84,400]),torch.randn([1,1,84,400]))
    x = NeuralDTW_TCN_SPP()
    print(x(torch.randn([1,1,84,400]),torch.randn([1,1,84,400]),torch.randn([1,1,84,400])))
    # x = SPPNet(num_level=5)
    # print(x(torch.rand([1,1,512,38])))