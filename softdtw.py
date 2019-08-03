from hpcp_loader_for_softdtw import *
from torch.utils.data import DataLoader
import models
from config import DefaultConfig, opt
from tqdm import tqdm
import torch
from utility import *
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
import os
import pandas as pd
import seaborn as sns
import resource
import numpy as np
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
torch.backends.cudnn.benchmark =True #cudnn有很多种并行计算卷积的算法，
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
import visdom

def visualize(softdtw,dataloader):
    # softdtw.eval()
    # softdtw.model.eval()
    softdtw.eval()
    softdtw.model.eval()
    
    seqs = []
    Song_input = []
    i, j, k = 26,27,80
    Song_data = []
    Song_Cover_data = []    
    Song_Cover_same = []
    Song_Cover_metrix = []
    Song_MaxSame_id =[]
    Song_MaxSame_metrix = []
    Song_MaxSame_sam = []
    Song_nocover_metrix = []
    Song_nocover_id = []
    Song_nocover_sam = []
    Song_inhance = []
    inhance_kernel = np.array([[ 0, 0, 0, 0,-1],
                               [ 0, 0, 0,-1, 0],
                               [ 0, 0, 4, 0, 0],
                               [ 0,-1, 0, 0, 0],
                               [-1, 0, 0, 0, 0]])
    inhance_kernel = torch.autograd.Variable(torch.from_numpy(inhance_kernel).unsqueeze(0).unsqueeze(0).float() )
    print(inhance_kernel.shape)
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        # if ii == i or ii == j or ii == k:
        #     input = data.cuda()
        #     Song_input.append(input)
        #     _,seq,_,_= softdtw.model(input)
        #     seqs.append(seq)
        # if ii<80:
        input = data.cuda()
        Song_input.append(input)
        
    for i in range(0,len(Song_input)-1,2):
        
        _,_,seqa,_= softdtw.model(Song_input[i])
        Song_data.append(seqa.data.cpu().numpy())
        _,_,seqb,_= softdtw.model(Song_input[i+1])
        Song_Cover_data.append(seqb.data.cpu().numpy())
        metrix=softdtw.metric(seqa, seqb, True)
        # si= F.conv2d(metrix.unsqueeze(0),inhance_kernel.cuda(),padding=2)####锐化
        # Song_inhance.append(si.squeeze(0).squeeze(0).data.cpu().numpy())####锐化
        Song_Cover_metrix.append(metrix.squeeze(0).data.cpu().numpy())
        sam = softdtw.multi_compute_s(Song_input[i],Song_input[i+1])
        Song_Cover_same.append(sam.data.cpu().numpy())
        max_num = sam
        max_id= i+1
        max_metrix = metrix
        Song_i_nocover_metrix = []
        Song_i_nocover_sam = []
        Song_i_nocover_id = []
        Song_i_nocover_data = []
        for j in range(0,i):
            _,_,seqb,_= softdtw.model(Song_input[j])
            Song_i_nocover_data.append(seqb.data.cpu().numpy())
            nocover_metrix = softdtw.metric(seqa, seqb, True)
            Song_i_nocover_metrix.append(nocover_metrix.squeeze(0).data.cpu().numpy())
            Song_i_nocover_id.append(j)
            nocover_sam = softdtw.multi_compute_s(Song_input[i],Song_input[j])
            Song_i_nocover_sam.append(nocover_sam.data.cpu().numpy())
            if max_num<nocover_sam:
                max_num= nocover_sam
                max_id = j
                max_metrix =nocover_metrix
        for j in range(i+2,len(Song_input)):
            _,_,seqb,_= softdtw.model(Song_input[j])
            Song_i_nocover_data.append(seqb.data.cpu().numpy())
            nocover_metrix = softdtw.metric(seqa, seqb, True)
            Song_i_nocover_metrix.append(nocover_metrix.squeeze(0).data.cpu().numpy())
            Song_i_nocover_id.append(j)
            nocover_sam = softdtw.multi_compute_s(Song_input[i],Song_input[j])
            Song_i_nocover_sam.append(nocover_sam.data.cpu().numpy())
            if max_num<nocover_sam:
                max_num= nocover_sam
                max_id = j
                max_metrix =nocover_metrix
        Song_MaxSame_id.append(max_id )
        Song_MaxSame_metrix.append(max_metrix.squeeze(0).data.cpu().numpy())
        Song_MaxSame_sam.append(max_num.data.cpu().numpy())
        Song_nocover_metrix.append(Song_i_nocover_metrix)
        Song_nocover_id.append(Song_i_nocover_id)
        Song_nocover_sam.append(Song_i_nocover_sam)
    # sap, d_ap, b_d_ap,align_ap,s_ap = softdtw.metric(seqs[0], seqs[1], True)
    # san, d_an, b_d_an,align_an,s_an = softdtw.metric(seqs[0], seqs[2], True)
    # print(softdtw(Song_input[0],Song_input[1],Song_input[2]))
    # b_d_ap = softdtw.metric(seqs[0], seqs[1], True)
    # b_d_an = softdtw.metric(seqs[0], seqs[2], True)
    # b_d_ap ,b_d_an = b_d_ap.squeeze(0),b_d_an.squeeze(0)
    image_count=0
    vis = visdom.Visdom()
    for i in range(len(Song_data)):
        print("Song_",i," :")
        print("  Song_Cover sam:",Song_Cover_same[i])
        vis.heatmap( Song_Cover_metrix[i],opts={'title':"image"+str(image_count)+' Cover_metrix'+str(i)+"&"+str(i+1)+':'+str(Song_Cover_same[i])})
        image_count+=1
        #vis.heatmap(      Song_inhance[i],opts={'title':"image"+str(image_count)+' inhance_Cover_metrix'+str(i)+"&"+str(i+1)})
        if Song_MaxSame_id[i]!= (2*i+1):
            print("  Song_",i,"_maxSam: id ",Song_MaxSame_id[i]," Sam ",Song_MaxSame_sam[i])
            print("-------------------Error pred: image_id ",image_count)
            print(Song_MaxSame_metrix[i].shape)
            print(Song_MaxSame_metrix[i])
            vis.heatmap( Song_MaxSame_metrix[i],opts={'title':"image"+str(image_count)+' Max_metrix'+str(i)+"&"+str(Song_MaxSame_id[i])+':'+str(Song_MaxSame_sam[i])})
           
    # d_ap, b_d_ap,align_ap =  d_ap.squeeze(0), b_d_ap.squeeze(0),align_ap.squeeze(0)
    # d_an, b_d_an,align_an =  d_an.squeeze(0), b_d_an.squeeze(0),align_an.squeeze(0)
    #print(align_an.shape)
    # d_ap, b_d_ap, align_ap = d_ap.data.cpu().numpy(), b_d_ap.data.cpu().numpy(), align_ap.data.cpu().numpy()
    # d_an, b_d_an, align_an = d_an.data.cpu().numpy(), b_d_an.data.cpu().numpy(), align_an.data.cpu().numpy()
    # s_ap = s_ap.data.cpu().numpy()[0]
    # s_an = s_an.data.cpu().numpy()[0]
    
    # vis.heatmap(align_ap,opts={'title':'align_ap'})
    # vis.heatmap(s_ap,opts={'title':'s_ap'})
    #vis.heatmap(b_d_ap,opts={'title':'b_d_ap'})
    
    # vis.heatmap(align_an,opts={'title':'align_an'})
    # vis.heatmap(s_an,opts={'title':'s_an'})
    #vis.heatmap(b_d_an,opts={'title':'b_d_an'})

    # print(sap, san)

    softdtw.train()
    softdtw.model.train()


def neuralwarp_train(**kwargs):
    # 多尺度图片训练 396+
    print(kwargs)
    #print("Mask == 1")

    with open(kwargs['params']) as f:
        params = json.load(f)
    if kwargs['manner'] == 'train':
        params['is_train'] = True
    else:
        params['is_train'] = False
    params['batch_size'] = kwargs['batch_size']
    if torch.cuda.device_count() > 1:
        print("-------------------Parallel_GPU_Train--------------------------")
        parallel = True
    else:
        print("------------------Single_GPU_Train----------------------")
        parallel = False
    opt.feature = 'cqt'
    opt.notes = 'SoftDTW'
    opt.model = 'SoftDTW'
    opt.batch_size = 'batch_size'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(kwargs["Device"])
    opt.Device=kwargs["Device"]
    #device_ids = [2]
    opt._parse(kwargs)

    model = getattr(models, opt.model)(params)

    p = 'check_points/' + model.model_name + opt.notes
    #f = os.path.join(p, "0620_07:05:30.pth")#使用Neural_dtw目前最优 0620_07:05:30.pth cover80 map:0.705113267654046 0.08125 7.96875
    #f = os.path.join(p, "0620_17:37:35.pth")
    #f = os.path.join(p, "0621_22:42:59.pth")#NeuralDTW_Milti_Metix_res 0622_16:33:07.pth 0621_22:42:59.pth
    #f = os.path.join(p, "0628_17:00:52.pth")#0628_17:00:52.pth  FCN
    #f = os.path.join(p,"0623_16:01:05.pth") #3seq
    #f = os.path.join(p,"0630_07:59:56.pth")#VGG11 0630_01:10:15.pth 0630_07:59:56.pth
    if  kwargs['model'] == 'NeuralDTW_CNN_Mask_dilation_SPP':
        f = os.path.join(p,"0704_19:58:25.pth")
    elif kwargs['model'] == 'NeuralDTW_CNN_Mask_dilation_SPP2':
        f = os.path.join(p,"0709_00:31:23.pth")
    elif kwargs['model'] == 'NeuralDTW_CNN_Mask_dilation':
        f = os.path.join(p,"0704_06:40:41.pth")
    opt.load_model_path = f
    if kwargs['model'] != 'NeuralDTW' and kwargs['manner'] != 'train':
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            print("load_model:",opt.load_model_path)
            model.load(opt.load_model_path)
    
    if parallel == True:
        model = DataParallel(model)
    model.to(opt.device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    # step2: data
    out_length =400
    if kwargs['model'] == 'NeuralDTW_CNN_Mask_300':
        out_length = 300
    if kwargs['model'] == 'NeuralDTW_CNN_Mask_spp':
        train_data0 = triplet_CQT(out_length=200, is_label=kwargs['is_label'], is_random=kwargs['is_random'])
        train_data1 = triplet_CQT(out_length=300, is_label=kwargs['is_label'], is_random=kwargs['is_random'])
        train_data2 = triplet_CQT(out_length=400, is_label=kwargs['is_label'], is_random=kwargs['is_random'])
    else:
        train_data0 = triplet_CQT(out_length=out_length, is_label=kwargs['is_label'], is_random=kwargs['is_random'])
        train_data1 = triplet_CQT(out_length=out_length, is_label=kwargs['is_label'], is_random=kwargs['is_random'])
        train_data2 = triplet_CQT(out_length=out_length, is_label=kwargs['is_label'], is_random=kwargs['is_random'])
    val_data80 = CQT('songs80', out_length=kwargs['test_length'])
    val_data = CQT('songs350', out_length=kwargs['test_length'])
    val_data_marukars = CQT('Mazurkas',out_length=kwargs['test_length'])
    
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
    val_dataloader_marukars = DataLoader(val_data_marukars,1, shuffle=False, num_workers=1)
    if kwargs['manner'] == 'test':
        # val_slow(model, val_dataloader, style='null')
        val_slow_batch(model,val_dataloader_marukars, batch=100, is_dis=kwargs['zo'])
    elif kwargs['manner'] == 'visualize':
        visualize(model, val_dataloader80)
    elif kwargs['manner'] == 'mul_test':
        p = 'check_points/' + model.model_name + opt.notes
        l = sorted(os.listdir(p))[: 20]
        best_MAP, MAP = 0, 0
        for f in l:
            f = os.path.join(p, f)
            model.load(f)
            model.to(opt.device)
            MAP += val_slow_batch(model, val_dataloader, batch=400, is_dis=kwargs['zo'])
            MAP += val_slow_batch(model, val_dataloader80, batch=400, is_dis=kwargs['zo'])
            if MAP > best_MAP:
                print('--best result--')
                best_MAP = MAP
            MAP = 0
    else:
        # step3: criterion and optimizer
        be = torch.nn.BCELoss()

        lr = opt.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

        # if parallel is True:
        #     optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True, min_lr=5e-6)
        # step4: train
        best_MAP = 0
        for epoch in range(opt.max_epoch):
            running_loss = 0
            num = 0
            for ii, ((a0, p0, n0, la0, lp0, ln0), (a1, p1, n1, la1, lp1, ln1), (a2, p2, n2, la2, lp2, ln2)) in tqdm(
                    enumerate(zip(train_dataloader0, train_dataloader1, train_dataloader2))):
                # for ii, (a2, p2, n2) in tqdm(enumerate(train_dataloader2)):
                for flag in range(3):
                    if flag == 0:
                        a, p, n, la, lp, ln = a0, p0, n0, la0, lp0, ln0
                    elif flag == 1:
                        a, p, n, la, lp, ln = a1, p1, n1, la1, lp1, ln1
                    else:
                        a, p, n, la, lp, ln = a2, p2, n2, la2, lp2, ln2
                    B, _, _, _ = a.shape
                    if kwargs["zo"] == True:
                        target = torch.cat((torch.zeros(B), torch.ones(B))).cuda()
                    else:
                        target = torch.cat((torch.ones(B), torch.zeros(B))).cuda()
                    # train model
                    a = a.requires_grad_().to(opt.device)
                    p = p.requires_grad_().to(opt.device)
                    n = n.requires_grad_().to(opt.device)

                    optimizer.zero_grad()
                    pred = model(a, p, n)
                    pred = pred.squeeze(1)   
                    loss = be(pred, target)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num += a.shape[0]

                if ii % 5000 == 0:
                    running_loss /= num
                    print("train_loss:",running_loss)
                
                    MAP = 0
                    print("Youtube350:")
                    MAP += val_slow_batch(model, val_dataloader, batch=1    , is_dis=kwargs['zo'])
                    print("CoverSong80:")
                    MAP += val_slow_batch(model, val_dataloader80, batch=1, is_dis=kwargs['zo'])
                    # print("Marukars:")
                    # MAP += val_slow_batch(model, val_dataloader_marukars, batch=100, is_dis=kwargs['zo'])
                    if MAP > best_MAP:
                        best_MAP = MAP
                        print('*****************BEST*****************')
                    if kwargs['save_model'] == True:
                        if parallel:
                            model.module.save(opt.notes)
                        else:
                            model.save(opt.notes)
                    scheduler.step(running_loss)
                    running_loss = 0
                    num = 0


@torch.no_grad()
def val_slow(softdtw, dataloader, style='null'):
    softdtw.eval()
    softdtw.model.eval()
    
    seqs, labels = [], []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda()
       
        seqs.append(input)
        labels.append(label)
    labels = torch.cat(labels, dim=0)

    N = labels.shape[0]
    if N == 350:
        query_l = [i // 100 for i in range(100 * 100, 350 * 100)]
        ref_l = [i for i in range(100)] * 250
    else:
        query_l = [i // N for i in range(N * N)]
        ref_l = [i for i in range(N)] * N
    dis2d = np.zeros((N, N))
    
    for st in range(0, N * N if N != 350 else 100 * 250):
        query = seqs[query_l[st]]
        ref = seqs[ref_l[st]]
        if style == 'min':
            T = min(query.shape[1], ref.shape[1])
            query, ref = query[:, :T, :], ref[:, :T, :]
        # print(softdtw.metric(query, ref))
        s = softdtw.multi_compute_s(query, ref).data.cpu().numpy()
        i, j = query_l[st], ref_l[st]
        dis2d[i, j], dis2d[j, i] = -s[0], -s[0]

    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1 )

    softdtw.train()
    softdtw.model.train()
    return MAP


@torch.no_grad()
def val_slow_batch(softdtw, dataloader, batch=100, is_dis='False'):
    softdtw.eval()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.eval()
    else:
        softdtw.model.eval()
    seqs, labels = [], []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda()
        #_, seq, _ = softdtw.model(input)
        seqs.append(input)
        labels.append(label)
    seqs = torch.cat(seqs, dim=0)
    labels = torch.cat(labels, dim=0)

    N = labels.shape[0]
    if N == 350:
        query_l = [i // 100 for i in range(100 * 100, 350 * 100)]
        ref_l = [i for i in range(100)] * 250
    else:
        query_l = [i // N for i in range(N * N)]
        ref_l = [i for i in range(N)] * N
    dis2d = np.zeros((N, N))

    N = N * N if N != 350 else 100 * 250
    for st in range(0, N, batch):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        if torch.cuda.device_count() > 1:
            s = softdtw.module.multi_compute_s(query, ref).data.cpu().numpy()
        else:
            s = softdtw.multi_compute_s(query, ref).data.cpu().numpy()
        for k in range(st, fi):
            i, j = query_l[k], ref_l[k]
            # print(i, j)
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]
    
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1 )

    softdtw.train()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.train()
    else:
        softdtw.model.train()
    return MAP



if __name__=='__main__':
    import fire
    fire.Fire()