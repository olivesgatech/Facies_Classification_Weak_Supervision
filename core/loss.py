import torch
import torch.nn.functional as F
import numpy as np

def focal_loss2d(input, target, conf=None, alpha=None ,sim=None , gamma=None, size_average=False, loss_type=None,soft_dev=None):
    N,c,w,h=target.shape
    #one_hot = torch.cuda.FloatTensor(input.shape).zero_()
    #target = one_hot.scatter_(1, target.data, 1)#create one hot representation for the target

    logpt = F.log_softmax(input,dim=1) #log softmax of network output (softmax across first dimension)
    pt = logpt.exp() #Confidence of the network in the output
   #confidence values
    if gamma == None:
        raise ValueError("Gamma Not Defined")

    if conf is not None:
        qt = torch.cat((torch.cuda.FloatTensor(N,1,w,h).zero_(), conf.squeeze().permute(0,3,1,2)), 1) #TODO question: all conf for class 0 is 0? so class 0 is not used in loss fun?!
                                                                                                      #Why conf is 6 channels while nw op is 7 channels?
    else:
        qt = torch.cuda.FloatTensor(pt.shape).fill_(1)

    #similarity values
    if sim is not None:
        st = torch.cuda.FloatTensor(pt.shape).fill_(1)*sim.float().view(-1,1,1,1)
    else:
        st = torch.cuda.FloatTensor(pt.shape).fill_(1)

    # alpha is used for class imbalance:
    if alpha is not None:
        at = torch.cuda.FloatTensor(pt.shape).fill_(1) * alpha.float().view(1, -1, 1, 1)
    else:
        at = torch.cuda.FloatTensor(pt.shape).fill_(1)

    if loss_type =='baseline':
        loss = -1 * (1-pt)**gamma * at*target*logpt
    elif loss_type == 'LinConf_NonSim':
        loss = -1 * (1 - pt) ** gamma *qt * at * target * logpt
    elif loss_type =='LinConf_LinSim':
        loss = -1 * (1 - pt) ** gamma * qt * st * at * target * logpt
    elif loss_type =='SoftConf_SoftSim':
        if soft_dev is not None:
            c1,c2=soft_dev
        else:
            raise ValueError('soft_dev is None.')
        loss = -1 * (1 - pt) ** gamma * F.softmax(qt*c2) * F.softmax(st*c1) * at * target * logpt
    else:
        raise ValueError("Loss Type Not Defined")

    if size_average: 
        return loss.mean()
    else:
        return loss.sum()