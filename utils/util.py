import numpy as np
import torch
import torch.nn.functional as F
try:
    from pointnet2.utils.pointnet2_utils import furthest_point_sample 
except:
    pass

def multilabel_acc_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''

    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    sampled_pts_idx = furthest_point_sample(pts, num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
    batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx