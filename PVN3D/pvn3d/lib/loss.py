from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from lib.utils.meanshift_pytorch import MeanShiftTorch
from lib.KNN_CUDA.knn_cuda import KNN
import torchgeometry as tgm


class FocalLoss(_Loss):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def of_l1_loss(
        pred_ofsts, kp_targ_ofst, labels,
        sigma=1.0, normalize=True, reduce=False
):
    '''
    :param pred_ofsts:      [bs, n_kpts, n_pts, c]
    :param kp_targ_ofst:    [bs, n_pts, n_kpts, c]
    :param labels:          [bs, n_pts, 1]
    '''
    w = (labels > 1e-8).float()
    bs, n_kpts, n_pts, c = pred_ofsts.size()
    sigma_2 = sigma ** 3
    w = w.view(bs, 1, n_pts, 1).repeat(1, n_kpts, 1, 1).contiguous()
    kp_targ_ofst = kp_targ_ofst.view(bs, n_pts, n_kpts, 3)
    kp_targ_ofst = kp_targ_ofst.permute(0, 2, 1, 3).contiguous()
    diff = pred_ofsts - kp_targ_ofst
    abs_diff = torch.abs(diff)
    abs_diff = w * abs_diff
    in_loss = abs_diff

    if normalize:
        in_loss = torch.sum(
            in_loss.view(bs, n_kpts, -1), 2
        ) / (torch.sum(w.view(bs, n_kpts, -1), 2) + 1e-3)

    if reduce:
        torch.mean(in_loss)

    return in_loss


class OFLoss(_Loss):
    def __init__(self):
        super(OFLoss, self).__init__(True)

    def forward(
            self, pred_ofsts, kp_targ_ofst, labels,
            normalize=True, reduce=False
    ):
        l1_loss = of_l1_loss(
            pred_ofsts, kp_targ_ofst, labels,
            sigma=1.0, normalize=True, reduce=False
        )

        return l1_loss


def BMM(A, B):
    C = torch.empty(A.size()[0], A.size()[1], B.size()[2])
    B = torch.transpose(B, 2, 1)
    for i in range(A.size()[0]):
        for j in range(A.size()[1]):
            for k in range(B.size()[1]):
                C[i][j][k] = torch.dot(A[i][j], B[i][k])
    return C


def smooth_l1_loss(vertex_pred, vertex_targets, sigma=1.0, normalize=True):
    b,ver_dim,num_points=vertex_pred.shape
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * num_points)

    return in_loss


def batch_least_square(A, B, w):

    assert A.shape == B.shape
    num = A.shape[0]
    centroid_A = torch.mean(A, dim=1)
    centroid_B = torch.mean(B, dim=1)
    AA = A - centroid_A.unsqueeze(1)
    BB = B - centroid_B.unsqueeze(1)

    # H = torch.bmm(torch.transpose(AA, 2, 1), BB)
    H = BMM(torch.transpose(AA, 2, 1), BB)
    U = torch.empty(H.size()[0], H.size()[1], H.size()[2])
    S = torch.empty(H.size()[0], H.size()[2])
    Vt = torch.empty(H.size()[0], H.size()[2], H.size()[2])
    for i in range(H.size()[0]):
        U[i], S[i], Vt[i] = torch.svd(H[i])
    U, S, Vt = U.cuda(), S.cuda(), Vt.cuda()

    R = BMM(Vt, U.permute(0, 2, 1))
    j = torch.empty(R.size()[0])
    for i in range(R.size()[0]):
        j[i] = torch.det(R[i])
    i = j < 0
    tmp = torch.ones([num, 3, 3], dtype=torch.float32).cuda()
    tmp[i, :, 2] = -1
    Vt = Vt * tmp

    R = torch.bmm(Vt, U.permute(0, 2, 1))
    t = centroid_B - torch.bmm(R, centroid_A.unsqueeze(2)).squeeze()
    return R, t


def calculate_error(r, t, model_points, scene_points):
    pred = torch.bmm(model_points[0].expand(r.shape[0], model_points.shape[1], 3), r.permute(0, 2, 1)) + t.expand(model_points.shape[1],r.shape[0],3).permute(1,0,2)
    knn = KNN(k=1, transpose_mode=True)
    target = scene_points[0].expand(r.shape[0], scene_points.shape[1], 3)
    dist, inds = knn(pred, target)
    dis = torch.mean(dist.squeeze(), 1)
    return dis


class CONFLoss(_Loss):

    def __init__(self):
        super(CONFLoss, self).__init__(True)
        # self.num_pt_mesh = num_points_mesh
        # self.sym_list = sym_list

    def forward(self, pred_kp_of, c_pred, points, model_points, model_kp, target_r, target_t):
        # vertex_loss = smooth_l1_loss(vertex_pred.view(1, self.num_pt_mesh, -1), vertex_gt.view(1, self.num_pt_mesh, -1))

        bs, n_kps, n_pts, c = pred_kp_of.size()
        
        # print(points.size(), pred_kp_of.size())
        kp_set = points.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of[0]
        # confidence = c_pred / (0.00001 + torch.sum(c_pred))
        confidence = torch.empty(n_kps, n_pts, 1).cuda()
        # print(kp_set.size())
        for i in range(n_kps):
            confidence[i] = c_pred[0][i] / torch.sum(0.00001 + c_pred[0][i])
        points_pred = torch.sum(confidence * kp_set, 1)

        all_index = torch.tensor([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6], [0, 2, 7], [0, 3, 4], [0, 3, 5], [0, 3, 6], [0, 3, 7], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 5, 6], [0, 5, 7], [0, 6, 7], [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7], [1, 3, 4], [1, 3, 5], [1, 3, 6], [1, 3, 7], [1, 4, 5], [1, 4, 6], [1, 4, 7], [1, 5, 6], [1, 5, 7], [1, 6, 7], [2, 3, 4], [2, 3, 5], [2, 3, 6], [2, 3, 7], [2, 4, 5], [2, 4, 6], [2, 4, 7], [2, 5, 6], [2, 5, 7], [2, 6, 7], [3, 4, 5], [3, 4, 6], [3, 4, 7], [3, 5, 6], [3, 5, 7], [3, 6, 7], [4, 5, 6], [4, 5, 7], [4, 6, 7], [5, 6, 7]])
        # print(model_kp.size())
        all_r, all_t = batch_least_square(model_kp[0].squeeze()[all_index, :], points_pred.squeeze()[all_index, :], torch.ones([all_index.shape[0], 3]).cuda())
        all_e = calculate_error(all_r, all_t, model_points, points)
        e = all_e.unsqueeze(0).unsqueeze(2)
        w = torch.softmax(1 / e, 1).squeeze().unsqueeze(1)
        all_qua = tgm.rotation_matrix_to_quaternion(torch.cat((all_r, torch.tensor([0., 0., 1.]).cuda().unsqueeze(1).repeat(all_index.shape[0], 1, 1)), dim=2))
        pred_qua = torch.sum(w * all_qua, 0)
        pred_r = pred_qua.view(1, 1, -1)
        bs, num_p, _ = pred_r.size()
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        pred_r = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p,1), \
                            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1), \
                            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
        pred_r = pred_r.squeeze()
        pred_t = torch.sum(w * all_t, 0)

        target_r = target_r.squeeze()
        target_t = target_t.squeeze()
        pose_loss = torch.norm(pred_t - target_t) + 0.01 * torch.norm(torch.mm(pred_r, torch.transpose(target_r, 1, 0)) - torch.eye(3).cuda())

        return pose_loss