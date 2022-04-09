import _init_paths
from PIL import Image
import os
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import torch.utils.data
import torchgeometry as tgm
from lib.network import PoseNet, PoseRefineNet
from lib.KNN_CUDA.knn_cuda import KNN
from lib.loss import calculate_error, batch_least_square
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from tools.utils import load
import warnings
warnings.filterwarnings("ignore")

obj_id = 1
img_id = 0
kp_id = 0

dataroot = 'datasets/linemod/data'
trained_model = 'trained_checkpoints/linemod/pose_model_34_0.009908490548935764.pth'
refine_model = 'trained_checkpoints/linemod/pose_refine_model_980_0.004116693084371488.pth'
obj_num = 13
obj_list = [1, 2, 4, 5, 6, 8, 9 ,10, 11, 12, 13, 14, 15]
if obj_id in [10, 11]:
    issym = True
else:
    issym = False
num_points = 500
num_vote = 9
iteration = 2

knn = KNN(k=1, transpose_mode=True)
estimator = PoseNet(num_points=num_points, num_vote=num_vote, num_obj = obj_num)
estimator.cuda()
estimator.load_state_dict(torch.load(trained_model))
estimator.eval()
refiner = PoseRefineNet(num_points=num_points, num_obj = obj_num)
refiner.cuda()
refiner.load_state_dict(torch.load(refine_model))
refiner.eval()

rgb = Image.open('{0}/data/{1}/rgb/{2}.png'.format(dataroot, '%02d'%obj_id, '%04d'%img_id))
depth = np.array(Image.open('{0}/data/{1}/depth/{2}.png'.format(dataroot, '%02d'%obj_id, '%04d'%img_id)))
if (os.path.exists('{0}/segnet_results/{1}_label/{2}_label.png'.format(dataroot, '%02d'%obj_id, '%04d'%img_id))):
    label = np.array(Image.open('{0}/segnet_results/{1}_label/{2}_label.png'.format(dataroot, '%02d'%obj_id, '%04d'%img_id)))
else:
    label = np.array(Image.open('{0}/data/{1}/mask/{2}.png'.format(dataroot, '%02d'%obj_id, '%04d'%img_id)))
pose = np.load('{0}/data/{1}/pose/pose{2}.npy'.format(dataroot, '%02d'%obj_id, img_id))

model_file = '{0}/models/obj_{1}.ply'.format(dataroot, '%02d'%obj_id)
kp = np.loadtxt('{0}/keypoints/fps_{1}.txt'.format(dataroot, '%02d'%obj_id))
models_info = yaml.load(open('{0}/models/models_info.yml'.format(dataroot), 'r'))
diameter= models_info[obj_id]['diameter'] / 1000.0 * 0.1
cam_cx = 325.26110
cam_cy = 242.04899
cam_fx = 572.41140
cam_fy = 573.57043

points, choose, img, target, model_points, model_kp, scene_kp = load.loaddata(rgb, depth, label, pose, model_file, kp)
if len(points.size()) == 2:
    print('NOT Pass! Lost detection!')
idx = torch.LongTensor([obj_list.index(obj_id)]).cuda()
points, choose, img, target, model_points, model_kp, scene_kp = points.cuda(), choose.cuda(), img.cuda(), target.cuda(), model_points.cuda(), model_kp.cuda(), scene_kp.cuda()
vertex_pred, c_pred, emb = estimator(img, points, choose, idx)
kp_set = vertex_pred + points.repeat(1, 1, 9).view(1, 500, 9, 3)
confidence = c_pred / (0.00001 + torch.sum(c_pred, 1))
points_pred = torch.sum(confidence * kp_set, 1)
points_weight = torch.ones(9).cuda()

all_index = torch.combinations(torch.arange(9), 3)
all_r, all_t = batch_least_square(model_kp.squeeze()[all_index, :], points_pred.squeeze()[all_index, :], torch.ones([all_index.shape[0], 3]).cuda())
all_e = calculate_error(all_r, all_t, model_points, points)
e = all_e.unsqueeze(0).unsqueeze(2)
conv = torch.nn.Conv1d(all_index.shape[0], all_index.shape[0], 1).cuda()
w = torch.softmax(conv(e), 1).squeeze().unsqueeze(1)
all_qua = tgm.rotation_matrix_to_quaternion(torch.cat((all_r, torch.tensor([0., 0., 1.]).cuda().unsqueeze(1).repeat(all_index.shape[0], 1, 1)), dim=2))
pred_qua = torch.sum(w * all_qua, 0)
pred_r = pred_qua.view(1, 1, -1)
bs, num_p, _ = pred_r.size()
pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
pred_r = torch.cat(((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                    (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                    (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                    (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1), \
                    (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                    (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                    (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                    (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1), \
                    (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)
my_r = pred_r.squeeze()
my_t = torch.sum(w * all_t, 0)
my_r = my_r.cpu().detach().numpy()
my_t = my_t.cpu().detach().numpy()

for ite in range(0, iteration):
    R = torch.unsqueeze(torch.from_numpy(my_r.astype(np.float32)), 0).cuda()
    ori_t = torch.unsqueeze(torch.from_numpy(my_t.astype(np.float32)), 0).cuda()
    T = ori_t.repeat(num_points, 1).contiguous().view(1, num_points, 3)
    my_mat = np.column_stack((my_r, my_t))
    my_mat = np.row_stack((my_mat, [0, 0, 0, 1]))

    new_points = torch.bmm((points - T), R).contiguous()
    pred_r, pred_t = refiner(new_points, emb, idx)
    pred_r = pred_r.view(1, 1, -1)
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
    my_r_2 = pred_r.view(-1).cpu().data.numpy()
    my_t_2 = pred_t.view(-1).cpu().data.numpy()
    my_mat_2 = quaternion_matrix(my_r_2)
    my_mat_2[0:3, 3] = my_t_2

    my_mat_final = np.dot(my_mat, my_mat_2)
    my_r = my_mat_final[:3,:3]
    my_t = my_mat_final[:3,3]

print (my_r)
print (my_t)
model_points = model_points[0].cpu().detach().numpy()
pred = np.dot(model_points, my_r.T) + my_t
target = target[0].cpu().detach().numpy()

if issym:
    pred = torch.from_numpy(pred.astype(np.float32)).cuda().contiguous()
    target = torch.from_numpy(target.astype(np.float32)).cuda().contiguous()
    dist, inds = knn(pred.unsqueeze(0), target.unsqueeze(0))
    dis = torch.mean(dist.squeeze())
else:
    dis = np.mean(np.linalg.norm(pred - target, axis=1))

if dis < diameter:
    print('Pass! Distance: {0}'.format(dis))
else:
    print('NOT Pass! Distance: {0}'.format(dis))

trans_point = np.zeros([pred.shape[0], 2]).astype(int)
for i, p in enumerate(pred):
    trans_point[i][0] = int(p[0] * cam_fx / p[2] + cam_cx)
    trans_point[i][1] = int(p[1] * cam_fy / p[2] + cam_cy)
vis_im_rgb = np.array(rgb).astype(np.float32)
for point in trans_point:
    cv2.circle(vis_im_rgb, tuple(point), 1, (0,255,0), -1)
cv2.imwrite('results/linemod/pose_{}_{:04d}.png'.format(obj_id, img_id), vis_im_rgb[...,::-1])
