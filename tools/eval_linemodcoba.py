import _init_paths as _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import trimesh
import cv2
import json
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def get_3D_corners(mesh):
    Tform = mesh.apply_obb()
    points = mesh.bounding_box.vertices
    min_x = np.min(points[:,0])
    min_y = np.min(points[:,1])
    min_z = np.min(points[:,2])
    max_x = np.max(points[:,0])
    max_y = np.max(points[:,1])
    max_z = np.max(points[:,2])
    corners = np.array([[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z], 
                        [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z], 
                        [max_x, max_y, min_z], [max_x, max_y, max_z]])
    corners = np.concatenate((np.transpose(corners), np.ones((1,8))), axis=0)
    return corners

def compute_projection(points_3D, transformation, internal_calibration):
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    projections_2d[0, :] = camera_projection[0, :] / camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :] / camera_projection[2, :]
    return projections_2d

def visualize_labels(img, mask, labelfile):
    cv2.addWeighted(mask, 0.4, img, 0.6, 0, img)
    if os.path.exists(labelfile):
        with open(labelfile, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                info = line.split()
        info = [float(i) for i in info]
        width, length = img.shape[:2]
        one = (int(info[3] * length), int(info[4] * width))
        two = (int(info[5] * length), int(info[6] * width))
        three = (int(info[7] * length), int(info[8] * width))
        four = (int(info[9] * length), int(info[10] * width))
        five = (int(info[11] * length), int(info[12] * width))
        six = (int(info[13] * length), int(info[14] * width))
        seven = (int(info[15] * length), int(info[16] * width))
        eight = (int(info[17] * length), int(info[18] * width))
        # Draw lines between the points to visualize bounding box
        cv2.line(img,one,two,(255,0,0),3)
        cv2.line(img,one,three,(255,0,0),3)
        cv2.line(img,two,four,(255,0,0),3)
        cv2.line(img,three,four,(255,0,0),3)
        cv2.line(img,one,five,(255,0,0),3)
        cv2.line(img,three,seven,(255,0,0),3)
        cv2.line(img,five,seven,(255,0,0),3)
        cv2.line(img,two,six,(255,0,0),3)
        cv2.line(img,four,eight,(255,0,0),3)
        cv2.line(img,six,eight,(255,0,0),3)
        cv2.line(img,five,six,(255,0,0),3)
        cv2.line(img,seven,eight,(255,0,0),3)
    return img

def main():

    opt.num_objects = 1
    opt.num_points = 3500
    opt.iteration = 1
    opt.objlist = [3]
    bs = 1
    vis_bbox = True

    # Load mesh and intrinsic data
    meshname = f'datasets/linemod/datasetbaru/models/obj_0{opt.objlist[0]}.ply'
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    mesh = trimesh.load(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(mesh)

    # Load Camera intrinsic array and depth scale
    
    fx = 613.23754883
    fy = 613.47509766
    cx = 312.71520996
    cy = 247.00277710
    depth_scale = 1.0
    intrinsic_calibration = get_camera_intrinsic(cx, cy, fx, fy)

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    estimator = estimator.cuda()
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects)
    refiner = refiner.cuda()
    
    # Load pretrained models
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    testdataset = PoseDataset_linemod('eval', opt.num_points, False, opt.dataset_root, 0.0, True)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)
    
    index = 120

    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    ori_image_path = testdataset.get_img(index)
    ori_mask_path = testdataset.get_mask(index)
    ori_label_path = testdataset.get_label(index)
    
    opt.diameter = np.array([0.0086610])
    print(opt.diameter)
    
    opt.success_count = [0 for i in range(opt.num_objects)]
    opt.num_count = [0 for i in range(opt.num_objects)]

    for i, data in enumerate(testdataloader, 0):
        image = cv2.imread(ori_image_path[i])
        mask = cv2.imread(ori_mask_path[i])
        points, choose, img, target, model_points, idx = data
        points, choose, img, target, model_points, idx = (points).cuda(), \
                                                          (choose).cuda(), \
                                                          (img).cuda(), \
                                                          (target).cuda(), \
                                                          (model_points).cuda(), \
                                                          (idx).cuda()
        
        # Initial pose estimation
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, opt.num_points, 1)
        pred_c = pred_c.view(bs, opt.num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * opt.num_points, 1, 3)
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * opt.num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)
        
        for ite in range(0, opt.iteration):
            T = (torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(opt.num_points, 1).contiguous().view(1, opt.num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = (torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t
            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2
            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = quaternion_from_matrix(my_mat_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])
            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final
        
        # Pose refinement result
        my_R = quaternion_matrix(my_r)[:3, :3]
        my_T = np.reshape(my_t, (len(my_t), 1))
        model_points = model_points[0].cpu().detach().numpy()
        my_r = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points, my_r.T) + my_t
        target = target[0].cpu().detach().numpy()
        
        if (idx[0].item()) in sym_list:
            pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
            inds = KNearestNeighbor.apply(target.unsqueeze(0), pred.unsqueeze(0))
            target = torch.index_select(target, 1, inds.view(-1) - 1)
            dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
        else:
            dis = np.mean(np.linalg.norm(pred - target, axis=1))
        
        if dis < opt.diameter[idx[0].item()]:
            opt.success_count[idx[0].item()] += 1
            print('No.{0} Pass! Distance: {1}'.format(i, dis))
        else:
            print('No.{0} NOT Pass! Distance: {1}'.format(i, dis))
        
        opt.num_count[idx[0].item()] += 1
        
        if vis_bbox:
            Rt_pred = np.concatenate((my_R, my_T), axis=1)
            proj_2d_pred = compute_projection(vertices, Rt_pred, intrinsic_calibration)
            proj_corners_pred = np.transpose(compute_projection(corners3D, Rt_pred, intrinsic_calibration))
            for edge in edges_corners:
                center_coordinates1 = (int(proj_corners_pred[edge[0], 0]), int(proj_corners_pred[edge[0], 1]))
                center_coordinates2 = (int(proj_corners_pred[edge[1], 0]), int(proj_corners_pred[edge[1], 1]))
                cv2.line(image, center_coordinates1, center_coordinates2, (0, 255, 255), 2)
            
            image = visualize_labels(image, mask, ori_label_path[i])
            cv2.imshow('output', image)
            cv2.waitKey(1)
    
    for i in range(opt.num_objects):
        print('Object {0} success rate: {1}'.format(opt.objlist[i], float(opt.success_count[i]) / opt.num_count[i]))
    
    print('ALL success rate: {0}'.format(float(sum(opt.success_count)) / sum(opt.num_count)))
    
if __name__ == '__main__':
    main()
