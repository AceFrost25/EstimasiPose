import _init_paths
import os
import sys
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from lib.network import PoseNet, PoseRefineNet
from lib.utils import cloud_to_dims, iterative_points_refine
from transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import cv2
import numpy as np

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
parser.add_argument('--model', type=str, default='', help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default='', help='resume PoseRefineNet model')
opt = parser.parse_args()


class Visualizer(object):
    def __init__(self, objlist, mesh_points, list_obj, list_rgb, cam_info, point_scale=1.):
        self.objlist = objlist
        self.pt = mesh_points
        self.list_obj = list_obj
        self.list_rgb = list_rgb
        self.point_scale = point_scale  # point cloud unit in mm, point_scale = 1000.
        self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy = cam_info[0], cam_info[1], cam_info[2], cam_info[3]

        self.list_dims = self.compute_obj_dim()
        self.cur_r = np.array([0., 0., 0., 1.])
        self.cur_t = np.zeros([3])

    def compute_obj_dim(self):
        list_dims = {}
        for i in self.objlist:
            model_points = self.pt[i] / self.point_scale
            obj_dims = cloud_to_dims(model_points)
            list_dims[i] = obj_dims
        return list_dims

    def update_transformation(self, new_r, new_t):
        self.cur_r = new_r
        self.cur_t = new_t

    def update_cam_info(self, cam_info):
        self.cam_cx, self.cam_cy, self.cam_fx, self.cam_fy = cam_info[0], cam_info[1], cam_info[2], cam_info[3]

    def transform_points(self, bbox_points):
        my_t = self.cur_t
        my_r = quaternion_matrix(self.cur_r.copy())[:3, :3]
        pred_bbox = np.dot(bbox_points, my_r.T) + my_t  # prediction of sampled pixel after being transformed
        return pred_bbox

    def visualize_item(self, index, target=None):
        img = cv2.imread(self.list_rgb[index])
        obj = self.list_obj[index]
        self.visualize_img(img, obj, target, True)

    def visualize_img(self, img, obj, target=None, cv_show=False):
        obj_bbox = self.list_dims[obj]['bbox']
        obj_bbox = self.transform_points(obj_bbox)
        obj_bbox_pxls = self.project_point_pxl(obj_bbox)
        for px in obj_bbox_pxls:
            img = cv2.circle(img, tuple(px), radius=1, color=(255, 0, 0), thickness=2)
        if target is not None:
            for px in target:
                img = cv2.circle(img, tuple(px), radius=1, color=(0, 0, 255), thickness=1)
        if cv_show:
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def project_point_pxl(self, points, distCoeffs=None):
        points = np.asarray(points)
        cam_matrix = np.eye(3)
        cam_matrix[0, 0] = self.cam_fx
        cam_matrix[1, 1] = self.cam_fy
        cam_matrix[0, 2] = self.cam_cx
        cam_matrix[1, 2] = self.cam_cy
        pixel_points, _ = cv2.projectPoints(points.reshape(1, -1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
                                            cam_matrix, distCoeffs)
        return np.floor(pixel_points.reshape((-1, 2))).astype(int)

    def swap_pxls(self, pxls):
        temp = pxls.copy()
        temp[:, 0], temp[:, 1] = temp[:, 1], temp[:, 0].copy()
        return temp


class PoseDataset_visualize(PoseDataset_linemod):
    def __init__(self, mode, num_pointcloud, add_noise, root, noise_trans, refine, objlist):
        super(PoseDataset_visualize, self).__init__(mode, num_pointcloud, add_noise, root, noise_trans, refine, objlist)


class PoseYCBDataset_visualize(PoseDataset_ycb, Visualizer):
    def __init__(self, mode, num_pointcloud, add_noise, root, noise_trans, refine):
        PoseDataset_ycb.__init__(self, mode, num_pointcloud, add_noise, root, noise_trans, refine)
        cam_info = [self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1]  # TODO: work with this
        objlist = self.list_obj.keys()
        pt = self.cld
        Visualizer.__init__(self, objlist, pt, self.list_obj, self.list_rgb, cam_info)


def main():
    num_objects = 1
    num_points = 500
    objlist = [3]
    iteration = 2
    bs = 1

    # Create PoseDataset_visualize instance
    testdataset = PoseDataset_visualize('eval', num_points, False, opt.dataset_root, 0.0, True, objlist)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=bs, shuffle=False, num_workers=10)

    # Create Visualizer instance
    visualizer = Visualizer(testdataset.objlist, testdataset.pt, testdataset.list_obj, testdataset.list_rgb,
                            [testdataset.cam_cx, testdataset.cam_cy, testdataset.cam_fx, testdataset.cam_fy])

    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()
    index = 500

    points, choose, img, target, model_points, idx = testdataloader.dataset[index]

    points, choose, img, target, model_points, idx = Variable(points.unsqueeze(0)).cuda(), \
                                                     Variable(choose.unsqueeze(0)).cuda(), \
                                                     Variable(img.unsqueeze(0)).cuda(), \
                                                     Variable(target.unsqueeze(0)).cuda(), \
                                                     Variable(model_points.unsqueeze(0)).cuda(), \
                                                     Variable(idx.unsqueeze(0)).cuda()

    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    _, my_r, my_t = iterative_points_refine(refiner, points, emb, idx, iteration, my_r, my_t, bs, num_points)

    # Update transformation using Visualizer instance
    visualizer.update_transformation(my_r, my_t)

    target = target.cpu().detach().numpy()
    target_pxl = visualizer.project_point_pxl(target)
    visualizer.visualize_item(index, target_pxl)


if __name__ == '__main__':
    main()

