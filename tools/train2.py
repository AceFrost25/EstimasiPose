# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ycb', help='ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir (\'YCB_Video_Dataset\' or \'datasetbaru\')')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', type=float, default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', type=float, default=0.015, help='learning rate weight')
    parser.add_argument('--w_rate', type=float, default=0.3, help='learning rate weight decay rate')
    parser.add_argument('--decay_margin', type=float, default=0.016, help='margin to decay lr & w')
    parser.add_argument('--refine_margin', type=float, default=0.013, help='margin to start the training of iterative refinement')
    parser.add_argument('--noise_trans', type=float, default=0.03, help='range of the random noise of translation added to the training data')
    parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
    parser.add_argument('--nepoch', type=int, default=4, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
    parser.add_argument('--resume_refinenet', type=str, default='', help='resume PoseRefineNet model')
    parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
    return parser.parse_args()

def setup_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_model(opt, device):
    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects).to(device)
    refiner = PoseRefineNet(num_points=opt.num_points, num_obj=opt.num_objects).to(device)

    if opt.resume_posenet:
        estimator.load_state_dict(torch.load(os.path.join(opt.outf, opt.resume_posenet)))
    if opt.resume_refinenet:
        refiner.load_state_dict(torch.load(os.path.join(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size //= opt.iteration
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    
    return estimator, refiner, optimizer

def load_data(opt):
    if opt.dataset == 'ycb':
        train_dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        train_dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    else:
        raise ValueError("Unknown dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)

    return train_loader, test_loader

def main():
    opt = parse_args()
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    device = setup_device()

    if opt.dataset == 'ycb':
        opt.num_objects = 21
        opt.num_points = 1000
        opt.outf = 'trained_models/ycb'
        opt.log_dir = 'experiments/logs/ycb'
        opt.repeat_epoch = 1
    elif opt.dataset == 'linemod':
        opt.num_objects = 12
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        raise ValueError('Unknown dataset')

    estimator, refiner, optimizer = load_model(opt, device)
    train_loader, test_loader = load_data(opt)

    opt.sym_list = train_loader.dataset.get_sym_list()
    opt.num_points_mesh = train_loader.dataset.get_num_points_mesh()

    print(f'Dataset loaded!\nLength of the training set: {len(train_loader.dataset)}\n'
          f'Length of the testing set: {len(test_loader.dataset)}\n'
          f'Number of sample points on mesh: {opt.num_points_mesh}\n'
          f'Symmetry object list: {opt.sym_list}')

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))

    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger(f'epoch{epoch}', os.path.join(opt.log_dir, f'epoch_{epoch}_log.txt'))
        logger.info(f'Train time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))}, Training started')
        
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):
            for i, data in enumerate(train_loader, 0):
                points, choose, img, target, model_points, idx = [Variable(x).to(device) for x in data]
                
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info(f'Train time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))} '
                                f'Epoch {epoch} Batch {train_count // opt.batch_size} Frame {train_count} Avg_dis: {train_dis_avg / opt.batch_size}')
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    model_save_path = f'{opt.outf}/pose_model_current.pth'
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), model_save_path.replace('pose_model', 'pose_refine_model'))
                    else:
                        torch.save(estimator.state_dict(), model_save_path)
        
        print(f'Epoch {epoch} train finish')

        logger = setup_logger(f'epoch{epoch}_test', os.path.join(opt.log_dir, f'epoch_{epoch}_test_log.txt'))
        logger.info(f'Test time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))}, Testing started')

        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        with torch.no_grad():
            for j, data in enumerate(test_loader, 0):
                points, choose, img, target, model_points, idx = [Variable(x).to(device) for x in data]

                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

                test_dis += dis.item()
                logger.info(f'Test time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))} '
                            f'Test Frame No.{test_count} dis: {dis}')
                test_count += 1
        
        test_dis /= test_count
        logger.info(f'Test time {time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time))} '
                    f'Epoch {epoch} TEST FINISH Avg dis: {test_dis}')

        if test_dis <= best_test:
            best_test = test_dis
            model_save_path = f'{opt.outf}/pose_model_{epoch}_{test_dis}.pth'
            if opt.refine_start:
                torch.save(refiner.state_dict(), model_save_path.replace('pose_model', 'pose_refine_model'))
            else:
                torch.save(estimator.state_dict(), model_save_path)
            print(f'Epoch {epoch} BEST TEST MODEL SAVED')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size //= opt.iteration
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            train_loader, test_loader = load_data(opt)
            opt.sym_list = train_loader.dataset.get_sym_list()
            opt.num_points_mesh = train_loader.dataset.get_num_points_mesh()

            print(f'Dataset loaded!\nLength of the training set: {len(train_loader.dataset)}\n'
                  f'Length of the testing set: {len(test_loader.dataset)}\n'
                  f'Number of sample points on mesh: {opt.num_points_mesh}\n'
                  f'Symmetry object list: {opt.sym_list}')

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()

