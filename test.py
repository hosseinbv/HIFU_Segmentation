import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import shutil, cv2
import matplotlib
from skimage.io import imsave
import random


random.seed(0)
matplotlib.use('tkagg')
torch.manual_seed(0)
np.random.seed(0)

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return np.mean(loss_accumulator)


@torch.no_grad()
def local_test(model, device, test_loader, perf_measure, saving_path, threshold=0.5, do_save=True):
    t = time.time()
    model.eval()
    perf_accumulator = []
    cnt = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        if do_save:
            for out_id in range(output.size()[0]):
                probs = torch.sigmoid(output[out_id,0,:,:])
                a = probs.cpu().detach().numpy()
                a[a>=threshold] = 1
                a[a<threshold] = 0
                imsave(os.path.join(saving_path, 'pred_'+str(cnt)+'.jpg'), a*255)

                y = target[out_id,0,:,:].cpu().detach().numpy()
                imsave(os.path.join(saving_path, 'true_'+str(cnt)+'.jpg'), y*255)
                cnt += 1

        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )
    print('performances per each case: ', perf_accumulator)
    return np.mean(perf_accumulator), np.std(perf_accumulator)


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.dataset == "Kvasir":
        img_path = args.root + "/images/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "/masks/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset == "CVC":
        img_path = args.root + "Original/*"
        input_paths = sorted(glob.glob(img_path))
        depth_path = args.root + "Ground Truth/*"
        target_paths = sorted(glob.glob(depth_path))
    elif args.dataset.lower() == "hifu":
        img_path_b = args.root+"images/before/*"
        img_path_a = args.root+"images/after/"
        mask_path = args.root + "masks/"
        input_paths = sorted(glob.glob(img_path_b))
    test_loader = dataloaders.get_dataloaders(
        input_paths, img_path_a, mask_path, input_paths, img_path_a, mask_path, batch_size=args.batch_size, test_only=True
    )

    perf = performance_metrics.DiceScore()

    model = models.FCBFormer()

    state_dict = torch.load(
        "./Trained models/FCBFormer_{}.pt".format(args.dataset)
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)

    return (
        device,
        test_loader,
        perf,
        model,
    )


def test(args):
    (
        device,
        test_dataloader,
        perf,
        model,
    ) = build(args)
    if os.path.exists(args.saving_path):
        shutil.rmtree(args.saving_path)
        os.mkdir(args.saving_path)
    else:
        os.mkdir(args.saving_path)
    test_measure_mean, test_measure_std = local_test(model, device, test_dataloader, perf, args.saving_path, args.threshold,
    do_save=True)
    print('Finished...!')


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default='Hifu', choices=["hifu", "Kvasir", "CVC"])
    parser.add_argument("--data-root", type=str, dest="root")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=0.5)
    parser.add_argument("--saving_path", type=str, default="prediction_results")
    parser.add_argument(
        "--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"]
    )

    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    args.root = 'Data/HIFU_data/test_data/'

    test(args)


if __name__ == "__main__":
    main()
