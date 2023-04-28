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
from transunet_networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from transunet_networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils import DiceLoss
from Models.asam import ASAM
from skimage.io import imread
import json
import multiprocessing


p = multiprocessing.Pool(4)

random.seed(12)
matplotlib.use('tkagg')
torch.manual_seed(0)
np.random.seed(0)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

kernel = np.ones((3, 3), np.uint8)
def save_overlay(background, true_mask, pred_mask, x_path, y_path, json_path, margin, mean=0.13466596,std=0.04942362):
    # background = background* std+ mean
    # background = ((background* std+ mean)*255).astype('uint8')
    true_mask=true_mask.astype('uint8')
    pred_mask=pred_mask.astype('uint8')
    colored_true_mask = np.zeros((np.shape(true_mask)[0],np.shape(true_mask)[1],3), dtype="uint8")
    colored_true_mask[true_mask==255, 0] = 200
    colored_true_mask[true_mask==255, 1] = 100
    colored_true_mask[true_mask==255, 2] = 0

    colored_pred_mask = np.zeros((np.shape(pred_mask)[0],np.shape(pred_mask)[1],3), dtype="uint8")
    colored_pred_mask[pred_mask==255, 0] = 50
    colored_pred_mask[pred_mask==255, 1] = 250
    colored_pred_mask[pred_mask==255, 2] = 200

    img = imread(x_path)
    mask = imread(y_path)
    f = open(json_path)
    data = json.load(f)
    points = data['shapes'][0].get('points')
    points = np.asarray(points)
    mins = np.min(points, axis=0).astype('int32') - margin
    maxs = np.max(points, axis=0).astype('int32') + margin

    mask_of_interest = mask[mins[1]:maxs[1], mins[0]:maxs[0], :]
    A = cv2.resize(pred_mask,(mask_of_interest.shape[1],mask_of_interest.shape[0]))
    orig_size_pred_mask = np.zeros((np.shape(mask)[0],np.shape(mask)[1]), dtype="uint8")
    colored_orig_size_pred_mask = np.zeros((np.shape(mask)[0],np.shape(mask)[1], 3), dtype="uint8")

    
    mask_edge = cv2.Canny(mask[:,:,0], 90, 130)
    mask_edge = cv2.dilate(mask_edge, kernel, iterations=1)

    orig_size_pred_mask[mins[1]:maxs[1], mins[0]:maxs[0]] = A
    colored_orig_size_pred_mask[orig_size_pred_mask==255, 0] = 50
    colored_orig_size_pred_mask[orig_size_pred_mask==255, 1] = 250
    colored_orig_size_pred_mask[orig_size_pred_mask==255, 2] = 200

    # colored_orig_size_pred_mask[mask_edge==255, 0] = 255
    # colored_orig_size_pred_mask[mask_edge==255, 1] = 0
    # colored_orig_size_pred_mask[mask_edge==255, 2] = 0

    added_image = cv2.addWeighted(colored_true_mask,0.7,colored_pred_mask,0.3,0)
    orig_size_added_image = cv2.addWeighted(img, 0.7, colored_orig_size_pred_mask, 0.3, 0)

    return added_image, orig_size_added_image


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, minimizer):
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target, _, _, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = 0.6*Dice_loss(output, target) + 0.4*BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        # # Ascent Step
        # outputs = model(data)
        # loss =  BCE_loss(torch.sigmoid(outputs), target)
        # loss.mean().backward()
        # minimizer.ascent_step()

        # # Descent Step
        # outputs = model(data)
        # (BCE_loss(torch.sigmoid(outputs), target)).mean().backward()
        # minimizer.descent_step()
    

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
def test(model, device, test_loader, epoch, perf_measure, args, do_save=False):
    if not os.path.exists(os.path.join(args.root,args.model)) and do_save:
        os.mkdir(os.path.join(args.root,args.model))
    t = time.time()
    model.eval()
    perf_accumulator = []
    cnt = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for batch_idx, (data, target, data_paths, target_paths, json_paths) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        start.record()
        output = model(data)
        end.record()
        if do_save:
            for out_id in range(output.size()[0]):
                probs = torch.sigmoid(output[out_id,0,:,:])
                dice = perf_measure(output, target).item()
                a = probs.cpu().detach().numpy()
                a[a>=0.5] = 1
                a[a<0.5] = 0
                true = target[out_id,0,:,:].cpu().detach().numpy()
                mae = np.mean(np.abs(a - true))
                img_name = os.path.split(data_paths[out_id])[-1].split('.')[0]
                p.map(os.remove, glob.glob(os.path.join(args.root,args.model,"*_"+img_name+"*.jpg")))
                imsave(os.path.join(args.root,args.model,'pred_'+img_name+'_'+str(cnt)+'_MAE'+str(round(mae,3))+'_dice'+str(round(dice,4))+'.jpg'), a*255)
                y = target[out_id,0,:,:].cpu().detach().numpy()
                imsave(os.path.join(args.root,args.model,'true_'+img_name+'_'+str(cnt)+'.jpg'), y*255)
                
                background = data[out_id,0,:,:].cpu().detach().numpy()
                overlay, orig_size_added_image = save_overlay(background, y*255, a*255, data_paths[out_id], target_paths[out_id], json_paths[out_id], args.margin)
                imsave(os.path.join(args.root,args.model,'overlay_'+img_name+'_'+str(cnt)+'_MAE'+str(round(mae,3))+'_dice'+str(round(dice,4))+'.jpg'), overlay)
                imsave(os.path.join(args.root,args.model,'orig_overlay_'+img_name+'_'+str(cnt)+'_MAE'+str(round(mae,3))+'_dice'+str(round(dice,4))+'.jpg'), orig_size_added_image)
                cnt += 1
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
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
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )
    print('performances per each case: ', perf_accumulator)
    torch.cuda.synchronize()
    print('Time is : ', start.elapsed_time(end))
    return np.mean(perf_accumulator), np.std(perf_accumulator)

def batch_mean_and_sd(files, files2):
    
    mean = np.array([0.])
    stdTemp = np.array([0.])
    std = np.array([0.])
    
    numSamples = len(files)
    
    for i in range(numSamples):
        im = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.
        
        # for j in range(3):
        mean += np.mean(im[:,:])
    
    # numSamples2 = len(files2)
    # for i in range(numSamples2):
    #     im = cv2.imread(str(files2[i]), cv2.IMREAD_GRAYSCALE)
    #     # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #     im = im.astype(float) / 255.
        
    #     # for j in range(3):
    #     mean += np.mean(im[:,:])
        
    mean = (mean/(numSamples))
    
    print(mean) #0.51775225 0.47745317 0.35173384]

    for i in range(numSamples):
        im = cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        
        im = im.astype(float) / 255.
        # for j in range(3):
        stdTemp += ((im[:,:] - mean)**2).sum()/(im.shape[0]*im.shape[1])
 
    std = np.sqrt(stdTemp/numSamples)
 
    print(std) #[0.28075549 0.25811162 0.28913701]


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
        img_path_b = args.root+"/Data/HIFU_data/images/before/*"
        img_path_a = args.root+"/Data/HIFU_data/images/after/"
        mask_path = args.root + "/Data/HIFU_data/masks/"
        input_paths = sorted(glob.glob(img_path_b))

        test_img_path_b = args.root+"/Data/HIFU_data/test_data_asam/images/before/*"
        test_img_path_a = args.root+"/Data/HIFU_data/test_data_asam/images/after/"
        test_mask_path = args.root + "/Data/HIFU_data/test_data_asam/masks/"
        test_input_paths = sorted(glob.glob(test_img_path_b))
        
        
    train_dataloader, test_loader = dataloaders.get_dataloaders(
        input_paths, img_path_a, mask_path, test_input_paths, test_img_path_a, test_mask_path, batch_size=args.batch_size, img_size=args.img_size,
        margin=args.margin
    )
    # mean, std = batch_mean_and_sd(input_paths, img_path_a)
    # print("mean and std: \n", mean, std)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    # Dice_loss = DiceLoss(1)
    perf = performance_metrics.DiceScore()
    if args.model == 'transunet':                             #=============== TransUnet
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 1
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load('model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
        print('\n transunet is set...\n')

    elif args.model == 'swinunet':
        print('\n swinunet is set...\n')
        from Models.vision_transformer import SwinUnet as ViT_seg
        from config import get_config
        config = get_config(args)
        model = ViT_seg(config, img_size=args.img_size, num_classes=1).cuda()
        model.load_from(config)
    else:
        model = models.FCBFormer()
        print('\n FCBFormer is set...\n')
        try:
            if os.path.exists(os.path.join(args.root, "Trained models/FCBFormer_Hifu.pt")):
                model.load_state_dict(torch.load(os.path.join(args.root,"Trained models/FCBFormer_Hifu.pt")))
                model.eval()
                
                print('\n load successfully from ./Trained models/FCBFormer_Hifu.pt \n')
        except:
            pass
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    minimizer = ASAM(optimizer, model, rho=args.rho, eta=args.eta)

    return (
        device,
        train_dataloader,
        
        test_loader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
        minimizer,
    )


def train(args):
    (
        device,
        train_dataloader,
        
        test_dataloader,
        Dice_loss,
        BCE_loss,
        perf,
        model,
        optimizer,
        minimizer,
    ) = build(args)

    if not os.path.exists(os.path.join(args.root, "Trained models")):
        os.makedirs(os.path.join(args.root, "Trained models"))

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(
                model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss,minimizer
            )
            
            test_measure_mean, test_measure_std = test(
                model, device, test_dataloader, epoch, perf, args
            )
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        print('\n*** prev_best_test: ', prev_best_test,'\n')
        if prev_best_test == None or test_measure_mean > prev_best_test:
            if test_measure_mean > 0.8:
                print("Saving...")
                test(model, device, test_dataloader, epoch, perf, args, do_save=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict()
                        if args.mgpu == "false"
                        else model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "test_measure_mean": test_measure_mean,
                        "test_measure_std": test_measure_std,
                    },
                    "Trained models/FCBFormer_" + args.dataset + '_'+ args.run_name+ ".pt",
                )
            prev_best_test = test_measure_mean
        
    # test_measure_mean, test_measure_std = test(model, device, test_dataloader, epoch, perf, do_save=True)
    print('Finished...!')


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default='Hifu', choices=["hifu", "Kvasir", "CVC"])
    parser.add_argument("--model", type=str, default='swinunet', choices=["fcbformer", "transunet", "swinunet"])
    parser.add_argument("--data-root", type=str, dest="root")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--margin", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4, dest="lr")
    parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-5, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="true", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
    parser.add_argument("--rho", default=0.01, type=float, help="Rho for ASAM.")
    parser.add_argument("--eta", default=0, type=float, help="Eta for ASAM.")
    parser.add_argument('--cfg', type=str, default='/home/hossein/projects/hifu/FCBFormer_V1/configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                                'full: cache all data, '
                                'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--run_name', default='0', type=str, help='a name for saving the model')
    return parser


def main(run_name=None):
    parser = get_args()
    args = parser.parse_args()
    args.root = '/home/hossein/projects/hifu/FCBFormer_V1'
    if run_name is not None:
        args.run_name = run_name
    train(args)


if __name__ == "__main__":
    main()
