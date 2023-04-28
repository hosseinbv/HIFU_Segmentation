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

random.seed(12)
matplotlib.use('tkagg')
torch.manual_seed(0)
np.random.seed(0)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def save_overlay(background, true_mask, pred_mask, mean=0.13466596,std=0.04942362):
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

    

    added_image = cv2.addWeighted(colored_true_mask,0.7,colored_pred_mask,0.3,0)

    return added_image


@torch.no_grad()
def test(model, device, test_loader, perf_measure, do_save=False, model_name=None):
    t = time.time()
    model.eval()
    perf_accumulator = []
    cnt = 0
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        if do_save:
            for out_id in range(output.size()[0]):
                probs = torch.sigmoid(output[out_id,0,:,:])
                a = probs.cpu().detach().numpy()
                a[a>=0.7] = 1
                a[a<0.7] = 0
                imsave(os.path.join(model_name,'pred_'+str(cnt)+'.jpg'), a*255)
                y = target[out_id,0,:,:].cpu().detach().numpy()
                imsave(os.path.join(model_name, 'true_'+str(cnt)+'.jpg'), y*255)
                cnt += 1
                background = data[out_id,0,:,:].cpu().detach().numpy()
                overlay = save_overlay(background, y*255, a*255)
                imsave(os.path.join(model_name, 'overlay_'+str(cnt)+'.jpg'), overlay)
        perf_accumulator.append(perf_measure(output, target).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
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
                "\rTest   [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )
    print('performances per each case: ', perf_accumulator)
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

    if args.dataset.lower() == "hifu":
        test_img_path_b = args.root+"test_data_asam/images/before/*"
        test_img_path_a = args.root+"test_data_asam/images/after/"
        test_mask_path = args.root + "test_data_asam/masks/"
        test_input_paths = sorted(glob.glob(test_img_path_b))
        
        
    _, test_loader = dataloaders.get_dataloaders(
        None, None, None, test_input_paths, test_img_path_a, test_mask_path, batch_size=args.batch_size, img_size=args.img_size, test_only=True
    )
    # mean, std = batch_mean_and_sd(input_paths, img_path_a)
    # print("mean and std: \n", mean, std)
    # Dice_loss = losses.SoftDiceLoss()
    # BCE_loss = nn.BCELoss()
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
    else:
        model = models.FCBFormer()
        print('\n FCBFormer is set...\n')
        try:
            model_path = "./Trained models/FCBFormer_Hifu.pt"
            if os.path.exists(model_path):
                ans = input('Do you want to load this model: ', model_path, ' ? (y/n)')
                if ans=='y':
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    print('\n model loaded successfully... \n')
                else:
                    model_path = input('Enter the model path:')
                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path))
                        model.eval()
                        print('\n model loaded successfully...\n')
                    else:
                        raise ValueError('The model path does not exist...')
        except Exception as e:
            print(e)
            raise ValueError('The model path does not exist or there is an error')

    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    # minimizer = ASAM(optimizer, model, rho=args.rho, eta=args.eta)

    return (
        device,
        test_loader,
        perf,
        model
    )


def predict(args):
    (
        device,
        test_loader,
        perf,
        model
    ) = build(args)

    test(model, device, test_loader, perf, do_save=True)

    print('Finished...!')


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, default='Hifu', choices=["hifu", "Kvasir", "CVC"])
    parser.add_argument("--model", type=str, default='fcbformer', choices=["fcbformer", "transunet"])
    parser.add_argument("--data-root", type=str, dest="root")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6, dest="lr")
    parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
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
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
    parser.add_argument("--eta", default=0.1, type=float, help="Eta for ASAM.")
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    args.root = 'Data/HIFU_data/'

    predict(args)


if __name__ == "__main__":
    main()























# import torch
# import os
# import argparse
# import time
# import numpy as np
# import glob
# import cv2

# import torch
# import torch.nn as nn

# from Data import dataloaders
# from Models import models
# from Metrics import performance_metrics


# def build(args):
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")

#     if args.test_dataset == "Kvasir":
#         img_path = args.root + "images/*"
#         input_paths = sorted(glob.glob(img_path))
#         depth_path = args.root + "masks/*"
#         target_paths = sorted(glob.glob(depth_path))
#     elif args.test_dataset == "CVC":
#         img_path = args.root + "Original/*"
#         input_paths = sorted(glob.glob(img_path))
#         depth_path = args.root + "Ground Truth/*"
#         target_paths = sorted(glob.glob(depth_path))
#     _, test_dataloader, _ = dataloaders.get_dataloaders(
#         input_paths, target_paths, batch_size=1
#     )

#     _, test_indices, _ = dataloaders.split_ids(len(target_paths))
#     target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]

#     perf = performance_metrics.DiceScore()

#     model = models.FCBFormer()

#     state_dict = torch.load(
#         "./Trained models/FCBFormer_{}.pt".format(args.train_dataset)
#     )
#     model.load_state_dict(state_dict["model_state_dict"])

#     model.to(device)

#     return device, test_dataloader, perf, model, target_paths


# @torch.no_grad()
# def predict(args):
#     device, test_dataloader, perf_measure, model, target_paths = build(args)

#     if not os.path.exists("./Predictions"):
#         os.makedirs("./Predictions")
#     if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
#         os.makedirs("./Predictions/Trained on {}".format(args.train_dataset))
#     if not os.path.exists(
#         "./Predictions/Trained on {}/Tested on {}".format(
#             args.train_dataset, args.test_dataset
#         )
#     ):
#         os.makedirs(
#             "./Predictions/Trained on {}/Tested on {}".format(
#                 args.train_dataset, args.test_dataset
#             )
#         )

#     t = time.time()
#     model.eval()
#     perf_accumulator = []
#     for i, (data, target) in enumerate(test_dataloader):
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         perf_accumulator.append(perf_measure(output, target).item())
#         predicted_map = np.array(output.cpu())
#         predicted_map = np.squeeze(predicted_map)
#         predicted_map = predicted_map > 0
#         cv2.imwrite(
#             "./Predictions/Trained on {}/Tested on {}/{}".format(
#                 args.train_dataset, args.test_dataset, os.path.basename(target_paths[i])
#             ),
#             predicted_map * 255,
#         )
#         if i + 1 < len(test_dataloader):
#             print(
#                 "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
#                     i + 1,
#                     len(test_dataloader),
#                     100.0 * (i + 1) / len(test_dataloader),
#                     np.mean(perf_accumulator),
#                     time.time() - t,
#                 ),
#                 end="",
#             )
#         else:
#             print(
#                 "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
#                     i + 1,
#                     len(test_dataloader),
#                     100.0 * (i + 1) / len(test_dataloader),
#                     np.mean(perf_accumulator),
#                     time.time() - t,
#                 )
#             )


# def get_args():
#     parser = argparse.ArgumentParser(
#         description="Make predictions on specified dataset"
#     )
#     parser.add_argument(
#         "--train-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
#     )
#     parser.add_argument(
#         "--test-dataset", type=str, required=True, choices=["Kvasir", "CVC"]
#     )
#     parser.add_argument("--data-root", type=str, required=True, dest="root")

#     return parser.parse_args()


# def main():
#     args = get_args()
#     predict(args)


# if __name__ == "__main__":
#     main()

