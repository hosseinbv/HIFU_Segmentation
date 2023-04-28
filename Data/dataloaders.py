import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
import glob
from Data.dataset import SegDataset


def split_ids(len_ids):
    train_size = int(round((90 / 100) * len_ids))
    # valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=40,
    )

    # train_indices, val_indices = train_test_split(
    #     train_indices, test_size=test_size, random_state=42
    # )
    print('Test indices are: ', test_indices)
    return train_indices, test_indices


def get_dataloaders(input_paths_b, input_paths_a, target_paths, test_input_paths_b, test_input_paths_a, test_target_paths, batch_size=None, test_only=False,
                    img_size=224, margin=25):

    if test_only:
        transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize((0.04942362), (0.13466596)),
        ]
    )

        transform_target = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((img_size, img_size))]
        )
        test_dataset = SegDataset(
            input_paths_b=test_input_paths_b,
            input_paths_a=test_input_paths_a,
            target_paths=test_target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
            margin = margin
        )
        test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True
        )
        
        return None, test_dataloader 
    
    else:
        transform_input4train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.Normalize((0.04942362), (0.13466596)),
            ]
        )

        transform_input4test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size), antialias=True),
                transforms.Normalize((0.04942362), (0.13466596)),
            ]
        )

        transform_target = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((img_size, img_size), antialias=True)]
        )

        train_dataset = SegDataset(
            input_paths_b=input_paths_b,
            input_paths_a=input_paths_a,
            target_paths=target_paths,
            transform_input=transform_input4train,
            transform_target=transform_target,
            hflip=False,
            vflip=False,
            affine=False,
            margin = margin
        )

        test_dataset = SegDataset(
            input_paths_b=test_input_paths_b,
            input_paths_a=test_input_paths_a,
            target_paths=test_target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
            margin = margin
        )

        # val_dataset = SegDataset(
        #     input_paths_b=input_paths_b,
        #     input_paths_a=input_paths_a,
        #     target_paths=target_paths,
        #     transform_input=transform_input4test,
        #     transform_target=transform_target,
        # )
        
        # train_indices, test_indices = split_ids(len(input_paths_b))

        # train_dataset = data.Subset(train_dataset, train_indices)
        # # val_dataset = data.Subset(val_dataset, val_indices)
        # test_dataset = data.Subset(test_dataset, test_indices)

        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            persistent_workers=True
        )

        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            persistent_workers=True
        )

        # val_dataloader = data.DataLoader(
        #     dataset=val_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=1,
        #     persistent_workers=True
        # )

    return train_dataloader, test_dataloader



