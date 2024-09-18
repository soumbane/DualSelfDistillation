# pyright: reportPrivateImportUsage=false
import os
import glob
import random
from typing import Tuple

from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Orientationd,
    ScaleIntensityd,
    SpatialPadd,
    FgBgToIndicesd,
    RandCropByPosNegLabeld,
    DeleteItemsd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord
)

def load_hippocampus_data(root_dir: str, img_size: Tuple[int, ...] = (32, 32, 32), num_samples: int = 8, train_split: int = 60,show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, CacheDataset, int]:
    """
    Load Hippocampus MRI dataset.

    - Parameters:
        - root_dir (str): Base directory containing 'Task04_Hippocampus'.
        - img_size (Tuple[int, ...], optional): Spatial size for cropping. Defaults to (32, 32, 32).
        - num_samples (int, optional): Number of samples for random cropping. Defaults to 8.
        - train_split (int): Number of test patients.
        - show_verbose (bool, optional): Flag to show loading progress. Defaults to False.
    - Returns:
        - Returns: A `tuple` of training `DataLoader`, validation `DataLoader`, and the number of classes in `int`
    """
    base_dir = os.path.join(root_dir, "Task04_Hippocampus")
    num_classes = 3  

    # load images and labels
    train_images = sorted(
        glob.glob(os.path.join(base_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(base_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    # Assign Testing Set First (Last 24 patients)
    test_data_dicts = data_dicts[-train_split:] # Last 60 patients for testing
    # print(f'MSD-BraTS Test length: {len(test_data_dicts)}')

    train_data_dicts_temp = data_dicts[:-train_split] # First 200 patients for train/validation
    # print(f'Train/val MSD-BraTS length: {len(train_data_dicts_temp)}')

    # randomly shuffle the dataset
    seed = 0
    random.Random(seed).shuffle(train_data_dicts_temp)

    # Assign Train/Validation Sets (170 for training & 30 for validation)
    val_split = 30
    train_data_dicts, val_data_dicts = train_data_dicts_temp[:-val_split], train_data_dicts_temp[-val_split:] 

    # Define training transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode="edge"),
        FgBgToIndicesd(keys=["label"], image_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,
            pos=1,
            neg=0.1,
            num_samples=num_samples,
            fg_indices_key="label_fg_indices",
            bg_indices_key="label_bg_indices",
        ),
        DeleteItemsd(keys=["label_fg_indices", "label_bg_indices"]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=1.0),
        ToTensord(keys=["image", "label"]),
    ])

    # Define validation transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ToTensord(keys=["image", "label"]),
    ])

    # Test transforms (same as Validation Transforms)
    test_transforms = val_transforms

    # Create datasets    
    train_set = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_num=2,
        cache_rate=1.0,
        num_workers=8,
        progress=show_verbose
    )
    val_set = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        num_workers=8,
        progress=show_verbose
    )

    test_set = CacheDataset(
        data=test_data_dicts,
        transform=test_transforms,
        num_workers=8,
        progress=show_verbose
    )    
    
    return train_set, val_set, test_set, num_classes


# Example Usage
if __name__ == "__main__":    
    
    # Parameters
    root_directory = "/home/share/Data/"

    train_set, val_set, test_set, num_classes = load_hippocampus_data(root_directory, img_size=(32, 32, 32))

    print(f'Length of Train set: {len(train_set)}')
    print(f'Length of Val set: {len(val_set)}')
    print(f'Length of Test set: {len(test_set)}')
    print(f'Number of classes: {num_classes}')
