'''Testing script for testing with pre-trained models for both CT and MR of MMWHS Dataset'''
# pyright: reportPrivateImportUsage=false
import os
import torch
import numpy as np
from torchmanager import losses
from torchmanager_monai import Manager, metrics

from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.data.dataloader import DataLoader
from monai.data.utils import pad_list_data_collate
import data

from monai.transforms import LoadImage, SaveImage

## Select Dataset
dataset = "MMWHS"
# dataset = "MSD_BraTS"

## Set Modality
modality = "CT" # for MMWHS: Please set scale_intensity_ranged = True for MMWHS CT in data loading file 
# modality = "MR" # for MMWHS: Please set scale_intensity_ranged = False for MMWHS MR in data loading file
# modality = "multimodalMR" # for MSD_BraTS

## Set Testing Type
mode = "validation"
# mode = "testing"

## Set Pre-trained Model Type
# train_type = "Basic"
# train_type = "Deep_Super"
train_type = "Self_Distill_Original"
# train_type = "Self_Distill_DistMaps"

## Set Pre-trained Model Architecture
arch = "UNETR"
# arch = "SwinUNETR"
# arch = "nnUnet"

## Fold number for MMWHS dataset ONLY
fold_no = "4"
fold = "Fold_" + fold_no

## choose best or last model
# saved_model = "last"
saved_model = "best"

device = "cuda:0"

## Root path for experiments and pre-trained models
root = "/home/neil/Lab_work/Medical_Image_Segmentation"

if dataset == "MMWHS":
    data_dir = "/home/share/Data/Data_MMWHS_All/" + modality  # for MMWHS
    img_size = (96,96,96) # for MMWHS
elif dataset == "MSD_BraTS":
    data_dir = "/home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS"  # for MSD_BraTS
    img_size = (128,128,128) # for MSD_BraTS
else:
    raise NameError("Dataset not found - Only defined for MMWHS and MSD-BraTS.")

## UNETR Models on MMWHS Dataset
## Important: Please set scale_intensity_ranged = True for CT and scale_intensity_ranged = False for MR in both data loading files challenge.py and challenge_dist_map.py. These are the preprocessing steps needed for inference (validation and testing)

if train_type == "Basic":

    best_model_path = os.path.join(root, "DSD_experiments_TMI", "pretrained_MMWHS_UNETR_CT_models_final", "Fold_4", "CT_MMWHS_UNETR_Basic_Fold4.exp/last.model")

    # best_model_path = os.path.join(root, "DSD_experiments_TMI", "pretrained_" + dataset + "_" + arch + "_" + modality + "_models_final", fold, modality + "_" + dataset + "_" + arch + "_Basic_Fold" + fold_no + ".exp/" + saved_model + ".model")    

    # best_model_path = os.path.join(root, "DSD_experiments_TMI", "pretrained_MSD_BraTS_SwinUNETR_multimodalMR_models_final", "multimodalMR_MSD_BraTS_SwinUNETR_Basic.exp/last.model")

    print(f"{mode} with the Basic {arch} architecture on {dataset} dataset with {modality} modality with {saved_model} saved model on fold {fold_no}....")
    # print(f"{mode} with the Basic {arch} architecture on {dataset} dataset with {modality} modality with {saved_model} saved model ....")

elif train_type == "Deep_Super":
    best_model_path = os.path.join("experiments", "pretrained_" + dataset + "_" + arch + "_" + modality + "_models_final", modality + "_" + dataset + "_" + arch + "_DeepSuperOnly.exp/best.model")
    # best_model_path = os.path.join("experiments", "CT_MMWHS_UNETR_DeepSuperOnly.exp/best.model")

    print(f"{mode} with the Basic {arch} architecture with Deep Supervision on {dataset} dataset with {modality} modality ....")

elif train_type == "Self_Distill_Original":

    best_model_path = os.path.join(root, "DSD_experiments_TMI", "pretrained_MMWHS_UNETR_CT_models_final", "Fold_4", "CT_MMWHS_UNETR_SelfDist_Original_Fold4.exp/last.model")

    # best_model_path = os.path.join(root, "DSD_experiments_TMI", "pretrained_" + dataset + "_" + arch + "_" + modality + "_models_final", fold, modality + "_" + dataset + "_" + arch + "_SelfDist_Original_Fold" + fold_no + ".exp/" + saved_model + ".model")

    # best_model_path = os.path.join(root, "DSD_experiments_TMI", "pretrained_MSD_BraTS_SwinUNETR_multimodalMR_models_final", "multimodalMR_MSD_BraTS_SwinUNETR_SelfDist_Original.exp/last.model")
    # best_model_path = os.path.join("experiments", "CT_MMWHS_nnUnet_SelfDist_Original_Fold1_multi_upsample_trainable.exp/last.model")

    print(f"{mode} with the {arch} architecture with Dual self-distillation on {dataset} dataset with {modality} modality with {saved_model} saved model on fold {fold_no}....")
    # print(f"{mode} with the {arch} architecture with Dual self-distillation on {dataset} dataset with {modality} modality with {saved_model} saved model ....")

elif train_type == "Self_Distill_DistMaps":
    best_model_path = os.path.join("experiments", "pretrained_" + dataset + "_" + arch + "_" + modality + "_models_final", modality + "_" + dataset + "_" + arch + "_SelfDist_DistMaps.exp/best.model")
    # best_model_path = os.path.join("experiments", "CT6_MMWHS_nnUnet_SelfDist_DistMaps.exp/best.model")

    print(f"{mode} with the Basic {arch} architecture with Dual self-distillation with shape priors on {dataset} dataset with {modality} modality ....")

else:
    raise NotImplementedError("Train Type Undefined.")


manager = Manager.from_checkpoint(best_model_path) # type:ignore

# Print weights of Loss Functions
loss_fn: losses.MultiLosses = manager.compiled_losses # type: ignore

if train_type == "Self_Distill_Original":
    kl_loss_fn: losses.Loss = loss_fn.losses[1] # type: ignore
    # print(type(kl_loss_fn), kl_loss_fn.weight)
    print(f'KL Loss function weight is: {kl_loss_fn.weight}')

elif train_type == "Self_Distill_DistMaps":
    kl_loss_fn: losses.Loss = loss_fn.losses[3] # type: ignore
    # print(type(kl_loss_fn), kl_loss_fn.weight)
    print(f'KL Loss function weight is: {kl_loss_fn.weight}')
    boundary_loss_fn: losses.Loss = loss_fn.losses[2] # type: ignore
    # print(type(kl_loss_fn), kl_loss_fn.weight)
    print(f'Boundary Loss function weight is: {boundary_loss_fn.weight}')

# Initialize Metrics
dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

hd_fn = metrics.CumulativeIterationMetric(HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="none", get_not_nans=False), target="out")

msd_fn = metrics.CumulativeIterationMetric(SurfaceDistanceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

metric_fns: dict[str, metrics.Metric] = {
        "val_dice": dice_fn,
        "val_hd": hd_fn,
        "val_msd": msd_fn,
        } 

manager.loss_fn = None
manager.metric_fns = metric_fns # type:ignore

if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
else: model = manager.model

manager.model = model
print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

# load Testing dataset - Load MMWHS Challenge Test Data (First 4 patients)
if train_type == "Basic" or train_type == "Deep_Super" or train_type == "Self_Distill_Original":
    ## Testing dataset with (image,label) for regional losses
    if dataset == "MMWHS":
        _, validation_dataset, num_classes = data.load_challenge(data_dir, img_size=img_size, train_split=4, show_verbose=True)  # for MMWHS Dataset
    elif dataset == "MSD_BraTS":
        _, validation_dataset, testing_dataset, num_classes = data.load_msd(data_dir, img_size=img_size, train_split=24, show_verbose=True)  # for MSD_BraTS Dataset 
    else:
        raise NameError("Dataset not found - Only defined for MMWHS and MSD-BraTS.")

elif train_type == "Self_Distill_DistMaps":
    ## Testing dataset with (image,label,dist_maps) for Boundary Loss
    if dataset == "MMWHS":
        _, validation_dataset, testing_dataset, num_classes = data.load_challenge_boun(data_dir, img_size=img_size, train_split=4, show_verbose=True) 
    else:
        raise NameError("Dataset not found - Only defined for MMWHS.")
else:
    raise ValueError("Train Type Undefined.")

if mode == "validation":
    testing_dataset = validation_dataset
elif mode == "testing":
    testing_dataset = testing_dataset # type:ignore
    # testing_dataset = validation_dataset
else:
    raise ValueError("Mode should be either validation or testing.")
        
test_dataset = DataLoader(testing_dataset, batch_size=1, collate_fn=pad_list_data_collate, num_workers=8, pin_memory=True) # type:ignore
if mode == "validation": print("Validation Data Loaded ...") 
if mode == "testing": print("Test Data Loaded ...") 

summary = manager.test(test_dataset, device=torch.device(device), use_multi_gpus=False, show_verbose=True)
if mode == "validation": print("Results on the Validation Set ...")
if mode == "testing": print("Results on the Test Set ...")
print(".......")
print(summary)

## Generate Model Predictions
# patient_id = 2 # Select patient for whom to generate predictions (for MMWHS-CT)
# patient_id = 3 # Select patient for whom to generate predictions (for MMWHS-CT)
patient_id = 0 # Select patient for whom to generate predictions (for MMWHS-CT)
# patient_id = 1 # Select patient for whom to generate predictions (for MMWHS-CT)

# patient_id = 22 # Select patient for whom to generate predictions (for MSD-BraTS)
# patient_id = 4 # Select patient for whom to generate predictions (for MSD-BraTS)
# patient_id = 18 # Select patient for whom to generate predictions (for MSD-BraTS)
# patient_id = 2 # Select patient for whom to generate predictions (for MSD-BraTS)
# patient_id = 5 # Select patient with worst label 2 preds

preds = manager.predict(test_dataset, device=torch.device(device), use_multi_gpus=False, show_verbose=True) # type:ignore
# print(preds[patient_id].shape)
preds_1 = preds[patient_id].squeeze(0)
# print(preds_1.shape)
preds_f = torch.argmax(preds_1, dim=0)
# print(preds_f.shape)

## Save Model Predictions
out_dir = os.path.join(root, "Predicted_Labels_TMI") 

## Define your case
if train_type == "Basic":
    case = 'basic_' + arch
elif train_type == "Deep_Super":
    case = 'deepSuper_' + arch
elif train_type == "Self_Distill_Original":
    case = 'selfDistil_' + arch
elif train_type == "Self_Distill_DistMaps":
    case = 'selfDistilDistMap_' + arch
else:
    raise NotImplementedError("Train Type Undefined.")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
saver = SaveImage(
    output_dir=out_dir, output_postfix='pred_label_'+case+'_patientID_'+str(patient_id), output_ext='.nii.gz', 
    output_dtype=np.float32, resample=False, scale=None, dtype=None, squeeze_end_dims=True, 
    data_root_dir='', separate_folder=False, print_log=False, channel_dim=None
    )
    
saver(preds_f, meta_data={'filename_or_obj':modality})

load_path = os.path.join(out_dir, modality+'_pred_label_'+case+'_patientID_'+str(patient_id)+'.nii.gz')
loader = LoadImage()

img = loader(load_path)
print(f"Loaded Image Shape: {img[0].shape}")
print("\n")

# Print the Dice/HD of all classes for all samples
print(f"Entire Val Dice Matrix (no of samples x classes):\n {manager.metric_fns['val_dice'].results.squeeze(1)}") # type:ignore
print(f"Entire Val HD Matrix (no of samples x classes):\n {manager.metric_fns['val_hd'].results.squeeze(1)}") # type:ignore
print(f"Entire Val MSD Matrix (no of samples x classes):\n {manager.metric_fns['val_msd'].results.squeeze(1)}") # type:ignore

print("\n")

# Mean of Dice/HD for all samples in validation set (all classes)
print(f"Mean Val Dice (for all classes of all samples): {manager.metric_fns['val_dice'].results.squeeze(1).mean()}") # type:ignore
print(f"Mean Val HD (for all classes of all samples): {manager.metric_fns['val_hd'].results.squeeze(1).mean()}") # type:ignore
print(f"Mean Val MSD (for all classes of all samples): {manager.metric_fns['val_msd'].results.squeeze(1).mean()}") # type:ignore
print("\n")

# Std Dev of Dice/HD for all samples in validation set (all classes)
print(f"Std of Val Dice (for all classes of all samples): {manager.metric_fns['val_dice'].results.squeeze(1).std()}") # type:ignore
print(f"Std of Val HD (for all classes of all samples): {manager.metric_fns['val_hd'].results.squeeze(1).std()}") # type:ignore
print(f"Std of Val MSD (for all classes of all samples): {manager.metric_fns['val_msd'].results.squeeze(1).std()}") # type:ignore
print("\n")

num_foreground_classes = num_classes - 1
print(f"No. of Foreground classes: {num_foreground_classes}")

for class_id in range(num_foreground_classes):
    # Mean of Dice/HD for class_id of all samples in validation set (individual classes)
    print(f"Mean Val Dice of foreground class {class_id} across all val samples: {manager.metric_fns['val_dice'].results.squeeze(1).mean(0)[class_id]}") # type:ignore
    print(f"Mean Val HD of foreground class {class_id} across all val samples:: {manager.metric_fns['val_hd'].results.squeeze(1).mean(0)[class_id]}") # type:ignore
    print(f"Mean Val MSD of foreground class {class_id} across all val samples:: {manager.metric_fns['val_msd'].results.squeeze(1).mean(0)[class_id]}") # type:ignore

    # Std of Dice/HD for class_id of all samples in validation set (individual classes)
    print(f"Std of Val Dice of foreground class {class_id} across all val samples: {manager.metric_fns['val_dice'].results.squeeze(1).std(0)[class_id]}") # type:ignore
    print(f"Std of Val HD of foreground class {class_id} across all val samples: {manager.metric_fns['val_hd'].results.squeeze(1).std(0)[class_id]}") # type:ignore
    print(f"Std of Val MSD of foreground class {class_id} across all val samples: {manager.metric_fns['val_msd'].results.squeeze(1).std(0)[class_id]}") # type:ignore
    print("\n")

