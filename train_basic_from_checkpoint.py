"""
Main training script to train a Basic UNETR (from checkpoint) with MSD-BraTS
"""
# pyright: reportPrivateImportUsage=false
from csv import writer
import logging, os, torch
from pyrsistent import b
from typing import Union

from monai.data.dataloader import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss

from torch.backends import cudnn

import data
from configs import TrainingConfig
from torchmanager_monai import Manager, metrics
from networks import SelfDistillUNETRWithDictOutput as SelfDistilUNETR
from networks import SelfDistillnnUNetWithDictOutput as SelfDistilnnUNet
from torchmanager import callbacks, losses


if __name__ == "__main__":
    # get configurations
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize checkpoint and data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    best_ckpt_dir = os.path.join(config.experiment_dir, "best.model")
    last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")

    # load dataset - Load MSD-BraTS Data
    in_channels = 4
    training_dataset, validation_dataset, _, num_classes = data.load_msd(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose) 
        
    training_dataset = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    validation_dataset = DataLoader(validation_dataset, batch_size=1, collate_fn=pad_list_data_collate)

    ##########################################################################################################
    ## Load the UNETR model from checkpoint
    # Select Dataset
    dataset = "MSD_BraTS"

    # Set Modality
    modality = "multimodalMR" # for MSD_BraTS

    # Set Testing Type
    mode = "Training"

    # Set Pre-trained Model Architecture
    arch = "UNETR"
    
    best_model_path = os.path.join("experiments", "multimodalMR_MSD_BraTS_UNETR_Basic.exp/best.model")   

    # Test model on validation set with last saved best model checkpoint

    manager = Manager.from_checkpoint(best_model_path) # type:ignore

    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=torch.device("cuda:1"), use_multi_gpus=False, show_verbose=True)
    print("Results of the best model (from checkpoint) on the Validation Set ...")
    print(".......")
    print(summary)
    
    ##########################################################################################################

    print("Loading Best Model from checkpoint (for retraining)...")
    model = manager.model

    print(f"{mode} with the Basic {arch} architecture on {dataset} dataset with {modality} modality (Training from last saved checkpoint)....")
    
    ##########################################################################################################
    
    # initialize optimizer, loss, metrics, and post processing
    # lr=0.0001 # for MSD-BraTS to continue from checkpoint 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Initilialize Loss Functions
    loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

    ## Basic UNETR/nnUNet ONLY
    loss_dice = losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True), target="out") # out_main and GT labels

    loss_fn = {
        "dice": loss_dice
    }
   
    # Initialize Metrics
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

    hd_fn = metrics.CumulativeIterationMetric(HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="none", get_not_nans=False), target="out")

    metric_fns: dict[str, metrics.Metric] = {
        "val_dice": dice_fn,
        "val_hd": hd_fn
        } 

    post_labels = data.transforms.AsDiscrete(to_onehot=num_classes)
    post_predicts = data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes)

    # compile manager
    manager = Manager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size) # type: ignore

    ## All callbacks defined below
    # initialize callbacks
    tensorboard_callback = callbacks.TensorBoard(data_dir)

    last_ckpt_callback = callbacks.LastCheckpoint(manager, last_ckpt_dir)
    besti_ckpt_callback = callbacks.BestCheckpoint("dice", manager, best_ckpt_dir)

    # Final callbacks list
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, besti_ckpt_callback, last_ckpt_callback]

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test with last model
    model = manager.model
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

    # save and test with best model on validation dataset  
    # manager = Manager.from_checkpoint("experiments/CT_MMWHS_UNETR_Basic.exp/best.model") # for Basic Unetr
    
    # manager = Manager.from_checkpoint("experiments/multimodalMR_MSD_BraTS_UNETR_Basic.exp/best.model") # for Basic 
    manager = Manager.from_checkpoint("experiments/multimodalMR_MSD_BraTS_UNETR_Basic_from_checkpoint_294_epochs.exp/best.model") # for Basic 

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

