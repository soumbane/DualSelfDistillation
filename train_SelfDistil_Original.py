"""
Main training script to train a UNETR/nnUNet with Original Self Distillation on MMWHS Challenge dataset for CT/MR and MSD-BraTS
"""
# pyright: reportPrivateImportUsage=false
from csv import writer
import logging, os, torch
from pyrsistent import b
from typing import Union
from torch.nn import MSELoss, KLDivLoss

from monai.data.dataloader import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss

from torch.backends import cudnn

import data
from configs import TrainingConfig
from torchmanager_monai import Manager, metrics
from networks import SelfDistillUNETRWithDictOutput as SelfDistilUNETR
from networks import SelfDistillnnUNetWithDictOutput as SelfDistilnnUNet
from networks import SelfDistillSwinUNETRWithDictOutput as SelfDistilSwinUNETR
from torchmanager import callbacks, losses
from monai.utils import UpsampleMode, InterpolateMode

from loss_functions import Self_Distillation_Loss_Dice, PixelWiseKLDiv, Self_Distillation_Loss_KL

from torchmanager_core import random
from torch.backends import cudnn
from utils import count_parameters

# initialization
# seed = 100
# random.freeze_seed(seed)
# cudnn.benchmark = False 
# cudnn.deterministic = True  


if __name__ == "__main__":
    # get configurations
    config = TrainingConfig.from_arguments()
    cudnn.benchmark = True
    if config.show_verbose: config.show_settings()

    # initialize checkpoint and data dirs
    data_dir = os.path.join(config.experiment_dir, "data")
    best_ckpt_dir = os.path.join(config.experiment_dir, "best.model")
    last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")
    
    # load dataset - Load MMWHS Challenge Data
    # in_channels = 1
    # training_dataset, validation_dataset, num_classes = data.load_challenge(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose) # type:ignore

    # load dataset - Load MSD-BraTS Data
    in_channels = 4
    training_dataset, validation_dataset, _, num_classes = data.load_msd(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose) 
        
    training_dataset = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    validation_dataset = DataLoader(validation_dataset, batch_size=1, collate_fn=pad_list_data_collate)

    ##########################################################################################################
    ## Initialize the UNETR model
    model = SelfDistilUNETR(in_channels, num_classes, img_size=config.img_size, self_distillation=True, use_feature_maps=False, mode=UpsampleMode.DECONV, interp_mode=InterpolateMode.BILINEAR, multiple_upsample=True, feature_size=64, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)  # for BraTS

    ##########################################################################################################

    ## Count model parameters
    print(f'The total number of model parameter is: {count_parameters(model)}')

    # initialize optimizer, loss, metrics, and post processing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # lr=0.0001 # for MSD-BraTS

    # Initilialize Loss Functions
    loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

    # Hyper-parameters for Self Distillation
    alpha_KL: float = 1.0  # weight of KL Div Loss Term (for UNETR/nnUNet/SwinUNETR)

    temperature: int = 3 # divided by temperature (T) to smooth logits before softmax (required for KL Div)
    
    ## For Multiple Losses
    # Deep Supervision or Self Distillation from GT Labels to decoders 
     
    # For DiceCE Loss between GT Labels(target="out") [target] and softmax(out_main) [input]
    # Dice Loss Only between GT Labels(target="out") [target] and softmax(out_dec1,out_dec2,out_dec3,out_dec4) [input]
    loss_dice = Self_Distillation_Loss_Dice([
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_main and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec4 and GT labels 
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec3 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec2 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec1 and GT labels
        ], weight=1.0, target="out", learn_weights=False)
    
    # The weights of the above Self_Distillation_Loss_Dice can be set to be different instead of constant - it has to be set in a list with increasing order [0.4,0.6,0.8,1.0] for [out_dec4, out_dec3, out_dec2, out_dec1] respectively; the weight for out_main is always 1.0

    # Self Distillation from deepest encoder/decoder (out_enc4/out_dec1): Teacher (T), to shallower encoders/decoders (out_enc2/out_dec2,out_enc3/out_dec3,out_dec4/out_enc1): Students (S)  
    # For KL Div between softmax(out_dec1/out_enc4) [target] and log_softmax((out_dec2/out_enc3,out_dec3/out_enc2,out_dec4/out_enc1)) [input]
    loss_KL = Self_Distillation_Loss_KL([ 
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec4/out_enc1 (S) & out_dec1/out_enc4 (T)
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec3/out_enc2 (S) & out_dec1/out_enc4 (T)
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec2/out_enc3 (S) & out_dec1/out_enc4 (T)
    ], weight=alpha_KL, include_background=True, T=temperature, learn_weights=False) # pass the entire dict NOT just "out"

    # The weights of the above Self_Distillation_Loss_KL can be set to be different instead of constant - it has to be set in a list with increasing order [0.4,0.6,0.8,1.0]

    # For Deep Supervision and Self-Distillation between Encoders and Decoders
    loss_fn = {
        "dice": loss_dice,
        "KL": loss_KL
    }

    # For Deep Supervision Only
    # loss_fn = {
    #     "dice": loss_dice
    # }
   
    # Initialize Metrics
    dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

    hd_fn = metrics.CumulativeIterationMetric(HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="none", get_not_nans=False), target="out")

    msd_fn = metrics.CumulativeIterationMetric(SurfaceDistanceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")

    metric_fns: dict[str, metrics.Metric] = {
        "val_dice": dice_fn,
        "val_hd": hd_fn,
        "val_msd": msd_fn,
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
    
    ##############################################################################################################

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
    manager = Manager.from_checkpoint("experiments/multimodalMR_MSD_BraTS_LargeUNETR_SelfDist.exp/best.model") # for Self Distillation Original

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

