"""
Main training script to train a UNETR (from checkpoint) with Original Self Distillation on MSD-BraTS
"""
# pyright: reportPrivateImportUsage=false
from csv import writer
import logging, os, torch
from pyrsistent import b
from typing import Union
from torch.nn import MSELoss, KLDivLoss

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

from loss_functions import Self_Distillation_Loss_Dice, Self_Distillation_Loss_CE, PixelWiseKLDiv, Self_Distillation_Loss_KL, Self_Distillation_Loss_L2


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
    
    best_model_path = os.path.join("experiments", "multimodalMR_MSD_BraTS_UNETR_SelfDist_Original.exp/best.model")   

    # Test model on validation set with last saved best model checkpoint

    manager = Manager.from_checkpoint(best_model_path) # type:ignore

    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=torch.device("cuda:0"), use_multi_gpus=False, show_verbose=True)
    print("Results of the best model (from checkpoint) on the Validation Set ...")
    print(".......")
    print(summary)
    
    ##########################################################################################################

    print("Loading Best Model from checkpoint (for retraining)...")
    model = manager.model

    print(f"{mode} with the Basic {arch} architecture with Dual self-distillation on {dataset} dataset with {modality} modality (Training from last saved checkpoint)....")
    
    ##########################################################################################################
    
    # initialize optimizer, loss, metrics, and post processing
    # lr=0.0001 # for MSD-BraTS to continue from checkpoint 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  

    # Initilialize Loss Functions
    loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

    # Initial weights for cross-entropy (CE) term (L_CE)
    eta_ce_init = 0.1086 # at the end of 290 epochs

    # Hyper-parameters for Self Distillation
    alpha_KL: float = 1.0  # weight of KL Div Loss Term (for UNETR/nnUNet)

    # lambda_feat: float = 0.0001  # weight of L2 Loss Term between feature maps
    temperature: int = 3 # divided by temperature (T) to smooth logits before softmax (required for KL Div)
    
    ## For Multiple Losses
    # Deep Supervision or Self Distillation from GT Labels to decoders  
     
    # For DiceCE Loss between GT Labels(target="out") [target] and softmax(out_main) [input]
    # Dice Loss Only between GT Labels(target="out") [target] and softmax(out_dec1,out_dec2,out_dec3,out_dec4) [input]
    loss_dice = Self_Distillation_Loss_Dice([
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_main and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.0)), #out_dec4 and GT labels 
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.0)), #out_dec3 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.0)), #out_dec2 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.0)), #out_dec1 and GT labels
        ], weight=1.0, target="out")

    # CE Loss Only between GT Labels(target="out") [target] and softmax(out_dec1,out_dec2,out_dec3,out_dec4) [input]
    loss_dice_ce = Self_Distillation_Loss_CE([        
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.0, lambda_ce=1.0)), #out_dec4 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.0, lambda_ce=1.0)), #out_dec3 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.0, lambda_ce=1.0)), #out_dec2 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.0, lambda_ce=1.0)), #out_dec1 and GT labels
        ], weight=eta_ce_init, target="out")

    # Self Distillation from deepest encoder/decoder (out_enc4/out_dec1): Teacher (T), to shallower encoders/decoders (out_enc2/out_dec2,out_enc3/out_dec3,out_dec4/out_enc1): Students (S)  
    # For KL Div between softmax(out_dec1/out_enc4) [target] and log_softmax((out_dec2/out_enc3,out_dec3/out_enc2,out_dec4/out_enc1)) [input]
    loss_KL = Self_Distillation_Loss_KL([ 
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec4/out_enc1 (S) & out_dec1/out_enc4 (T)
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec3/out_enc2 (S) & out_dec1/out_enc4 (T)
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec2/out_enc3 (S) & out_dec1/out_enc4 (T)
    ], weight=alpha_KL, include_background=True, T=temperature) # pass the entire dict NOT just "out"

    loss_fn = {
        "dice": loss_dice,
        "ce": loss_dice_ce,
        "KL": loss_KL
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
    
    ##############################################################################################################
    # Decease cross-entropy weight by this amount every epoch (To reduce L_CE for L_DCE term)
    decr_eta_ce = 9.7818e-4 # decrease from 0.1086 to 0.001 for 110 epochs (total 400 epochs) (for UNETR with MSD-BraTS from checkpoint)

    def getw_ce(e):
        return (loss_fn["ce"].weight - decr_eta_ce)

    weights_callback_ce = callbacks.LambdaDynamicWeight(getw_ce, loss_fn["ce"], writer=tensorboard_callback.writer, name='ce_weight')

    ##############################################################################################################

    # Final callbacks list
    # callbacks_list: list[callbacks.Callback] = [tensorboard_callback, besti_ckpt_callback, last_ckpt_callback, lr_scheduler_callback, weights_callback_ce]
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, besti_ckpt_callback, last_ckpt_callback, weights_callback_ce]

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test with last model
    model = manager.model
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

    # save and test with best model on validation dataset  
    # manager = Manager.from_checkpoint("experiments/CT_MMWHS_UNETR_SelfDist_Original.exp/best.model") # for Self Distillation Original
    # manager = Manager.from_checkpoint("experiments/multimodalMR_MSD_BraTS_UNETR_SelfDist_Original.exp/best.model") # for Self Distillation Original
    manager = Manager.from_checkpoint("experiments/multimodalMR_MSD_BraTS_UNETR_SelfDist_Original_from_checkpoint_290_epochs.exp/best.model") # for Self Distillation Original

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

