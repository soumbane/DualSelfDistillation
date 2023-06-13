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
from torchmanager import callbacks, losses
from monai.utils import UpsampleMode, InterpolateMode

from loss_functions import Self_Distillation_Loss_Dice, PixelWiseKLDiv, Self_Distillation_Loss_KL

from torchmanager_core import random
from torch.backends import cudnn
from utils import count_parameters

# initialization
seed = 100
random.freeze_seed(seed)
cudnn.benchmark = False 
cudnn.deterministic = True  


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
    in_channels = 1
    training_dataset, validation_dataset, num_classes = data.load_challenge(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose) # type:ignore

    # load dataset - Load MSD-BraTS Data
    # in_channels = 4
    # training_dataset, validation_dataset, _, num_classes = data.load_msd(config.data, config.img_size, train_split=config.training_split, show_verbose=config.show_verbose) 
        
    training_dataset = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    validation_dataset = DataLoader(validation_dataset, batch_size=1, collate_fn=pad_list_data_collate)

    ##########################################################################################################
    ## Initialize the UNETR model
    # model = SelfDistilUNETR(in_channels, num_classes, img_size=config.img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)

    model = SelfDistilUNETR(in_channels, num_classes, img_size=config.img_size, self_distillation=True, use_feature_maps=False, mode=UpsampleMode.DECONV, interp_mode=InterpolateMode.BILINEAR, feature_size=32, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed="perceptron", norm_name="instance", res_block=True, dropout_rate=0.0)  # MMWHS CT only

    ##########################################################################################################
    ## Initialize the nnUNet model

    # kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]] # input + 3 Enc-Dec Layers + Bottleneck
    # strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] # input + 3 Enc-Dec Layers + Bottleneck
    # filters = [32,64,128,256,320]  # originally used (for MMWHS)
    # # filters = [16,32,64,128,256] # for MSD-BraTS due to memory limitations

    # model = SelfDistilnnUNet(
    #     spatial_dims = 3,
    #     in_channels = in_channels,
    #     out_channels = num_classes,
    #     kernel_size = kernel_size,
    #     strides = strides,
    #     upsample_kernel_size = strides[1:],
    #     filters=filters,
    #     norm_name="instance",
    #     deep_supervision=False,
    #     deep_supr_num=3,
    #     self_distillation=True,
    #     self_distillation_num=4,
    #     dataset = "MMWHS",
    #     res_block=True,
    #     )
    
    ##########################################################################################################

    ## Count model parameters
    print(f'The total number of model parameter is: {count_parameters(model)}')
    
    # initialize optimizer, loss, metrics, and post processing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5) # lr used by MMWHS challenge winner/MSD-BraTS (nnUnet)

    # initialize learning rate scheduler (lr used by MMWHS challenge winner)
    lr_step = max(int(config.epochs / 6), 1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_step, gamma=0.5) # used for MMWHS
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # lr=0.0001 # for MSD-BraTS

    # Initilialize Loss Functions
    loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

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
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec4 and GT labels 
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec3 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec2 and GT labels
        losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)), #out_dec1 and GT labels
        ], weight=1.0, target="out", learn_weights=False)

    # Self Distillation from deepest encoder/decoder (out_enc4/out_dec1): Teacher (T), to shallower encoders/decoders (out_enc2/out_dec2,out_enc3/out_dec3,out_dec4/out_enc1): Students (S)  
    # For KL Div between softmax(out_dec1/out_enc4) [target] and log_softmax((out_dec2/out_enc3,out_dec3/out_enc2,out_dec4/out_enc1)) [input]
    loss_KL = Self_Distillation_Loss_KL([ 
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec4/out_enc1 (S) & out_dec1/out_enc4 (T)
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec3/out_enc2 (S) & out_dec1/out_enc4 (T)
        losses.Loss(PixelWiseKLDiv(log_target=False)), #out_dec2/out_enc3 (S) & out_dec1/out_enc4 (T)
    ], weight=alpha_KL, include_background=True, T=temperature, learn_weights=False) # pass the entire dict NOT just "out"

    loss_fn = {
        "dice": loss_dice,
        "KL": loss_KL
    }

    '''# For L2 loss between feature maps/hints dec1_f/enc4_f [target: Teacher (T)] and feature maps (dec3_f/enc3_f,dec2_f/enc2_f) [input: Students (S)]
    loss_L2 = Self_Distillation_Loss_L2([
        losses.Loss(MSELoss(reduction="mean")), #dec3_f/enc2_f (S) and dec1_f/enc4_f (T)
        losses.Loss(MSELoss(reduction="mean")), #dec2_f/enc3_f (S) and dec1_f/enc4_f (T)
    ], weight=lambda_feat) # pass the entire dict NOT just "out"

    loss_fn = {
        "dice": loss_dice,
        "ce": loss_dice_ce,
        "KL": loss_KL,
        "L2": loss_L2
    }'''
   
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
    lr_scheduler_callback = callbacks.LrSchedueler(lr_scheduler, tf_board_writer=tensorboard_callback.writer)

    ##############################################################################################################

    # Final callbacks list
    callbacks_list: list[callbacks.Callback] = [tensorboard_callback, besti_ckpt_callback, last_ckpt_callback, lr_scheduler_callback]

    # train
    manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

    # save and test with last model
    model = manager.model
    torch.save(model, config.output_model)
    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

    # save and test with best model on validation dataset  
    manager = Manager.from_checkpoint("experiments/CT_MMWHS_UNETR_SelfDist_Original_Fold1.exp/best.model") # for Self Distillation Original

    if isinstance(manager.model, torch.nn.parallel.DataParallel): model = manager.model.module
    else: model = manager.model

    manager.model = model
    print(f'The best Dice score on validation set occurs at {manager.current_epoch + 1} epoch number')

    summary = manager.test(validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logging.info(summary)

