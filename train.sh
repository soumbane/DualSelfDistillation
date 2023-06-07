######################################################################################################################
############################################## UNETR/nnUNET (MMWHS) ##################################################
######################################################################################################################
## Modality - CT
## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE - Basic
# python train_basic.py /home/share/Data/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_Basic.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_MMWHS_UNETR_Basic.exp --training_split 4 --device cuda:0

python train_basic.py /home/share/Data/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_Basic.pth --img 96 96 96 --batch_size 1 --epochs 600 --experiment CT_MMWHS_UNETR_Basic.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE - Deep Supervision
# python train_DeepSuperOnly.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_DeepSuperOnly.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_MMWHS_UNETR_DeepSuperOnly.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE + KL Div - Self Distillation Original
# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT1_MMWHS_UNETR_SelfDist_Original.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE + KL Div + Boundary - Self Distillation with Distance Maps
# python train_SelfDistil_DistMaps.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_SelfDistil_DistMaps.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment CT6_MMWHS_nnUnet_SelfDist_DistMaps.exp --training_split 4 --device cuda:1 --show_verbose

#####################################################################################################################

## Modality - MR
## For running train script for MMWHS Challenge Multi-class with MR ONLY (on 1 GPU): Dice CE - Basic
# python train_basic.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/MR /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MR_MMWHS_Basic.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment MR_MMWHS_nnUnet_Basic.exp --training_split 4 --device cuda:1 --show_verbose

## For running train script for MMWHS Challenge Multi-class with MR ONLY (on 1 GPU): Dice CE - Deep Supervision
# python train_DeepSuperOnly.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/MR /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MR_MMWHS_DeepSuperOnly.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment MR_MMWHS_nnUnet_DeepSuperOnly.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with MR ONLY (on 1 GPU): Dice CE + KL Div - Self Distillation Original
# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/MR /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MR_MMWHS_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment MR_MMWHS_nnUnet_SelfDist_Original.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with MR ONLY (on 1 GPU): Dice CE + KL Div + Boundary - Self Distillation with Distance Maps
# python train_SelfDistil_DistMaps.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/MR /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MR_MMWHS_SelfDistil_DistMaps.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment MR_MMWHS_nnUnet_SelfDist_DistMaps.exp --training_split 4 --device cuda:1 --show_verbose


######################################################################################################################
############################################# UNETR/nnUNET (MSD-BraTS) ###############################################
######################################################################################################################

## Multi-Modal MR (T1,T2,T1gd,FLAIR)
## For running train script for MSD-BraTS Challenge Multi-class multi-modal MR (on 1 GPU): Dice CE - Basic
# for nnUnet
# python train_basic.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MSD_BraTS_Basic.pth --img_size 128 128 128 --batch_size 1 --epochs 300 --experiment multimodalMR_MSD_BraTS_nnUnet_Basic.exp --training_split 24 --device cuda:0 --show_verbose

# for UNETR
# python train_basic.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MSD_BraTS_Basic.pth --img_size 128 128 128 --batch_size 1 --epochs 325 --experiment multimodalMR_MSD_BraTS_UNETR_Basic.exp --training_split 24 --device cuda:1 --show_verbose

# for UNETR (from checkpoint)
# python train_basic_from_checkpoint.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MSD_BraTS_Basic_from_ch.pth --img_size 128 128 128 --batch_size 1 --epochs 106 --experiment multimodalMR_MSD_BraTS_UNETR_Basic_from_checkpoint_294_epochs.exp --training_split 24 --device cuda:1 --show_verbose

## For running train script for MSD-BraTS Challenge Multi-class multi-modal MR (on 1 GPU): Dice CE + KL Div - Self Distillation Original
# for nnUnet
# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MSD_BraTS_SelfDistil_Original.pth --img_size 128 128 128 --batch_size 1 --epochs 300 --experiment multimodalMR_MSD_BraTS_nnUnet_SelfDist_Original.exp --training_split 24 --device cuda:1 --show_verbose

# for UNETR
# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MSD_BraTS_SelfDistil_Original.pth --img_size 128 128 128 --batch_size 1 --epochs 325 --experiment multimodalMR_MSD_BraTS_UNETR_SelfDist_Original.exp --training_split 24 --device cuda:0 --show_verbose

# for UNETR (from checkpoint)
# python train_SelfDistil_Original_from_checkpoint.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_MSD_BraTS_SelfDistil_Original_from_ch.pth --img_size 128 128 128 --batch_size 1 --epochs 110 --experiment multimodalMR_MSD_BraTS_UNETR_SelfDist_Original_from_checkpoint_290_epochs.exp --training_split 24 --device cuda:0 --show_verbose

######################################################################################################################
######################################## UNETR/nnUNET (MMWHS - Ablation Study) #######################################
######################################################################################################################
## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE + KL Div - Self Distillation Original (Decoders only)
# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_MMWHS_UNETR_SelfDist_Original_Decoders_Only.exp --training_split 4 --device cuda:0 --show_verbose

# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment CT1_MMWHS_nnUnet_SelfDist_Original_Decoders_Only.exp --training_split 4 --device cuda:0 --show_verbose

## For running train script for MMWHS Challenge Multi-class with CT ONLY (on 1 GPU): Dice CE + KL Div - Self Distillation Original (Encoders only)
# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment CT_MMWHS_nnUnet_SelfDist_Original_Encoders_Only.exp --training_split 4 --device cuda:1 --show_verbose

# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/last_CT_MMWHS_SelfDistil_Original.pth --img_size 96 96 96 --batch_size 1 --epochs 600 --experiment CT_MMWHS_UNETR_SelfDist_Original_Encoders_Only.exp --training_split 4 --device cuda:0 --show_verbose



