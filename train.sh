python train_SelfDistil_Original.py /home/share/Data/Data_MMWHS_All/CT /home/neil/Lab_work/Medical_Image_Segmentation/Dual_Self-Distillation/trained_models/last_CT_MMWHS_SelfDistil_Original_Fold1.pth --img_size 96 96 96 --batch_size 1 --epochs 300 --experiment CT_MMWHS_nnUnet_SelfDist_Original_Fold1_multi_upsample_NON_trainable.exp --training_split 4 --device cuda:0 --show_verbose