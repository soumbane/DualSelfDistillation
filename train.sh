python train_basic.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/test.pth --img_size 128 128 128 --batch_size 1 --epochs 325 --experiment multimodalMR_MSD_BraTS_LargeUNETR_Basic.exp --training_split 24 --device cuda:0 --show_verbose

# python train_SelfDistil_Original.py /home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS /home/neil/Lab_work/Medical_Image_Segmentation/Dual_SelfDistillation/trained_models/test1.pth --img_size 128 128 128 --batch_size 1 --epochs 325 --experiment multimodalMR_MSD_BraTS_LargeUNETR_SelfDist.exp --training_split 24 --device cuda:1 --show_verbose