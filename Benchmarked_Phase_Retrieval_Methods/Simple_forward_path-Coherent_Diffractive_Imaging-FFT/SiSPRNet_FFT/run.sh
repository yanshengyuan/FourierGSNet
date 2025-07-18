train:
python3 train_SiSPRNet.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 1 --lr 0.0002 --step_size 2 --seed 123 --pth_name SiSPRNet_CDI_defocus_bs10.pth.tar --val_vis_path SiSPRNet_CDI_defocus_bs10

eval:
python3 train_SiSPRNet.py --data ../RAF_CDI_defocus --gpu 0 --batch_size 1 --seed 123 --pth_name SiSPRNet_CDI_defocus_bs10.pth.tar --val_vis_path SiSPRNet_CDI_defocus_bs10 --eval