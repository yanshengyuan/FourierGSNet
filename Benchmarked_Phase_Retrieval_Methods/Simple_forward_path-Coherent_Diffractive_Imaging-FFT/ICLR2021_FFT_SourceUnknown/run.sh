train:
python3 train_ICLR.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 0 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_CDI.pth.tar --val_vis_path ICLR_CDI_bs10_PixNorm

test:
python3 train_ICLR.py --data ../RAF_CDI_defocus --batch_size 1 --gpu 0 --seed 22 --pth_name ICLR_CDI.pth.tar --val_vis_path ICLR_CDI_bs10_PixNorm --eval