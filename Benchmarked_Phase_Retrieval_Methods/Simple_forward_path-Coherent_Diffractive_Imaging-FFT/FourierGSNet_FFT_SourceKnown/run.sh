python3 train_GSNet.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 0 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_CDI_defocus_bs10.pth.tar --val_vis_path GSNet_CDI_defocus_bs10_PixNorm

python3 train_reg.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name reg_CDI_defocus_bs10.pth.tar --val_vis_path reg_CDI_defocus_bs10_PixNorm

eval:

python3 train_GSNet.py --data ../RAF_CDI_defocus --batch_size 1 --gpu 0 --seed 22 --pth_name GSNet_CDI_defocus_bs10.pth.tar --val_vis_path GSNet_CDI_defocus_bs10_PixNorm --eval

python3 train_reg.py --data ../RAF_CDI_defocus --batch_size 2 --gpu 0 --seed 22 --pth_name reg_CDI_defocus_bs10_PixNorm.pth.tar --val_vis_path reg_CDI_defocus_bs10_PixNorm --eval