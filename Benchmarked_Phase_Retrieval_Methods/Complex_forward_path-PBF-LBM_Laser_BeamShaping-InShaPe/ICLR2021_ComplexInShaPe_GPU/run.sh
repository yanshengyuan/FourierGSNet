For Gaussian and Tophat beamshape:
python3 train_ICLR.py --data ../InShaPe_dataset/densegaussian30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.00001 --step_size 2 --seed 22 --pth_name ICLR_gaussian.pth.tar --val_vis_path ICLR_gaussian

For all other beamshapes:
python3 train_ICLR.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_rec.pth.tar --val_vis_path ICLR_rec

eval:
python3 train_ICLR.py --data ../InShaPe_dataset/densegaussian30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name ICLR_gaussian.pth.tar --val_vis_path ICLR_gaussian --eval