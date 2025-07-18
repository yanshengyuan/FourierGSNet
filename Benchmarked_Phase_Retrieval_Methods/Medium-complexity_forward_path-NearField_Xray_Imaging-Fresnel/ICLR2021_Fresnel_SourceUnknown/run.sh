python3 train_ICLR.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_Xray.pth.tar --val_vis_path ICLR_Xray_result

python3 train_ICLR.py --data ../PhaseGAN_data --batch_size 1 --gpu 0 --seed 22 --pth_name ICLR_Xray.pth.tar --val_vis_path ICLR_Xray_result --eval