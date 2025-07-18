python3 train_SiSPRNet.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 0 --lr 0.0002 --step_size 2 --seed 123 --pth_name SiSPRNet_Xray.pth.tar --val_vis_path SiSPRNet_Xray_result

eval:
python3 train_SiSPRNet.py --data ../PhaseGAN_data --batch_size 2 --gpu 0 --seed 123 --pth_name SiSPRNet_Xray.pth.tar --val_vis_path SiSPRNet_Xray_result --eval