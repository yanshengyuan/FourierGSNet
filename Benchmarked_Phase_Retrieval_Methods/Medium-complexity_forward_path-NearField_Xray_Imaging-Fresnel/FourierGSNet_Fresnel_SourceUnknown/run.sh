python3 train_GSNet.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_Xray.pth.tar --val_vis_path GSNet_Xray_result

eval:

python3 train_GSNet.py --data ../PhaseGAN_data --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_Xray.pth.tar --val_vis_path GSNet_Xray_result --eval