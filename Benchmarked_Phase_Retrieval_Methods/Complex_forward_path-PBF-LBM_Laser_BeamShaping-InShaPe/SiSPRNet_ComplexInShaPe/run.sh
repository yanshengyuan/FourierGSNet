train:
python3 train_SiSPRNet.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 0 --lr 0.0002 --step_size 2 --seed 123 --pth_name SiSPRNet_rec.pth.tar --val_vis_path SiSPRNet_rec

eval:
python3 train_SiSPRNet.py --data ../InShaPe_dataset/densegaussian30k_pre --batch_size 2 --gpu 0 --seed 123 --pth_name SiSPRNet_gaussian.pth.tar --val_vis_path SiSPRNet_gaussian --eval