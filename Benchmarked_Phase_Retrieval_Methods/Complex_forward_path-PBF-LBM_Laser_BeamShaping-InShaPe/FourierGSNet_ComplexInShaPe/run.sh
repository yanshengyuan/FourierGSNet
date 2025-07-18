python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec

python3 train_reg.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name reg_rec.pth.tar --val_vis_path reg_rec

eval:
python3 train_reg.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name reg_rec.pth.tar --val_vis_path reg_rec --eval

python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec --eval