# FourierGSNet

This repository is the code officially published and submitted in SPIE, USA of our Advanced Photonics journal paper publication in 2025.

Our paper: Efficient Gerchberg-Saxton algorithm deep unrolling for phase retrieval with complex forward path

Journal: Advanced Photonics of SPIE, USA.

DOI:

Citation:

Acknowledgements:

This work was supported by the EU InShaPe project (https://inshape-horizoneurope.eu) funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).

The root directory of this repository contains three folders: "Benchmarked_Phase_Retrieval_Methods", "Evaluations", and "Plottings" containing the codes for benchmarked methods implementations, experiment results evaluation codes, and experiment results analysis, respectively. The following instructions guide the users how to use the codes in different functionalities in different folders.

1), Benchmarked_Phase_Retrieval_Methods (Phase Retrieval methods codes):

In this folder, there are 5 sub-folders:

"Complex_forward_path-PBF-LBM_Laser_BeamShaping-InShaPe",

"Medium-complexity_forward_path-NearField_Xray_Imaging-Fresnel",

"Simple_forward_path-Coherent_Diffractive_Imaging-FFT",

"GS_Algorithms" and "M290_Laser_BeamShaping_BatchGPU_simulation".

The first 3 folders contains the experiment codes of benchmarked methods on the three applications with three different degrees of forward path complexity. The last 2 folders contains the codes of the Pytorch wave optics simulation that is used in the implementation of the benchmarked methods GS direct unrolling, Gradient Descent, and Newton Raphson method and a GS algorithm implemented using LightPipes for PBF-LB/M laser beam shaping EOS M290 system. In the followings of this section, we will give instructures on how to reproduce these experiments in each sub-folder.

Library and version requirements:

python==3.11.9

numpy==1.26.3

h5py==3.12.1

hdf5==1.14.3

matplotlib==3.8.0

opencv-python==4.9.0

pytorch==2.1.1

tensorflow==2.3.0

scipy==1.11.4

scikit-image==0.24.0

seaborn==0.13.2

LightPipes==2.1.4

    (a. Complex_forward_path-PBF-LBM_Laser_BeamShaping-InShaPe

    

    a1. Data preparation:

    cd InShaPe_dataset, there are 6 sub-folders in it. Operations to reproduce the InShaPe sub-set in each sub-folder are the same, so here we only take Chair shape for example

    cd densebench30k_pre

    Move the Zernike coefficients numpy file of the training set and test set of the original InShaPe dataset released by the OE (Optics Express) paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" into the current directory.

    Run the data processing command to re-run the original simulation released by the OE paper and produce SLM phase maps and intensity images:

    python3 data_processing_train.py

    python3 data_processing_test.py

    

    a2. Trainings:

    FourierGSNet with physics knowledged injected:

    cd FourierGSNet_ComplexInShaPe
    
    python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec
    

    FourierGSNet without physics knowledged injected:

    cd FourierGSNet_ComplexInShaPe

    python3 train_reg.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name reg_rec.pth.tar --val_vis_path reg_rec

    

    Inferences:

    FourierGSNet with physics knowledged injected:

    cd FourierGSNet_ComplexInShaPe

    python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec --eval
    

    FourierGSNet without physics knowledged injected:

    cd FourierGSNet_ComplexInShaPe

    python3 train_reg.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name reg_rec.pth.tar --val_vis_path reg_rec --eval






















    
