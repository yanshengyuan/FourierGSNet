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

    1. FourierGSNet with physics knowledged injected:
    cd FourierGSNet_ComplexInShaPe
    python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec

    2. FourierGSNet without physics knowledged injected:
    cd FourierGSNet_ComplexInShaPe
    python3 train_reg.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name reg_rec.pth.tar --val_vis_path reg_rec

    3. GS direct unrolling:
    cd ICLR2021_ComplexInShaPe_GPU

    For Gaussian and Tophat beamshape:
    python3 train_ICLR.py --data ../InShaPe_dataset/densegaussian30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.00001 --step_size 2 --seed 22 --pth_name ICLR_gaussian.pth.tar --val_vis_path ICLR_gaussian
    
    For all other beamshapes:
    python3 train_ICLR.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_rec.pth.tar --val_vis_path ICLR_rec

    4. SiSPRNet
    cd SiSPRNet_ComplexInShaPe
    python3 train_SiSPRNet.py --data ../InShaPe_dataset/denserec30k_pre --epochs 30 --batch_size 2 --gpu 0 --lr 0.0002 --step_size 2 --seed 123 --pth_name SiSPRNet_rec.pth.tar --val_vis_path SiSPRNet_rec

    5. deep-CDI
    cd deepcdi_ComplexInShaPe
    python3 0_train-cnn/train_cnn_chair.py

    6. Gradient Descent fitting method based on free SLM phase
    cd GradientDescent_InShaPe/SLM_phase
    cd chair
    python3 M290_GradientDescent_GPU.py --beamshape Chair --gpu 0

    7. Gradient Descent fitting method based on standard SLM phase mask + Zernike polynomials * Zernike Coefficients
    cd GradientDescent_InShaPe/Zernike
    cd chair
    python3 M290_GradientDescent_GPU.py --beamshape Chair --gpu 0

    Measure the fitting time averaged across six beam shapes, also referred to as inference time in our paper:
    cd ..
    python3 runtime_statistics.py

    8. Newton-Raphson fitting method based on free SLM phase
    cd NewtonRaphson_InShaPe/Phase_Fresnel
    cd chair
    python3 Fresnel_NewtonRaphson.py --beamshape Chair --gpu 0

    9. Newton-Raphson fitting method based on standard SLM phase mask + Zernike polynomials * Zernike Coefficients
    cd NewtonRaphson_InShaPe/Zernike
    cd chair
    python3 M290_NewtonRaphson_GPU.py --beamshape Chair --gpu 0

    Measure the fitting time averaged across six beam shapes, also referred to as inference time in our paper:
    cd ..
    python3 runtime_statistics.py
    

    The Trainings not included and did not contribute to our Advanced Photonics journal paper:

    10. Attention map, regression feature, and physics knowledge feature Feature Study scripts
    cd FourierGSNet_feature_study
    python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec --eval --log_features True

    11. CPU version non-differentiable GS direct unrolling implemented based on LightPipes
    cd ICLR2021_ComplexInShaPe_CPU
    python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec
    


    
    a3. Inferences:

    1. FourierGSNet with physics knowledged injected:
    python3 train_GSNet.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_rec.pth.tar --val_vis_path GSNet_rec --eval

    2. FourierGSNet without physics knowledged injected:
    python3 train_reg.py --data ../InShaPe_dataset/denserec30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name reg_rec.pth.tar --val_vis_path reg_rec --eval

    3. GS direct unrolling:
    python3 train_ICLR.py --data ../InShaPe_dataset/densegaussian30k_pre --batch_size 2 --gpu 0 --seed 22 --pth_name ICLR_gaussian.pth.tar --val_vis_path ICLR_gaussian --eval

    4. SiSPRNet
    python3 train_SiSPRNet.py --data ../InShaPe_dataset/densegaussian30k_pre --batch_size 2 --gpu 0 --seed 123 --pth_name SiSPRNet_gaussian.pth.tar --val_vis_path SiSPRNet_gaussian --eval

    5. deep-CDI
    python3 0_train-cnn/train_cnn_chair.py




(b. Medium-complexity_forward_path-NearField_Xray_Imaging-Fresnel




    b1. Data preparation:

    Training set:
    cd PhaseGAN_data
    Copy all .h5 files in the training set of the PhaseGAN example dataset in: https://drive.google.com/drive/folders/1rKTZYJa54WeG-2TikoXpdRcqTiSQ-Ps5
    Paste the copied files in the root directory of /PhaseGAN_data/.

    Test set:
    cd PhaseGAN_data/test
    Copy all .h5 files in the test set of the PhaseGAN example dataset in: https://drive.google.com/drive/folders/1rKTZYJa54WeG-2TikoXpdRcqTiSQ-Ps5
    Paste the copied files in the directory /PhaseGAN_data/test/.
    



    b2. Trainings:

    1. FourierGSNet with physics knowledged injected:
    cd FourierGSNet_Fresnel_SourceUnknown
    python3 train_GSNet.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_Xray.pth.tar --val_vis_path GSNet_Xray_result

    2. FourierGSNet without physics knowledged injected:
    cd FourierGSNet_Fresnel_SourceKnown
    python3 train_reg.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name reg_Xray.pth.tar --val_vis_path reg_Xray_result

    3. GS direct unrolling:
    cd ICLR2021_Fresnel_SourceUnknown
    python3 train_ICLR.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_Xray.pth.tar --val_vis_path ICLR_Xray_result

    4. SiSPRNet
    cd SiSPRNet_Fresnel
    python3 train_SiSPRNet.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 0 --lr 0.0002 --step_size 2 --seed 123 --pth_name SiSPRNet_Xray.pth.tar --val_vis_path SiSPRNet_Xray_result

    5. deep-CDI
    cd deepcdi_Fresnel
    python3 0_train-cnn/train_cnn_xray.py
    
    The Trainings not included and did not contribute to our Advanced Photonics journal paper:

    6. FourierGSNet with the source plane intensity known and given as the source plane intensity contraints
    cd FourierGSNet_Fresnel_SourceKnown
    python3 train_GSNet.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_Xray.pth.tar --val_vis_path GSNet_Xray_result
    python3 train_GSNet.py --data ../PhaseGAN_data --batch_size 1 --gpu 0 --seed 22 --pth_name GSNet_Xray.pth.tar --val_vis_path GSNet_Xray_result --eval

    7. GS direct unrolling with the source plane intensity known and given as the source plane intensity contraints
    cd ICLR2021_Fresnel
    python3 train_ICLR.py --data ../PhaseGAN_data --epochs 30 --batch_size 2 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_Xray.pth.tar --val_vis_path ICLR_Xray_result
    python3 train_ICLR.py --data ../PhaseGAN_data --batch_size 1 --gpu 0 --seed 22 --pth_name ICLR_Xray.pth.tar --val_vis_path ICLR_Xray_result --eval
    
    8. Random 7-3 train-test split 10-fold cross validation
    python3 random_testset.py

    9. Gradient Descent fitting method based on free object plane phase and with the source plane intensity known and given as the source plane intensity contraints
    cd GradientDescent_Fresnel
    python3 Fresnel_GradientDescent.py --data ../PhaseGAN_data --gpu 0


    
    b3. Inferences:

    1. FourierGSNet with physics knowledged injected:
    python3 train_GSNet.py --data ../PhaseGAN_data --batch_size 2 --gpu 0 --seed 22 --pth_name GSNet_Xray.pth.tar --val_vis_path GSNet_Xray_result --eval

    2. FourierGSNet without physics knowledged injected:
    python3 train_reg.py --data ../PhaseGAN_data --batch_size 1 --gpu 0 --seed 22 --pth_name reg_Xray.pth.tar --val_vis_path reg_Xray_result --eval

    3. GS direct unrolling:
    python3 train_ICLR.py --data ../PhaseGAN_data --batch_size 1 --gpu 0 --seed 22 --pth_name ICLR_Xray.pth.tar --val_vis_path ICLR_Xray_result --eval

    4. SiSPRNet
    python3 train_SiSPRNet.py --data ../PhaseGAN_data --batch_size 2 --gpu 0 --seed 123 --pth_name SiSPRNet_Xray.pth.tar --val_vis_path SiSPRNet_Xray_result --eval

    5. deep-CDI
    python3 0_train-cnn/train_cnn_xray.py




(c. Simple_forward_path-Coherent_Diffractive_Imaging-FFT




    c1. Data preparation:
    Get the RAF-CDI dataset from the authors of the Optics Express paper "Towards practical single-shot phase retrieval with physics-driven deep neural network." via email and put all data into the folder:
    cd Simple_forward_path-Coherent_Diffractive_Imaging-FFT/RAF_CDI_defocus/.

    Then run data processing script:
    python3 data_processing.py
    



    c2. Trainings:

    1. FourierGSNet with physics knowledged injected:
    cd FourierGSNet_FFT_SourceUnknown
    python3 train_GSNet.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 0 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_CDI_defocus_bs10_PixNorm.pth.tar --val_vis_path GSNet_CDI_defocus_bs10_PixNorm
    
    2. FourierGSNet without physics knowledged injected:
    cd FourierGSNet_FFT_SourceKnown
    python3 train_reg.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 1 --lr 0.0002 --step_size 2 --seed 22 --pth_name reg_CDI_defocus_bs10.pth.tar --val_vis_path reg_CDI_defocus_bs10_PixNorm

    3. GS direct unrolling:
    cd ICLR2021_FFT_SourceUnknown
    python3 train_ICLR.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 0 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_CDI.pth.tar --val_vis_path ICLR_CDI_bs10_PixNorm

    4. SiSPRNet
    cd SiSPRNet_FFT
    python3 train_SiSPRNet.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 1 --lr 0.0002 --step_size 2 --seed 123 --pth_name SiSPRNet_CDI_defocus_bs10.pth.tar --val_vis_path SiSPRNet_CDI_defocus_bs10

    5. deep-CDI
    cd deepcdi_FFT
    python3 0_train-cnn/train_cnn_defocus_bs2.py
    
    The Trainings not included and did not contribute to our Advanced Photonics journal paper:

    6. FourierGSNet with the source plane intensity known and given as the source plane intensity contraints
    cd FourierGSNet_FFT_SourceKnown
    python3 train_GSNet.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 0 --lr 0.0002 --step_size 2 --seed 22 --pth_name GSNet_CDI_defocus_bs10.pth.tar --val_vis_path GSNet_CDI_defocus_bs10_PixNorm
    python3 train_GSNet.py --data ../RAF_CDI_defocus --batch_size 1 --gpu 0 --seed 22 --pth_name GSNet_CDI_defocus_bs10.pth.tar --val_vis_path GSNet_CDI_defocus_bs10_PixNorm --eval

    7. GS direct unrolling with the source plane intensity known and given as the source plane intensity contraints
    cd ICLR2021_FFT
    python3 train_ICLR.py --data ../RAF_CDI_defocus --epochs 30 --batch_size 10 --gpu 0 --lr 0.0002 --step_size 2 --seed 22 --pth_name ICLR_CDI_bs10.pth.tar --val_vis_path ICLR_CDI_bs10
    python3 train_ICLR.py --data ../RAF_CDI_defocus --batch_size 1 --gpu 0 --seed 22 --pth_name ICLR_CDI_bs10.pth.tar --val_vis_path ICLR_CDI_bs10 --eval

    8. Gradient Descent fitting method based on free object plane phase and with the source plane intensity known and given as the source plane intensity contraints
    cd GradientDescent_FFT
    python3 FFT_GradientDescent.py --data ../RAF_CDI_defocus --gpu 0


    
    c3. Inferences:

    1. FourierGSNet with physics knowledged injected:
    python3 train_GSNet.py --data ../RAF_CDI_defocus --batch_size 1 --gpu 0 --seed 22 --pth_name GSNet_CDI_defocus_bs10_PixNorm.pth.tar --val_vis_path GSNet_CDI_defocus_bs10_PixNorm --eval

    2. FourierGSNet without physics knowledged injected:
    python3 train_reg.py --data ../RAF_CDI_defocus --batch_size 2 --gpu 0 --seed 22 --pth_name reg_CDI_defocus_bs10_PixNorm.pth.tar --val_vis_path reg_CDI_defocus_bs10_PixNorm --eval

    3. GS direct unrolling:
    python3 train_ICLR.py --data ../RAF_CDI_defocus --batch_size 1 --gpu 0 --seed 22 --pth_name ICLR_CDI.pth.tar --val_vis_path ICLR_CDI_bs10_PixNorm --eval

    4. SiSPRNet
    python3 train_SiSPRNet.py --data ../RAF_CDI_defocus --gpu 0 --batch_size 1 --seed 123 --pth_name SiSPRNet_CDI_defocus_bs10.pth.tar --val_vis_path SiSPRNet_CDI_defocus_bs10 --eval

    5. deep-CDI
    python3 0_train-cnn/train_cnn_defocus_bs2.py





2), Evaluations (Implementation of accuracy metrics MAE, SSIM, FRCM, ReconsErr and evaluation codes):

In this folder, there are 14 sub-folders. Each sub-folder contains the same codes that implements metrics MAE, SSIM, FRCM, ReconsErr and evaluation procedure. Therefore, we first list out which model's result is contained in which folder. Then we introduce inside each sub-folder which scripts should be run to compute the metrics for the results.

Library and version requirements:

python==3.7.12

numpy==1.21.6

pandas==1.3.5

matplotlib==3.5.3

pytorch==1.2.0




(a. List of sub-folder -- method pairs

    GSNet_CDI_SourceUnknown -- FourierGSNet on RAF-CDI dataset
    
    GSNet_CDI_WOT -- FourierGSNet without physics knowledge injected on RAF-CDI dataset
    
    GSNet_Xray_SourceUnknown -- FourierGSNet on PhaseGAN example dataset
    
    GSNet_Xray_WOT -- FourierGSNet without physics knowledge injected on PhaseGAN example dataset
    
    GSNet_inshape_pre/GSNet_chair -- FourierGSNet on InShaPe dataset
    
    GSNet_inshape_pre/GSNet_WOT_chair -- FourierGSNet without physics knowledge injected on InShaPe dataset
    
    ICLR2021_CDI_SourceUnknown -- GS direct unrolling on RAF-CDI dataset
    
    ICLR2021_Xray_SourceUnknown -- GS direct unrolling on PhaseGAN example dataset
    
    ICLR2021_inshape_pre -- GS direct unrolling on InShaPe dataset
    
    SiSPRNet_CDI -- SiSPRNet on RAF-CDI dataset
    
    SiSPRNet_Xray -- SiSPRNet on PhaseGAN example dataset
    
    SiSPRNet_inshape_pre -- SiSPRNet on InShaPe dataset
    
    deep-CDI_CDI -- deep-CDI on RAF-CDI dataset
    
    deep-CDI_Xray -- deep-CDI on PhaseGAN example dataset
    
    deep-CDI_inshape_pre -- deep-CDI on InShaPe dataset

(b. The functionalities in each sub-folder

    1. Run the MAE, SSIM, FRCM accuracy metric evaluation script:
    python3 FRCM.py

    2. Run the ReconsErr accuracy metric evaluation script:
    cd recons_err/.
    python3 recons_err.py





3), Plottings (Violin KDEs plots):

This folder contains the matplotlib codes that plots all the experiment results statistical analysis codes

There are 4 sub-folders in this folder: Figure-Ablation, Violin_FourierCDI, Violin_InShaPe, Violin_Xray containing the plotting codes for the experiment results on the contrast experiment between FourierGSNet with or without physics knowledge injected, RAF-CDI dataset, InShaPe dataset, and PhaseGAN example dataset, respectively.

For the Figure-Ablation:

    python3 Violins_overlap.py

For the Violin_FourierCDI and Violin_Xray, respectively:

    python3 Violin_plot.py

For the Violin_InShaPe:

    inside the script Violin_plot.py, revise line 14:
    beamshape="Gaussian/", change the beam shape to one of the 6 beam shapes of InShaPe dataset.
    Then run: python3 Violin_plot.py
    You will get the violin plot of the corresponding beam shape.

# Author: Shengyuan Yan, Inter-connected Resource-Aware Intelligent Systems (IRIS), TU Eindhoven, 12:57 AM, Eindhoven, Netherlands, EU West.
