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

numpy==1.26.3

python==3.11.9

matplotlib==3.5.2
