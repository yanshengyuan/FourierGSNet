import torch
import os
import numpy as np
from math import sqrt
from skimage.metrics import structural_similarity as ssim
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=1

def calculate_frcm(img1, img2):
    nz,nx,ny= [torch.tensor(i, device=device) for i in img1.shape]
    rnyquist = nx//2
    x = torch.cat((torch.arange(0, nx / 2), torch.arange(-nx / 2, 0))).to(device)
    y = x
    X, Y = torch.meshgrid(x, y)
    map = X ** 2 + Y ** 2
    index = torch.round(torch.sqrt(map.float()))
    r = torch.arange(0, rnyquist + 1).to(device)
    #print(r.type())
    r=r.float()
    F1 = torch.rfft(img1, 2, onesided=False).permute(1, 2, 0, 3)
    F2 = torch.rfft(img2, 2, onesided=False).permute(1, 2, 0, 3)
    C_r,C1,C2,C_i = [torch.empty(rnyquist + 1, batch_size).to(device) for i in range(4)]
    for ii in r:
        #print(ii.type())
        auxF1 = F1[torch.where(index == ii)]
        auxF2 = F2[torch.where(index == ii)]
        ii=ii.int()
        C_r[ii] = torch.sum(auxF1[:, :, 0] * auxF2[:, :, 0] + auxF1[:, :, 1] * auxF2[:, :, 1], axis=0)
        C_i[ii] = torch.sum(auxF1[:, :, 1] * auxF2[:, :, 0] - auxF1[:, :, 0] * auxF2[:, :, 1], axis=0)
        C1[ii] = torch.sum(auxF1[:, :, 0] ** 2 + auxF1[:, :, 1] ** 2, axis=0)
        C2[ii] = torch.sum(auxF2[:, :, 0] ** 2 + auxF2[:, :, 1] ** 2, axis=0)

    FRC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
    FRCm = 1 - torch.where(FRC != FRC, torch.tensor(1.0, device=device), FRC)
    My_FRCloss = torch.mean((FRCm) ** 2)
    #print("Fourier Ring Correlation metric:",My_FRCloss)
    return My_FRCloss

gt_folder = './Phi_gt/npy'
pred_folder = './Phi_pred/npy'
mae_list = []
ssim_list = []
frcm_list = []

cnt=0
# Traverse through all files in the gt folder
for gt_filename in os.listdir(gt_folder):
    # Load ground truth and prediction arrays
    gt_path = os.path.join(gt_folder, gt_filename)
    pred_path = os.path.join(pred_folder, gt_filename)

    if os.path.exists(pred_path):
        gt_array = np.load(gt_path)
        pred_array = np.load(pred_path)

        # Calculate RMSE
        diff = np.abs(gt_array - pred_array)
        mae = np.mean(diff)
        mae_list.append(mae)
        
        diff_name = gt_filename[:-4]+"_"+str(round(mae, 3))+".png"
        
        plt.imshow(diff, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Absolute Difference Phase')
        plt.title("Absolute Difference Map")
        plt.savefig('./diff/'+diff_name, dpi=150)
        plt.close()

        # Calculate DSSIM
        #'''
        ssim_value = ssim(gt_array, pred_array, data_range=gt_array.max() - gt_array.min())
        ssim_list.append(ssim_value)
        #'''

        # Calculate FRCM
        gt_array=torch.from_numpy(gt_array.astype(np.float32)).unsqueeze(0)
        pred_array=torch.from_numpy(pred_array.astype(np.float32)).unsqueeze(0)
        frcm = calculate_frcm(gt_array, pred_array)
        frcm=frcm.squeeze().cpu().numpy().item()
        frcm_list.append(frcm)
        cnt+=1
        print(cnt)

# Calculate mean RMSE, DSSIM, and FRCM
mean_mae = np.mean(mae_list)
mean_ssim = np.mean(ssim_list)
mean_frcm = np.mean(frcm_list)

print(f"Mean MAE: {mean_mae}")
print(f"Mean SSIM: {mean_ssim}")
print(f"Mean FRCM: {mean_frcm}")

sns.kdeplot(mae_list, fill=True, bw_adjust=0.5)

# Labels and title
plt.xlabel("Mean Absolute Error (MAE)")
plt.ylabel("Density")
plt.title("KDE Plot of MAE metric across 100 test samples")
plt.savefig("KDE_MAE.png", dpi=300)
plt.close()

sns.kdeplot(ssim_list, fill=True, bw_adjust=0.5)

# Labels and title
plt.xlabel("Structure Similarity (SSIM)")
plt.ylabel("Density")
plt.title("KDE Plot of SSIM metric across 100 test samples")
plt.savefig("KDE_SSIM.png", dpi=300)
plt.close()

sns.kdeplot(frcm_list, fill=True, bw_adjust=0.5)  # `fill=True` for shaded area, `bw_adjust` adjusts smoothness

# Labels and title
plt.xlabel("Fourier Ring Correlation Metric (FRCM)")
plt.ylabel("Density")
plt.title("KDE Plot of FRCM metric across 100 test samples")
plt.savefig("KDE_FRCM.png", dpi=300)
plt.close()

mae_list=np.array(mae_list)
ssim_list=np.array(ssim_list)
frcm_list=np.array(frcm_list)

np.save("mae.npy", mae_list)
np.save("ssim.npy", ssim_list)
np.save("frcm.npy", frcm_list)