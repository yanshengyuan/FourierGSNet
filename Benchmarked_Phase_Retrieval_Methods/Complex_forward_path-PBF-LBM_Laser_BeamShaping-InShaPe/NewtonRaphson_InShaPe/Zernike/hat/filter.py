import numpy as np

ssim = np.load('ssim.npy')
mae = np.load('mae.npy')
frcm = np.load('frcm.npy')
recons = np.load('recons.npy')

cnt = 0
maes = []
ssims = []
frcms = []
reconses = []
for i in range(len(recons)):
    if(recons[i]<=1):
        print("sample "+str(i))
        print(mae[i])
        print(ssim[i])
        print(frcm[i])
        print(recons[i])
        
        maes.append(mae[i])
        ssims.append(ssim[i])
        frcms.append(frcm[i])
        reconses.append(recons[i])
        
        cnt+=1
        
maes = np.array(maes)
ssims = np.array(ssims)
frcms = np.array(frcms)
reconses = np.array(reconses)

success_rate = str(round((cnt/len(ssim))*100, 1))+"%"

print('\n')
print("Success Ratio: "+success_rate)
print("avg MAE: ", np.mean(maes))
print("avg SSIM: ", np.mean(ssims))
print("avg FRCM: ", np.mean(frcms))
print("avg ReconsErr: ", np.mean(reconses))