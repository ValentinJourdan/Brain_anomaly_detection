import cv2
import os
import numpy as np
import random
import torch
print(torch.cuda.is_available())
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

TRAINPROP = 0.8
TESTPROP = 0.2
BATCHSIZE = 16
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 15
NUM_FOLDS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################ Classes ########################################################################################################################################

class CustomDataset(Dataset):
  def __init__(self):
    self.name = []
    self.folder_path = 'brain_scan/'
    for f1 in os.listdir(self.folder_path):
      if not (f1 == "data.csv" or f1 == "README.md"): 
        for f2 in os.listdir(os.path.join(self.folder_path,f1)):
          if f2.endswith('mask.tif'):
            self.name.append(os.path.join(f1,f2))

  def __len__(self):
    return len(self.name)

  def __getitem__(self,idx):
    self.mask = 1/255*cv2.imread(os.path.join(self.folder_path, self.name[idx]),cv2.IMREAD_GRAYSCALE)[:,:]
    self.image = 1/255*cv2.imread(os.path.join(self.folder_path, self.name[idx].replace("_mask.tif",".tif")),cv2.IMREAD_GRAYSCALE)[:,:]

    return (torch.tensor(self.image, dtype = torch.float).unsqueeze(0), torch.tensor(self.mask, dtype = torch.float).unsqueeze(0), self.name[idx])



class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.contracting_block(in_channels, 32,5)
        self.enc2 = self.contracting_block(32, 64,3)
        self.enc3 = self.contracting_block(64, 128,3)
        self.enc4 = self.contracting_block(128, 256,3)
        self.enc5 = self.contracting_block(256, 512,3)
        #self.drop = nn.Dropout(p=0.1)

        # Decoder
        self.upconv4 = self.upconv(512, 256)
        self.dec4 = self.expanding_block(512, 256)
        self.upconv3 = self.upconv(256, 128)
        self.dec3 = self.expanding_block(256, 128)
        self.upconv2 = self.upconv(128, 64)
        self.dec2 = self.expanding_block(128, 64)
        self.upconv1 = self.upconv(64, 32)
        self.dec1 = self.expanding_block(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1, padding='same')

        self.sigmoid = nn.Sigmoid()

        #self._initialize_weights()

    def contracting_block(self, in_channels, out_channels,size):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return block

    def expanding_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
        )
        return block

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        #enc3 = self.drop(enc3)
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))
        #enc5 = self.drop(enc5)

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = self.crop_and_concat(dec4, enc4, crop=True)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.crop_and_concat(dec3, enc3, crop=True)
        dec3 = self.dec3(dec3)
        #dec3 = self.drop(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.crop_and_concat(dec2, enc2, crop=True)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.crop_and_concat(dec1, enc1, crop=True)
        dec1 = self.dec1(dec1)

        out = self.final_conv(dec1)
        #out = self.drop(out)
        out = self.sigmoid(out)

        return out
 
################ Functions ###############################################################################################################################

#Defining the Dice Loss
def dice_coefficient_proba(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_coefficient(y_true, y_pred, smooth=1):
    y_pred = (y_pred  > 0.5).float()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def total_loss(y_true, y_pred, smooth=1):
    loss = 0.8*(1 - dice_coefficient_proba(y_true, y_pred, smooth)) + 0.2*nn.BCELoss()(y_pred, y_true)
    return loss

######################################################################################################################################################


dt = CustomDataset()
idx_P = []
idx_N = []
for idx in range(dt.__len__()):
    mask = dt.__getitem__(idx)[1]
    r = np.array(mask[:,:])
    if np.max(r) == 0:
        idx_N.append(idx)
    else :
        idx_P.append(idx)

#Delete some of the empty mask
list_idx = random.sample(idx_N, len(idx_P)) + idx_P
#Create a subset dataset to equilibrate the dataset
equil_dt = Subset(dt, list_idx)

train_ds, test_ds = random_split(equil_dt, [TRAINPROP, TESTPROP])

my_nn = UNet().to(device)
print(my_nn)

criterion = total_loss

optimizer = optim.Adam(my_nn.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

################ Trains ###############################################################################################################################

# Define data loaders for training and validation
train_dl = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True)

# Visualization list and flag for saving the label
ref_name = ''
name_saved = False

for epoch in range(EPOCHS):
    running_loss = 0.0
    running_dice_coeff = []
    dataloader = tqdm(train_dl, position=0, leave=True)

    for inputs, labels, name in dataloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = my_nn(inputs)
        
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice_coeff.append(dice_coefficient(labels, outputs).item())

        dataloader.set_description(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_dl):.5f} - Dice Coefficient: {100 * sum(running_dice_coeff) / len(running_dice_coeff):.1f}%')
        dataloader.refresh()
    
    scheduler.step()

    dataloader.close()

print('Training finished')

################ Test ###############################################################################################################################

test_dl = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=True)

my_nn.eval()
val_dice_coeff = []
example_showed = False
k = 0
for inputs, labels , name in test_dl:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = my_nn(inputs)
    val_dice_coeff.append(dice_coefficient(labels, outputs).item())
    


    if not example_showed and np.max(labels[0, 0, :, :].cpu().numpy()) == 1:
        input_image = inputs[0, 0, :, :].cpu().numpy()
        label_mask = labels[0, 0, :, :].cpu().numpy()
        predicted_mask_proba = outputs[0, 0, :, :].float().detach().cpu().numpy()
        predicted_mask = (outputs[0, 0, :, :]> 0.5).float().detach().cpu().numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(input_image, cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(label_mask, cmap='gray')
        plt.title('Label Mask')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(predicted_mask_proba, cmap='gray')
        plt.title('Proba Predicted Mask')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.show()

        print(dice_coefficient(outputs, labels).item())

        example_showed = True



dataloader.close()
print(f'Mean Dice Coefficient: {100 * sum(val_dice_coeff) / len(val_dice_coeff):.1f}%')

torch.save(my_nn.state_dict(), 'unet_model.pth')