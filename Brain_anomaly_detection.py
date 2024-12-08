import cv2
import os
import numpy as np
import random
import torch
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
VALPROP = 0.1
TESTPROP = 0.1
DICE_PROP = 0.8
SMOOTH = 0.5
BATCHSIZE = 16
LEARNING_RATE = 0.001
MOMENTUM = 0.9
EPOCHS = 30
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
        self.enc2 = self.contracting_block(32, 64,3,2) # Use strided convolution instead of Maxpool
        self.enc3 = self.contracting_block(64, 128,3,2)
        self.enc4 = self.contracting_block(128, 256,3,2)
        self.enc5 = self.contracting_block(256, 512,3,2)

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

        self._initialize_weights()

    def contracting_block(self, in_channels, out_channels,size,stride=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=size,stride=stride, padding='same'),
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
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)        

        # Decoder
        dec4 = self.upconv4(enc5)
        dec4 = self.crop_and_concat(dec4, enc4, crop=True)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.crop_and_concat(dec3, enc3, crop=True)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.crop_and_concat(dec2, enc2, crop=True)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.crop_and_concat(dec1, enc1, crop=True)
        dec1 = self.dec1(dec1)

        out = self.final_conv(dec1)
        out = self.sigmoid(out)

        return out

class EarlyStopping:
    def __init__(self, monitor='val_loss', patience=5, restore_best_weights=True):
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, model, current_score, current_epoch):
        if self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_weights:
                    model.load_state_dict(self.best_model_weights)
    
class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = None

    def __call__(self, model, current_score):
        if not self.save_best_only or self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            torch.save(model.state_dict(), self.filepath)
            print(f"Model saved to {self.filepath} \n")
 
################ Functions ###############################################################################################################################

#Defining the Dice Loss
def dice_coefficient_proba(y_true, y_pred, smooth=SMOOTH):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_coefficient(y_true, y_pred, smooth=SMOOTH):
    y_pred = (y_pred  > 0.5).float()
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def total_loss(y_true, y_pred, smooth=SMOOTH):
    loss = DICE_PROP*(1 - dice_coefficient_proba(y_true, y_pred, smooth)) + (1-DICE_PROP)*nn.BCELoss()(y_pred, y_true)
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

train_ds, val_ds, test_ds = random_split(equil_dt, [TRAINPROP, VALPROP, TESTPROP])

unet = UNet().to(device)
print(unet)

criterion = total_loss

optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

################ Trains & Validation ###############################################################################################################################

# Define data loaders for training and validation
train_dl = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=True)

# Instantiate callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('unet_model.pth', monitor='val_loss', save_best_only=True)

# Visualization list and flag for saving the label
ref_name = ''
name_saved = False

for epoch in range(EPOCHS):
    train_loss = []
    train_dice_coeff = []
    dataloader = tqdm(train_dl, position=0, leave=True)

    # Training Phase
    unet.train()
    for inputs, labels, name in dataloader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = unet(inputs)
        
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_dice_coeff.append(dice_coefficient(labels, outputs).item())

        dataloader.set_description(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {sum(train_loss) / len(train_loss):.5f} - Mean Training Dice Coefficient: {100 * sum(train_dice_coeff) / len(train_dice_coeff):.1f}%')        
        dataloader.refresh()

    scheduler.step()
    dataloader.close()
    del train_loss
    del train_dice_coeff

    # Validation Phase
    unet.eval()
    val_loss = []
    val_dice_coeff = []
    with torch.no_grad():
        for inputs, labels, name in val_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = unet(inputs)
            loss = criterion(labels, outputs)

            val_loss.append(loss.item())
            val_dice_coeff.append(dice_coefficient(labels, outputs).item())
        
        mean_val_loss = sum(val_loss) / len(val_loss)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {mean_val_loss:.5f} - Mean Validation Dice Coefficient: {100 * sum(val_dice_coeff) / len(val_dice_coeff):.1f}% \n')

    dataloader.close()
    del val_loss
    del val_dice_coeff

    # Callbacks
    early_stopping(unet, mean_val_loss, epoch)
    checkpoint(unet, mean_val_loss)

    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

print('Training finished')

################ Test ###############################################################################################################################

# Load the best model
unet = UNet().to(device)  
unet.load_state_dict(torch.load('unet_model.pth', weights_only=True))

test_dl = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=True)

unet.eval()
test_dice_coeff = []
example_showed = False
k = 0
for inputs, labels , name in test_dl:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = unet(inputs)
    test_dice_coeff.append(dice_coefficient(labels, outputs).item())
    


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

        print(f"Image's Dice Coefficient: {100*dice_coefficient(outputs, labels).item():.1f} %")

        example_showed = True


dataloader.close()
print(f'Mean Test Dice Coefficient: {100 * sum(test_dice_coeff) / len(test_dice_coeff):.1f}%')

