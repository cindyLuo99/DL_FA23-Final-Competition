import os
import torch
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import random
import torchmetrics
import torch.nn.functional as F
from torch import nn
# import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

device = torch.device('cuda:0')

pre_seq_length = 11
aft_seq_length = 11

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

in_shape = [11, 49, 160, 240]
learningRate = 0.0005
batchSize = 20
val_batchSize = 32
hid_S = 64
hid_T = 256 
N_S = 4
N_T = 8
groups = 7
epochs = 50
log_step = 5
train_step = 10
# if continue training from an old model:
# old_model = 'Sim_VP_best_model_segs_new_loss_9.pth'

last_frame_weight = 4
# last_frame_weight = None

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="deepLearning",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": learningRate,
#     "architecture": "SimVP",
#     "dataset": "videoSegs",
#     "loss": "cross-entropy",
#     "last_frame_weight": last_frame_weight,
#     "model_name" : f'test_{train_step}_jaccard_modified',
#     "epochs": epochs,
#     "scheduler": "On",
#     "scheduler_type": 'OneCycleLR',
#     "in_shape": in_shape,
#     "batch_size" : batchSize,
#     "val_batch_size": val_batchSize,
#     "hid_S": hid_S, 
#     "hid_T": hid_T,
#     "N_S": N_S,
#     "N_T": N_T,
#     "groups": groups,
#     "log_step": log_step,
#     "train_from_model": old_model
#     }
# )

class VideoSegDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.video_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]

        # Load segmentation masks
        mask_path = os.path.join(video_folder, 'mask.npy')
        masks = torch.from_numpy(np.load(mask_path)).long()  # Shape [22, 160, 240]
        
        # Split into input and target masks
        input_masks = masks[:11]   # First 11 masks
        target_masks = masks[11:]  # Last 11 masks

        # Add channel dimension
        # One-hot encode
        input_masks = F.one_hot(input_masks, num_classes=49).permute(0, 3, 1, 2).float() # Shape [11, 49, 160, 240]

        return input_masks, target_masks

# Example usage
train_dataset = VideoSegDataset(root_dir='train/')
val_dataset = VideoSegDataset(root_dir='val/')

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=val_batchSize, shuffle=True, pin_memory=True)

# Model Architecture Code adapted from https://github.com/A4Bio/SimVP-Simpler-yet-Better-Video-Prediction
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y

class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y
    
def stride_generator(N, reverse=False):
    '''
    ### 1. `stride_generator` Function
    This function generates a list of stride values for convolutional layers. 
    A stride determines how far a convolutional filter moves across the input. In this function:
    - `strides = [1, 2]*10` creates a list of alternating 1s and 2s.
    - `N` specifies how many elements from this list to use.
    - If `reverse` is `True`, the list is reversed; otherwise, it's used as is.
    '''
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N] # N should be less than 20

class Encoder(nn.Module):
    '''
    ### 2. `Encoder` Class
    This class defines an encoder module, typically used to downsample and extract features from the input.
    - `C_in` and `C_hid` are the number of input and hidden channels, respectively.
    - `N_S` is the number of strides, used to generate stride values.
    - `self.enc` is a sequential container of convolutional layers (`ConvSC`), where the first layer's stride is `strides[0]` and subsequent layers' strides are determined by the remaining elements in `strides`.
    '''
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    '''
    ### 3. `Decoder` Class
    This class defines a decoder module, which typically upsamples the feature maps to reconstruct or generate output.
    - It uses the same stride values as the encoder but in reverse order. This symmetry is common in encoder-decoder architectures.
    - `ConvSC` with `transpose=True` indicates transposed convolution, often used for upsampling.
    - The final layer concatenates its input with the feature map `enc1` from the encoder, which is a form of skip connection, often used to help the network better learn fine-grained details.
    '''
    def __init__(self,C_hid, C_out, N_S):
        super(Decoder,self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)
    
    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    '''
    ### 4. `Mid_Xnet` Class
    This class appears to be a middle processing network with multiple `Inception` modules.
    - It alternates between encoding and decoding Inception modules, with skip connections from encoder layers to corresponding decoder layers.
    '''
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    '''
    ### 5. `SimVP` Class
    This is the main model class combining the Encoder, Mid_Xnet, and Decoder.
    - `shape_in` describes the shape of the input tensor.
    - The model first reshapes the input and passes it through the encoder.
    - The encoded output is reshaped again and passed through `Mid_Xnet`.
    - Finally, the output of `Mid_Xnet` is reshaped and decoded to produce the final output.
    '''
    def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y
    
def train(model, device, train_loader, optimizer, epoch, last_frame_weight=last_frame_weight):
    print('Training...')
    model.train()
    train_loss = []
    train_pbar = tqdm(train_loader)
    for data, target in train_pbar:
        data, target = data.to(device), target.to(device)
        output = model(data) 
        
        if last_frame_weight is not None:
            # Separate the last frame (if want to add more weights to the last frame)
            preceding_output = output[:, :-1].reshape(-1, 49, 160, 240)
            last_output = output[:, -1].reshape(-1, 49, 160, 240)
            preceding_target = target[:, :-1].reshape(-1, 160, 240)
            last_target = target[:, -1].reshape(-1, 160, 240)

            # Compute loss for preceding frames and the last frame
            loss_preceding = criterion(preceding_output, preceding_target)
            loss_last = criterion(last_output, last_target) * last_frame_weight # 2 here

            # Normalize the loss
            loss = (loss_preceding * (10) + loss_last) / (10 + last_frame_weight)
        
        else:
            output = output.reshape(-1, 49, 160, 240)
            target = target.reshape(-1, 160, 240)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

    train_loss = np.mean(train_loss)
    print('Train Epoch: {}  Loss: {:.6f}'.format(epoch, train_loss.item()))
    return train_loss

def validate(model, device, val_loader, epoch):
    print('Validating...')
    model.eval()
    total_loss = []
    val_pbar = tqdm(val_loader)
    jaccard = torchmetrics.JaccardIndex(num_classes=49, task="multiclass").to(device)  # Initialize Jaccard Index (IoU)
    
    for i, (data, target) in enumerate(val_pbar):
        if i * data.shape[0] > 1000:
            break

        data, target = data.to(device), target.to(device)
        with torch.no_grad():  # No need to track gradients during validation
            pred = model(data)
            
            # Compute Jaccard Index (IoU)
            # pred must be converted to class indices for IoU calculation
            pred_last = pred[:,-1,:,:,:].squeeze()    #batch,49,160,240
            pred_classes = pred_last.argmax(dim=1)
            
            last_frame_target = target[:,-1,:,:]
            jaccard.update(pred_classes, last_frame_target)
                
            pred = pred.view(-1, 49, 160, 240)  # Reshape for CrossEntropyLoss
            target = target.view(-1, 160, 240)  # Reshape target for CrossEntropyLoss
            loss = criterion(pred, target)
            val_pbar.set_description('vali loss: {:.4f}'.format(loss.item()))
            total_loss.append(loss.item())

    # Compute average IoU after processing all batches
    avg_iou = jaccard.compute().item()
    jaccard.reset()
    
    # Average loss across all batches
    total_loss = torch.tensor(total_loss).mean().item()  # Convert tensor to scalar
    
    print(f'Validation - Average loss: {total_loss:.4f}, Average IoU: {avg_iou:.4f}')
    return total_loss, avg_iou

model = SimVP(tuple(in_shape), hid_S, hid_T, N_S, N_T, [3,5,7,11], groups).to(device)
# if training from an old model
# model.load_state_dict(torch.load(f'saved_models_seg_py/{old_model}'))
# print(f'Started traininig model from {old_model}')
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7, verbose=True)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learningRate, steps_per_epoch=len(train_loader), epochs=epochs)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.CrossEntropyLoss()

# best_loss = float('inf') 
best_iou = -float('inf')
train_losses, val_losses = [], []

validation_frequency = 5

for epoch in range(epochs):
    train_loss = train(model, device, train_loader, optimizer, epoch)
    train_losses.append(train_loss)
    with torch.no_grad():
        val_loss, avg_iou = validate(model, device, val_loader, epoch)
        val_losses.append(val_loss)
    
    # Optional: Log metrics
    # wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "avg_iou": avg_iou, "lr": optimizer.param_groups[0]['lr']})

    # Save model if validation loss improved
#     if val_loss <= best_loss:
#         best_loss = val_loss
    # Save model if iou loss improved
    if avg_iou >= best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), f'Sim_VP_best_model_segs_new_loss_{train_step}.pth')

# wandb.finish()

print('Train Losses')
print(train_losses)
print('Val Losses')
print(val_losses)
