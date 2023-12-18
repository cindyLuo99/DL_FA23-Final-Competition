import os
import torch
import numpy as np
import torchmetrics 
from torch import nn, optim
from torch.utils.data import random_split,Dataset,DataLoader 
from torchvision import tv_tensors
from torchvision.transforms import v2
import random 
import torch.nn.functional as F
import PIL.Image

class Semantic_Segmentation_Dataset(Dataset):

    def __init__(self, mask_list,input_transform = None,target_transform = None):
        self.mask_list = mask_list
        self.input_transform = input_transform 
        self.target_transform = target_transform 

    def __len__(self):
        return len(self.mask_list)
    

    def transform(self, image,mask):
        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        identical_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(size=(160,240),padding=(40,60),padding_mode='edge'),
            v2.RandomHorizontalFlip(p=.5),
            v2.RandomVerticalFlip(p=.5),
            v2.RandomRotation(degrees=45),
        ])

        new_img,new_msk = identical_transform((image,mask)
                                      )
        return new_img,new_msk
    
    def __getitem__(self, idx):
        image = self.mask_list[idx][0]
        target = self.mask_list[idx][1]

        if self.input_transform:
            image = self.input_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        image,target = self.transform(image,target)

        return image, target
    
def split_video_mask(video_folder):
    '''split_video_mask Creats a list of tuples (image,mask)

    :param video_folder: directory to video folder
    :type video_folder: path
    :return: list of tuples of form (pytorch tensor, pytorch tensor) (size [3, 160, 240],[160, 240])
    '''
    directory = video_folder
    
    img_mask_dict = {}
    list_of_segments = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            index = str(os.path.basename(file)[6:-4])
            img = tv_tensors.Image(PIL.Image.open(os.path.join(directory,file)))
            img_mask_dict[index] = img.to(torch.get_default_dtype())
 

        if filename.endswith(".npy"):
            #print(os.path.join(directory, filename))
            mask = torch.from_numpy(np.load(os.path.join(directory,file))).to(torch.get_default_dtype())
            mask = tv_tensors.Image(mask)
            

    for key,val in img_mask_dict.items():
        list_of_segments.append((val,tv_tensors.Image(mask[int(key)])))

    
    return list_of_segments

def load_data(video_folder):  
    
    subdir_list = os.listdir(video_folder)
    mask_list = []

    for subdir in subdir_list:
        train_mask_list = split_video_mask(os.path.join(video_folder,subdir))
        mask_list = mask_list + train_mask_list


    data_sample = random.sample(mask_list,400)
    input_mean = torch.mean(torch.Tensor(data_sample[0][0]),dim=[1,2])
    input_std = torch.std(torch.Tensor(data_sample[0][0]),dim=[1,2])


    input_transform = v2.Normalize(input_mean,input_std)


    ss_dataset = Semantic_Segmentation_Dataset(mask_list, 
                                            input_transform=input_transform,
                                            target_transform=None
                                            )

    generator = torch.Generator().manual_seed(10)

    train_set, test_set = random_split(ss_dataset,[.7, .3],generator=generator)

    return train_set, test_set

def train_practice(hyperparameters, train_subset,val_subset):
    
    #net = PracticeNet(hyperparameters["chan_1"])
    net = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights=None)
    net.classifier._modules['4'] = torch.nn.Conv2d(256, 49, kernel_size=(1, 1), stride=(1, 1))
    optimizer = optim.Adam(net.parameters(),lr=hyperparameters["lr"])
    
    if os.path.isfile("./mask_model_stop.pth") and os.path.getsize("./mask_model_stop.pth") > 0:
        checkpoint = torch.load("./mask_model_stop.pth")
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        running_loss = checkpoint['loss']
        best_val_jac = checkpoint['best_val_jac']
        print(f"Resuming training from epoch: {start_epoch}")
    else: 
        best_val_jac = 0
        start_epoch = 0 
        running_loss = 0 

    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    criterion = jaccard


    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

    print(f"Training on {device}")
    net.to(device)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    height = train_subset[0][1].shape[0]
    width = train_subset[0][1].shape[1]
   
    trainloader = DataLoader(
        train_subset, batch_size=int(hyperparameters["batch_size"]), shuffle=True, drop_last=True,
    )
    valloader = DataLoader(
        val_subset, batch_size=len(val_subset), 
    )

    for epoch in range(start_epoch,hyperparameters["max_epochs"]):
        net.train()
        batch_loss = 0
        count = 0
        for batch in trainloader:
            count += 1
            data = batch[0].to(device)
            labels = batch[1].to(device).squeeze()
            optimizer.zero_grad()

            predicted = torch.nn.functional.softmax(net(data)['out'],dim=1)
            loss = jaccard(predicted,labels).requires_grad_(True)
            batch_loss += batch_loss + loss.item()
            loss.backward()
            optimizer.step

        running_loss = batch_loss/count 
            
        torch.save({'epoch': epoch+1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        'best_val_jac':best_val_jac,
        }, "mask_model_stop.pth")


        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total_pixels = 0
        batch_cnt = 1

        for batch in valloader:
            net.eval()
            with torch.no_grad():
                data = batch[0].to(device)
                labels = batch[1].to(device).squeeze()

                predicted = F.softmax(net(data)['out'],dim=1)
                predicted = torch.argmax(predicted,dim=1)       #(Shape N,H,W)
     
                loss = criterion(predicted, labels)
                val_jac = 100 * jaccard(predicted,labels)

                # Save the best model
                if val_jac > best_val_jac:
                    best_val_jac = val_jac
                    torch.save({'epoch': epoch+1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                    'best_val_jac':best_val_jac,
                    }, "best_mask_model.pth")

                val_loss += loss.cpu().numpy()
                val_steps += 1
                batch_size = predicted.shape[0]
                correct_pixel = 0 
                
                for k in range(batch_size):     #should be 1
                    for i in range(height):
                        for j in range(width):
                            if predicted[k][i][j].item() == labels[k][i][j].item():
                                correct_pixel += 1  
                    
            val_jac = 100 * jaccard(predicted,labels)
            batch_cnt += 1
            total_pixels = batch_cnt * height * width
            if epoch % 5 == 0:
                print(f"Epoch: {epoch}, Validation Accuracy (Jaccard): {val_jac:.2f}%, Pixel Accuracy: {100*correct_pixel/total_pixels:.2f}%")
            




if __name__ == "__main__":
    #Loading Data
    train_location = "./project_data/dataset/Dataset_Student/train/"
    val_location = "./project_data/dataset/Dataset_Student/val/"
    train_subset, val_subset = load_data(train_location)
    hyperparameters = {"lr":0.001,"batch_size":32,"max_epochs":500}

    model_exists = os.path.isfile("./best_mask_model.pth") and os.path.getsize("./best_mask_model.pth") > 0
    threshold = .8

    if model_exists:
        #load model
        checkpoint = torch.load("./mask_model_stop.pth")

        #Loading Net
        net = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights=None)
        net.classifier._modules['4'] = torch.nn.Conv2d(256, 49, kernel_size=(1, 1), stride=(1, 1))
        optimizer = optim.Adam(net.parameters(),lr=hyperparameters["lr"])
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #Check best accuracy
        best_val_jac = checkpoint['best_val_jac']
        model_below_threshold = best_val_jac < threshold


        if model_below_threshold:   #train more
            train_practice(hyperparameters,train_subset,val_subset)
        else: #model above threshold 
            pass 
    else: #no model
        train_practice(hyperparameters,train_subset,val_subset)

    #Model is Trained
    net = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights=None)
    net.classifier._modules['4'] = torch.nn.Conv2d(256, 49, kernel_size=(1, 1), stride=(1, 1))
    optimizer = optim.Adam(net.parameters(),lr=hyperparameters["lr"])
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_jac = checkpoint["best_val_jac"]
    print(f"Final validation accuracy (Jaccard): {best_val_jac:.2f}%")
