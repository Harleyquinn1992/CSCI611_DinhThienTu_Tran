# Dinh Thien Tu Tran - Spring 2026, Chico
# These block of codes are demo for my assignment for CSCI611 class.
# Skeleton codes are provided by professor Bo Shen.
# The goal of this assignment is to teach me about CNN.

import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
matplotlib.use('TkAgg')  # or Qt5Agg
import matplotlib.pyplot as plt
#For CNN
import torch.nn as nn
import torch.nn.functional as F 

# -----------------------
# 1 LOAD TEST DATA
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2.
# CNN ARCHITECTURE
# -------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        
        self.pool = nn.MaxPool2d(2, 2) # reduce h,w by half
        
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256,10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
    
        x = x.view(x.size(0), -1)
        # optional add dropout layer
        x = self.dropout(x)
        
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # optional add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

# create a complete CNN
model = Net().to(device)
model.load_state_dict(torch.load("model_trained.pt", map_location=device))
model.eval()

#--------------------
# 3 DENORMALIZE IMAGE
#--------------------
# Note to self: What is this for?
# in A2T1.py, we converted data to normalize torch.FloatTensor
# transform = transforms.Compose([
#  transforms.ToTensor(),
#  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
# that means every pixel was transformed.
# CIFAR images are in [0,1], the normalization turns them to [-1,1]
# Because neural network train better when inputs are centered around 0
# So we need to demnormalize as we want to display images with matplotlib.
# It expects values in [0,1] for float images, [0, 255] for integer images.

def denormalize(img_tensor):
    return (img_tensor * 0.5 + 0.5).clamp(0,1)

#-----------------
# 4. PICK IMG TEST
#-----------------

picked = {}
for idx in range(len(test_data)):
    img, label = test_data[idx]
    if label not in picked:
        picked[label] = (img, label)
    if len(picked) == 3:
        break

# IF want to pick random
# import random

#picked = {}
#indices = list(range(len(test_data)))
#random.shuffle(indices)

#for idx in indices:
#    img, label = test_data[idx]
#    if label not in picked:
#        picked[label] = (img, label)
#    if len(picked) == 3:
#        break

picked_items = list(picked.values()) # 3 items(img, lablel)

#------------------------------------------
# 5. EXTRACT conv1 feature maps + visualize
#------------------------------------------
with torch.no_grad():
    for img, label in picked_items:
        x = img.unsqueeze(0).to(device) # [3, 32, 32] -> [1, 3, 32, 32]
                                        # add 1 dimension
                                        # CNN expect input: [Batch, Channels, H, W]
        fmap = model.conv1(x)  # [1, 32, 32, 32] 
        fmap = fmap.squeeze(0) # [32, 32, 32] : remove 1 dimension
                               # only doable if the dimension has value 1

        # Plot input and 8 feature maps
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(f"Input class: {classes[label]} | Layer: conv1 feature maps", fontsize=14)
        
        # input image
        ax0 = fig.add_subplot(2, 5, 1)
        ax0.imshow(denormalize(img).permute(1,2,0).cpu().numpy())
        ax0.set_title("Input")
        ax0.axis("off")

        # 8 feature maps (channels 0..7)
        for ch in range(8):
            ax = fig.add_subplot(2, 5, ch + 2)
            ax.imshow(fmap[ch].cpu().numpy(), cmap="gray")
            ax.set_title(f"conv1 ch{ch}")
            ax.axis("off")
        
        #plt.tight_layout()
        #plt.show()

#---
# Part B
#---

# Choose 2nd layer
# choose any 3 filters from conv2

import heapq
import itertools
filter_ind = [0, 25, 56]
k=5

from torch.utils.data import DataLoader
# [64, 3, 32, 32] grabbing 64 images at once
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=3)

# store top 5 results for each filter and only 5
# create empty heap
# heaps = {0: [], 7: [], 15: []}
heaps = {f: [] for f in filter_ind}

model.eval()
counter = itertools.count()

with torch.no_grad():
    for imgs, labels in test_loader: # 64 images per loop
        imgs = imgs.to(device)
        labels = labels.to(device)

        # forward to conv2 after ReLU
        x = F.relu(model.conv1(imgs))
        x = model.pool(x)
        acts = F.relu(model.conv2(x)) #[64, 64, H, W] 64 images, 64 filters

        for f in filter_ind:
            fmap = acts[:, f, :, :]
            scores = fmap.amax(dim=(1,2))

            for i in range(len(scores)):
                s = float(scores[i].cpu())
                img_cpu = imgs[i].cpu()
                lab = int(labels[i].cpu())

                # If detect any activation larger than any of the 5 in heap, replace the smallest with it
                if len(heaps[f]) < k:
                    heapq.heappush(heaps[f], (s, next(counter), img_cpu, lab))
                else:
                    if s > heaps[f][0][0]:
                        heapq.heapreplace(heaps[f], (s, next(counter), img_cpu, lab))
# sort and display
for f in filter_ind:
    top_images = sorted(heaps[f], key=lambda t: t[0], reverse=True)

    fig = plt.figure(figsize=(12,3))
    fig.suptitle(f"conv2 | Filter {f} | Activation = MAX after ReLU")

    for j, (score, _, img, lab) in enumerate(top_images):
        ax = fig.add_subplot(1, 5, j+1)
        ax.imshow(denormalize(img).permute(1,2,0).numpy())
        ax.set_title(f"{classes[lab]}\n{score:.3f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()