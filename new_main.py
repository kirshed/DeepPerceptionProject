import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from vgg import *
from utils import progress_bar

# functions for convert image to tensor for running net on changed images
from PIL import Image
from torchvision import transforms, models
def image_to_tensor(image_numpy, max_size=400, shape=None):
  
  # crop image if image is too big
  if max(image_numpy.size) > max_size:
    size = max_size
  else:
    size = max(image_numpy.size)
	
  size = (size, int(1.5*size))
  # if shape is given use it
  if shape is not None:
    size = shape
  
  # resize and normalize the image
  in_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  
  image = in_transform(image_numpy)[:3, :, :].unsqueeze(0)
  
  return image

def image_path_to_numpy(image_path):
  # load image into a numpy array from the given path
  return Image.open(image_path).convert('RGB') 

class HeatDataset(torch.utils.data.Dataset):
  # 'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        # 'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        X = image_path_to_numpy(f"./heatmap_imgs/{ID}.png")
        X = image_to_tensor(X)
        y = self.labels[index]

        return X, y

train_ids = list(range(0, 4000))
with open("./labels.txt", 'r') as f:
  labels = f.read().split('\n')
labels = labels[:-1]
# casting labels to ints 
for i in range(0, len(labels)):
    labels[i] = int(labels[i])

train_labs = labels[:4000]
test_ids = list(range(4000,5000))
test_labs = labels[4000:]

trainset = HeatDataset(train_ids, train_labs)
testset = HeatDataset(test_ids, test_labs)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
new_net = VGG('VGG19')
new_net = new_net.to(device)
if device == 'cuda':
    new_net = torch.nn.DataParallel(new_net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(new_net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    new_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = new_net(inputs[0])
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    new_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = new_net(inputs[0])
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'new_net': new_net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 50):
    train(epoch)
    test(epoch)
    scheduler.step()