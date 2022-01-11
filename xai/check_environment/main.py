#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torchvision.utils as utils


# In[3]:


import os
import random
import time


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


outf = './result'


# In[6]:


nz = 100 #画像を生成するための特徴マップの次元数
nch_g = 64 #Generatorの最終層の入力チャネル数
nch_d = 64 #Discriminatorの先頭層の出力チャネル数
workers = 2 #データロードに使用するコア数
batch_size=50 #バッチサイズ
n_epoch = 30 #エポック数（繰り返し学習する回数）
lr = 0.0002 #学習率
beta1 = 0.5 #最適化関数に使用するパラメータ

display_interval = 100 #学習経過を表示するスパン


# In[36]:


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root='./', train=True,download=True,transform=transform)


# In[37]:


#データローダーを作成する
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))


# In[39]:


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 2))
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

net = Autoencoder().to(device)


# In[40]:

#損失関数
criterion = nn.MSELoss() #二乗誤差損失

#最適化関数
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5)


# In[42]:


def plot_loss (losses, epoch):
    plt.figure(figsize=(10,5))
    plt.title("Autoencoder Loss - EPOCH "+ str(epoch))
    plt.plot(losses,label="L")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#全体でのlossを可視化
def plot_loss_average (loss_mean):
    plt.figure(figsize=(10,5))
    plt.title("Autoencoder Loss - EPOCH ")
    plt.plot(loss_mean,label="G")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# In[43]:


def to_img(x):
    x = 0.5 * (x + 1)  # [-1,1] => [0, 1]
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


# In[48]:


loss_mean = [] #学習全体でのlossを格納するリスト(generator)
epoch_time = [] #時間の計測結果を格納するリスト

for epoch in range(n_epoch):
    start = time.time() #時間の計測を開始
    losses = [] #1エポックごとのlossを格納するリスト
    
    for itr, data in enumerate(dataloader):
    #本物画像のロード
        real_image = data[0].to(device) #本物画像をロード
        real_label = data[1].to(device) #本物画像のラベルをロード
        real_image = real_image.view(real_image.size(0), -1)

        #Autoencoderの更新
        net.zero_grad() #勾配の初期化

        output = net(real_image) #順伝播させて出力（分類結果）を計算
        err = criterion(output,real_image) #本物画像に対する損失値
        E = output.mean().item()

        err.backward()
        optimizer.step()

        # lossの保存
        losses.append(err.item())

        #学習経過の表示
        if itr % display_interval == 0:
            print('[{}/{}]{} Loss: {:.3f} '.format(epoch + 1, n_epoch,itr + 1, len(dataloader), E))

        if epoch == 0 and itr == 0:
            utils.save_image(real_image, '{}/real_samples.png'.format(outf),normalize=True, nrow=10)

  #確認用画像の生成（1エポックごと）
    pic = to_img(output)
    utils.save_image(pic.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),normalize=True, nrow=10)

  #lossの平均を格納
    loss_mean.append(sum(losses) / len(losses))

    #lossのプロット
    plot_loss (losses, epoch)

    #1エポックごとの時間を記録
    epoch_time.append(time.time()- start)

#学習全体の状況をグラフ化
plot_loss_average(loss_mean)

torch.save(net.state_dict(), './autoencoder.pth')