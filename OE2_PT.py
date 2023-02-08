from __future__ import print_function
import cv2
import glob
from itertools import chain
import os
import random
import zipfile
from torch.nn import DataParallel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from torchvision import models
device='cuda'
epochs=5
class CatsDogsDataset(Dataset):
    """
    基于Sequence的自定义Keras数据生成器
    """



    def __init__(self,patient_list_txtfile,shuffle=True):
    # def __init__(self):

        """ 初始化方法
        :param splitflag；区分train还是validation
        :param patient_list_txtfile,save the txt file of sample order,such as train1.txt test1.txt
        :param shuffle: 每一个epoch后是否打乱数据
        :param batch_size: 每一个epoch中clips的个数
        :param label_file_csv: the label of each sample
        :param secondary_label_filename:去掉了不能用的样本的标签文件名
        """
        #read txt file to obtain all sample

        # labels = pandas.read_csv(label_file_csv)
        # patient_list_txtfile = patient_list_txtfile

        temp1 = [];
        temp2 = [];
        fid = open(patient_list_txtfile, 'r', encoding='utf-8')

        for line in fid.readlines():
            position = line.find('.jpg')

            templabel1 = line[position + 5:].replace("\n", "")
            # print(line[1:position+4])
            # print(int(templabel1))
            temp1.append(line[0:position + 4])
            temp2.append(int(templabel1))

        self.samplename = temp1
        self.shuffle = shuffle

        self.originalindexes = np.arange(len(self.samplename))
        self.labels = temp2

        print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
        # print(max(self.labels))
        print(len(temp1))
        print(len(temp2))






    def __getitem__(self, idx):  # 包括输入x和输出y
        """生成每一批次的图像
        :param list_IDs_temp: 批次数据索引列表
        :return: 一个批次的图像"""
        # 初始化
        originalsizey=originalsizex=512
        traindata = None
        trainlabel = None

        # 生成数据

            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(index1)
            # print(self.samplename[index1])
        # print(self.samplename[idx])
        # print(self.samplename[idx])
        # print(self.samplename[idx].replace('/media/tx-deepocean/5271ceaf-10dc-456f-b54c-3165996444a518/western_pathology','.'))
        # print(self.samplename[idx])
        if os.path.exists(self.samplename[idx]):
            tempnpysample=cv2.imread(self.samplename[idx])
            tempnpysample =cv2.resize(tempnpysample,(originalsizey,originalsizex))
            # if tempnpysample is None:
            #     print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            traindata=tempnpysample
            traindata = traindata.transpose((2, 0, 1))
            traindata = traindata.astype(np.float32)
            # traindata=np.expand_dims(tempnpysample,axis=0)
            trainlabel=self.labels[idx]
            # print(traindata.shape)
            # print(trainlabel)
            # tempnpysample=tempnpysample/255;
            # if trainlabel is None:
            #     print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")

        # traindata = scaler.transform(traindata)
        # traindata1 = scaler.fit_transform(traindata1).
        # trainlabel=trainlabel.reshape(trainlabel.shape[0])
        # trainlabel = to_categorical(trainlabel,2)
        return traindata,trainlabel


    def __len__(self):
        """每个epoch下的批次数量
        """
        return len(self.samplename)
train_data = CatsDogsDataset(patient_list_txtfile='./binary1/train4.txt',shuffle=True)
valid_data = CatsDogsDataset(patient_list_txtfile='./binary1/test4.txt',shuffle=True)

train_loader = DataLoader(dataset = train_data, batch_size=64, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=32, shuffle=True)

# model = vit3d_pytorch.ViT3D(
#             image_size=(256, 256, 256),
#             patch_size=32,
#             num_classes=3,
#             dim=1024,
#             depth=6,
#             heads=16,
#             mlp_dim=2048,
#             dropout=0.1,
#             emb_dropout=0.1
#     ).to(device)

model=models.resnet101(pretrained=True)
# use_gpu=torch.cuda.is_available()
num_features=model.fc.in_features
model.fc=nn.Linear(num_features,2)
model=model.cuda()
num_gpu=list(range(torch.cuda.device_count()))
# torch.is_distributed.init_process_group('nccl',init_method='file:///home/.../my_life',world_size=1,rank=0)
model = torch.nn.parallel.DataParallel(model, device_ids=num_gpu)
# gpus=[0,1,2];
# model=DataParallel(model,device_ids=gpus,output_device=gpus[0])
# loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,5]).to(torch.float32).cuda())
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_accuracy=float('-inf')
for epoch in range(epochs):
    # epoch_loss = 0
    # epoch_accuracy = 0
    model.train()
    total_correct_number=0
    total_sample_number=0
    total_loss=0
    for data, label in tqdm(train_loader,disable=True):
        data=torch.tensor(data, dtype=torch.float32)
        label=torch.tensor(label, dtype=torch.long)
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        pred_prob=F.softmax(output,dim=1)
        # output=output.cpu()
        # output=output.detach().numpy()
        # print(output)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_number = (pred_prob.argmax(dim=1) == label).float().sum()
        # epoch_accuracy += acc / len(train_loader)
        # epoch_loss += loss / len(train_loader)
        total_correct_number+=correct_number
        total_sample_number+=label.shape[0]
        total_loss+=loss*label.shape[0]
    print("+++++++++++++++++++++++++++",total_loss/total_sample_number,total_correct_number/total_sample_number)
    model.eval()
    with torch.no_grad():
        # epoch_val_accuracy = 0
        # epoch_val_loss = 0
        val_sample_number=0
        val_total_correct_number=0
        val_total_loss=0
        for val_data, val_label in valid_loader:
            val_data = torch.tensor(val_data, dtype=torch.float32)
            val_label = torch.tensor(val_label, dtype=torch.long)
            val_data = val_data.to(device)
            val_label = val_label.to(device)

            val_output = model(val_data)
            val_pred_prob = F.softmax(val_output, dim=1)
            val_loss = criterion(val_output, val_label)

            val_correct_number = (val_pred_prob.argmax(dim=1) == val_label).float().sum()
            # epoch_val_accuracy += val_acc / len(valid_loader)
            # epoch_val_loss += val_loss / len(valid_loader)
            val_total_correct_number += val_correct_number
            val_sample_number += val_label.shape[0]
            val_total_loss += val_loss * val_label.shape[0]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~", val_total_loss / val_sample_number,val_total_correct_number / val_sample_number)
        if (val_total_correct_number / val_sample_number)>best_accuracy:
            best_accuracy=(val_total_correct_number / val_sample_number)
            torch.save(model, 'ordinary_best_model4.pth')
    # print(
    #     f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    # )
# torch.save(model,'model4.pth')
