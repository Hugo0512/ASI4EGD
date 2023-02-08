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
model=torch.load('COE_best_six_model4.pth')
class CatsDogsDataset(Dataset):
    """
    基于Sequence的自定义Keras数据生成器
    """



    def __init__(self,patient_list_txtfile,patient_list_txtfile1,shuffle=True):
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
            position1 = line.find('.bmp')

            if position == -1:
                position = position1

            templabel1 = line[position + 5:].replace("\n", "")
            # print(line[1:position+4])
            # print(int(templabel1))
            temp1.append(line[0:position + 4])
            temp2.append(int(templabel1))
        fid.close()

        fid = open(patient_list_txtfile1, 'r', encoding='utf-8')

        for line in fid.readlines():
            position = line.find('.jpg')
            position1 = line.find('.bmp')

            if position == -1:
                position = position1

            templabel1 = line[position + 5:].replace("\n", "")
            # print(line[1:position+4])
            # print(int(templabel1))
            temp1.append(line[0:position + 4])
            temp2.append(int(templabel1))
        fid.close()
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
            traindata=tempnpysample
            traindata = traindata.transpose((2, 0, 1))
            traindata = traindata.astype(np.float32)
            # traindata=np.expand_dims(tempnpysample,axis=0)
            trainlabel=self.labels[idx]
            # tempnpysample=tempnpysample/255;


        # traindata = scaler.transform(traindata)
        # traindata1 = scaler.fit_transform(traindata1).
        # trainlabel=trainlabel.reshape(trainlabel.shape[0])
        # trainlabel = to_categorical(trainlabel,2)
        return traindata,trainlabel


    def __len__(self):
        """每个epoch下的批次数量
        """
        return len(self.samplename)

valid_data = CatsDogsDataset(patient_list_txtfile='./six/test4.txt',patient_list_txtfile1='./six1/test4.txt',shuffle=True)


valid_loader = DataLoader(dataset = valid_data, batch_size=1, shuffle=False)
model=model.to(device)
filename='CE_OE_six_best_prob4.txt'
confusion_maxtrix_filename='CE_OE_six_best_confusion_matrix4.txt'
all_prob=[]
confusion_matrix=np.zeros((6,6))
model.eval()
with torch.no_grad():
    for val_data, val_label in valid_loader:
        val_data = torch.tensor(val_data, dtype=torch.float32)
        val_label = torch.tensor(val_label, dtype=torch.long)
        val_data = val_data.to(device)
        val_label = val_label.to(device)

        val_output = model(val_data).to(device)
        val_pred_prob = F.softmax(val_output, dim=1)
        val_pred_prob=val_pred_prob.cpu().numpy()
        # print(type(val_pred_prob))
        pred_result = np.argmax(val_pred_prob,axis=1)
        # print(val_pred_prob.shape)
        all_prob.append(val_pred_prob)
        val_label=val_label.cpu().numpy()
        # print(val_label)
        confusion_matrix[val_label,pred_result]=confusion_matrix[val_label,pred_result]+1
all_prob=np.array(all_prob)
all_prob=np.squeeze(all_prob)
print(all_prob.shape)
np.savetxt(filename,all_prob)
np.savetxt(confusion_maxtrix_filename,confusion_matrix)
