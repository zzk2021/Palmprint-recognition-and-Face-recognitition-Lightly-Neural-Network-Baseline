from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import cv2
import os

class TxtImage(Dataset):
    def __init__(self,label,dataRoot,transform=None,size=(120,120),index=0):
        self.tranform=transform
        self.dataRoot=dataRoot
        self.size=size
        self.imgList=[]
        self.labelList=[]
        self.index=index
        for xx in label:
            x =xx.split(' ')
            self.imgList.append(x[0])
            self.labelList.append(int(x[1])-1)

    def __getitem__(self,index):
        imgName=self.imgList[index]
        imgName_=list(imgName)
        imgName_.insert(self.index,'_aligned')
        imgName="".join(imgName_)
        imgPath=os.path.join(self.dataRoot,imgName)
        img = cv2.imread(imgPath)
        if self.size is not None:
            img=cv2.resize(img,self.size)
        if self.tranform is not None:
            img=self.tranform(img)
        label=torch.IntTensor([self.labelList[index]])
        return img,label

    def __len__(self):
        return len(self.imgList)
