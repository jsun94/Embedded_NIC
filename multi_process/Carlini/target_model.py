from multiprocessing import Process, Queue, Lock
import multiprocessing
import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from thundersvm import OneClassSVM
import signal
from sklearn import svm
import joblib
import derived_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


class CarliniModel(nn.Module):
    def __init__(self):
        super(CarliniModel, self).__init__()
        self.layer1 = torch.nn.Conv2d(1,32,(3,3))
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Conv2d(32,32,(3,3))
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.MaxPool2d(kernel_size=2)

        self.layer6 = torch.nn.Conv2d(32,64,(3,3))
        self.layer7 = torch.nn.ReLU()
        self.layer8 = torch.nn.Conv2d(64,64,(3,3))
        self.layer9 = torch.nn.ReLU()
        self.layer10 = torch.nn.MaxPool2d(kernel_size=2)

        self.layer11 = torch.nn.Flatten()
        self.layer12 = torch.nn.Linear(1024,200)
        self.layer13 = torch.nn.ReLU()
        self.layer14 = torch.nn.Linear(200,200)
        self.layer15 = torch.nn.ReLU()
        self.layer16 = torch.nn.Linear(200,10)
        #self.layer17 = torch.nn.Softmax() ###

    def forward(self, x):	# each out size of byte : 64
        out = self.layer1(x)
        out = self.layer2(out)	#      
        out = self.layer3(out)         
        out = self.layer4(out)   # 
        out = self.layer5(out)   #    
        out = self.layer6(out)   
        out = self.layer7(out)	 #      
        out = self.layer8(out)           
        out = self.layer9(out)   #
        out = self.layer10(out)  #
        out = self.layer11(out)  #
        out = self.layer12(out)       
        out = self.layer13(out)  #
        out = self.layer14(out)       
        out = self.layer15(out)  #
        out = self.layer16(out)
        return out
                                




if __name__ == '__main__':	

    model = CarliniModel()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/carlini_process_final.pt'))
    model.eval()
    model.to(device)
    test_data = torch.rand(1,1,28,28).to(device)

    start_time = torch.cuda.Event(enable_timing = True)
    end_time = torch.cuda.Event(enable_timing = True)
    start_time.record()
    model(test_data)
    end_time.record()
    torch.cuda.synchronize()
    print(start_time.elapsed_time(end_time) / 1000, "sec")

