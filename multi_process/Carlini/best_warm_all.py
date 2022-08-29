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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


"""WARM UP MODEL"""
class CarliniModel2(nn.Module):
    def __init__(self):
        super(CarliniModel2, self).__init__()
        self.layer1 = torch.nn.Conv2d(3,64,(3,3))
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Conv2d(64,64,(3,3))
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.MaxPool2d(kernel_size=2)

        self.layer6 = torch.nn.Conv2d(64,128,(3,3))
        self.layer7 = torch.nn.ReLU()
        self.layer8 = torch.nn.Conv2d(128,128,(3,3))
        self.layer9 = torch.nn.ReLU()
        self.layer10 = torch.nn.MaxPool2d(kernel_size=2)

        self.layer11 = torch.nn.Flatten()
        self.layer12 = torch.nn.Linear(3200,256)
        self.layer13 = torch.nn.ReLU()
        self.layer14 = torch.nn.Linear(256,256)
        self.layer15 = torch.nn.ReLU()
        self.layer16 = torch.nn.Linear(256,10)
        #self.layer17 = torch.nn.Softmax() ###


    def forward(self, x):
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

class CarliniModel(nn.Module):
    def __init__(self):
        super(CarliniModel, self).__init__()
        self.layer1 = torch.nn.Conv2d(3,64,(3,3))
        self.layer2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Conv2d(64,64,(3,3))
        self.layer4 = torch.nn.ReLU()
        self.layer5 = torch.nn.MaxPool2d(kernel_size=2)

        self.layer6 = torch.nn.Conv2d(64,128,(3,3))
        self.layer7 = torch.nn.ReLU()
        self.layer8 = torch.nn.Conv2d(128,128,(3,3))
        self.layer9 = torch.nn.ReLU()
        self.layer10 = torch.nn.MaxPool2d(kernel_size=2)

        self.layer11 = torch.nn.Flatten()
        self.layer12 = torch.nn.Linear(3200,256)
        self.layer13 = torch.nn.ReLU()
        self.layer14 = torch.nn.Linear(256,256)
        self.layer15 = torch.nn.ReLU()
        self.layer16 = torch.nn.Linear(256,10)
        #self.layer17 = torch.nn.Softmax() ###


    def forward(self, x, hidden_queue,vi_svm_layer5_pid, hidden_1_mutex):
        out = self.layer1(x)

        out = self.layer2(out)	#


        out = self.layer3(out)
                
        out = self.layer4(out)   #

          
        out = self.layer5(out)   #
        out_3 = out
        out_3 = out_3.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_3)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer5_pid, signal.SIGUSR1)
        
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

class Visvm(Process):
    def __init__(self, hidden_queue, name, visvm_path, hidden_1_mutex):
        super(Process, self).__init__()
        self.hidden_queue = hidden_queue
        self.name = name
        self.visvm_path = visvm_path
        self.hidden_1_mutex = hidden_1_mutex
        
    def visvm_handler(self, signum, frame):
        # start_visvm = time.time()

        self.hidden_1_mutex.acquire()
        self.visvm_input = self.hidden_queue.get()
        self.hidden_1_mutex.release()	
        self.visvm_input = np.reshape(self.visvm_input, (1,-1))
        # print(self.name, 'shape : ', self.visvm_input.shape)


        self.pre_result = self.model.predict_gpu(self.visvm_input)



    def run(self):
        signal.signal(signal.SIGUSR1, self.visvm_handler) # 시그널 링크 
        self.model = OneClassSVM()
        self.model.load_from_file(self.visvm_path)
        # visvm WARM UP
        self.warm_input = np.zeros((1,1))
        self.model.predict_gpu(self.warm_input)
        # os.kill(os.getppid(), signal.SIGUSR1)
        print(self.name, ' VISVM START')
        signal.pause()	#	target model에서부터 signal 받기 전까지는 pause
        print(self.name, ' VISVM DONE')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    mutex = Lock()
    hidden_1_mutex = Lock()
    hidden_queue = Queue()




    vi_svm_path = ['/home/nvidia/joo/models/Carlini/VISVM/layer5_visvm.bin']



    print('derived model load done')

	# create vi svm process

    vi_svm_layer5_p = Visvm(hidden_queue, 'layer5', vi_svm_path[0], hidden_1_mutex)


	# visvm process start
    print('visvm process start')

    vi_svm_layer5_p.start()


	# get visvm process pid

    vi_svm_layer5_pid = vi_svm_layer5_p.pid

    # WARM UP MODEL LOAD
    warm_model = CarliniModel2()
    warm_model.load_state_dict(torch.load('./warmcarlini_cifar10.pt'))
    warm_model.eval()
    warm_model.to(device)

    model = CarliniModel()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/Carlini/carlini_cifar10_dict.pt'))
    model.eval()
    model.to(device)

    time.sleep(10)
    test_data = torch.rand(1,3,32,32).to(device)

    warm_model(test_data)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()

    model(test_data,hidden_queue, vi_svm_layer5_pid, hidden_1_mutex)
    end.record()
    torch.cuda.synchronize()
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)



    vi_svm_layer5_p.join()





    end_total.record()
    torch.cuda.synchronize()
    print("TOTAL TIME : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()

