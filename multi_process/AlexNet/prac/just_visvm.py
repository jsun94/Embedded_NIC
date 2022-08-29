import torch
import torch.nn as nn
from typing import Any
import torchvision.models as models


from multiprocessing import Process, Queue, Lock
import multiprocessing
import numpy as np
import time
import os
from thundersvm import OneClassSVM
import signal
import derived_model

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

WARM = 3

""" WARM UP MODEL """
class AlexNet2(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet2, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = nn.MaxPool2d(kernel_size=2)
        self.layer4 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.layer5 = nn.ReLU(inplace=True)
        self.layer6 = nn.MaxPool2d(kernel_size=2)
        self.layer7 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.layer8 = nn.ReLU(inplace=True)
        self.layer9 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.layer10 = nn.ReLU(inplace=True)
        self.layer11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer12 = nn.ReLU(inplace=True)
        self.layer13 = nn.MaxPool2d(kernel_size=2)
        self.layer14 = nn.Flatten()
        self.layer15 = nn.Dropout()
        self.layer16 = nn.Linear(256 * 2 * 2, 4096)
        self.layer17 = nn.ReLU(inplace=True)
        self.layer18 = nn.Dropout()
        self.layer19 = nn.Linear(4096, 4096)
        self.layer20 = nn.ReLU(inplace=True)
        self.layer21 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)      
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        return x

""" WARM UP MODEL """

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 10) -> None:
        super(AlexNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = nn.MaxPool2d(kernel_size=2)
        self.layer4 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.layer5 = nn.ReLU(inplace=True)
        self.layer6 = nn.MaxPool2d(kernel_size=2)
        self.layer7 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.layer8 = nn.ReLU(inplace=True)
        self.layer9 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.layer10 = nn.ReLU(inplace=True)
        self.layer11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.layer12 = nn.ReLU(inplace=True)
        self.layer13 = nn.MaxPool2d(kernel_size=2)
        self.layer14 = nn.Flatten()
        self.layer15 = nn.Dropout()
        self.layer16 = nn.Linear(256 * 2 * 2, 4096)
        self.layer17 = nn.ReLU(inplace=True)
        self.layer18 = nn.Dropout()
        self.layer19 = nn.Linear(4096, 4096)
        self.layer20 = nn.ReLU(inplace=True)
        self.layer21 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor, hidden_queue, vi_svm_layer5_pid, vi_svm_layer6_pid, vi_svm_layer8_pid, vi_svm_layer10_pid, vi_svm_layer12_pid, vi_svm_layer13_pid, vi_svm_layer14_pid, vi_svm_layer17_pid, vi_svm_layer20_pid, hidden_1_mutex) -> torch.Tensor:
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        out_3 = x
        out_3 = out_3.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_3)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer5_pid, signal.SIGUSR1)
        
        x = self.layer6(x)
        out_4 = x
        out_4 = out_4.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_4)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer6_pid, signal.SIGUSR1)
                
        x = self.layer7(x)
        x = self.layer8(x)
        out_5 = x
        out_5 = out_5.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_5)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer8_pid, signal.SIGUSR1)
        
        x = self.layer9(x)
        x = self.layer10(x)
        out_6 = x
        out_6 = out_6.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_6)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer10_pid, signal.SIGUSR1)
        
        x = self.layer11(x)
        x = self.layer12(x)
        out_7 = x
        out_7 = out_7.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_7)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer12_pid, signal.SIGUSR1)
        
        x = self.layer13(x)
        out_8 = x
        out_8 = out_8.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_8)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer13_pid, signal.SIGUSR1)
        
        x = self.layer14(x)
        out_9 = x
        out_9 = out_9.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_9)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer14_pid, signal.SIGUSR1)
        
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        out_10 = x
        out_10 = out_10.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_10)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer17_pid, signal.SIGUSR1)
        
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        out_11 = x
        out_11 = out_11.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_11)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer20_pid, signal.SIGUSR1)
        
        x = self.layer21(x)
        return x

class Visvm(Process):
    def __init__(self, hidden_queue, name, visvm_path, hidden_1_mutex, input_shape):
        super(Process, self).__init__()
        self.hidden_queue = hidden_queue
        self.name = name
        self.visvm_path = visvm_path
        self.hidden_1_mutex = hidden_1_mutex
        self.input_shape = input_shape
        
    def visvm_handler(self, signum, frame):
        # start_visvm = time.time()

        self.hidden_1_mutex.acquire()
        self.visvm_input = self.hidden_queue.get()
        self.hidden_1_mutex.release()	
        self.visvm_input = np.reshape(self.visvm_input, (1,-1))
        print(self.name, 'shape : ', self.visvm_input.shape)


        self.pre_result = self.model.predict_gpu(self.visvm_input)



    def run(self):
        signal.signal(signal.SIGUSR1, self.visvm_handler) # 시그널 링크 
        self.model = OneClassSVM()
        self.model.load_from_file(self.visvm_path)
        self.warm_input = np.zeros((1,self.input_shape))
        self.model.predict_gpu(self.warm_input)
        # os.kill(os.getppid(), signal.SIGUSR1)
        print(self.name, ' VISVM START')
        signal.pause()	#	target model에서부터 signal 받기 전까지는 pause
        print(self.name, ' VISVM DONE')


    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    mutex = Lock()
    hidden_1_mutex = Lock()
    hidden_2_mutex = Lock()

    hidden_queue = Queue()

	
    vi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
    dm_path = ['/home/nvidia/joo/models/AlexNet/D_M/alex_dict_16384', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_4096', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_12288', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_3072', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_6144', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_1024']
    pi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
	


	


    vi_svm_layer5_p = Visvm(hidden_queue, 'layer5', vi_svm_path[0], hidden_1_mutex, 12288)
    vi_svm_layer6_p = Visvm(hidden_queue, 'layer6', vi_svm_path[0], hidden_1_mutex, 3072)
    vi_svm_layer8_p = Visvm(hidden_queue, 'layer8', vi_svm_path[0], hidden_1_mutex, 6144)
    vi_svm_layer10_p = Visvm(hidden_queue, 'layer10', vi_svm_path[0], hidden_1_mutex, 4096)
    vi_svm_layer12_p = Visvm(hidden_queue, 'layer12', vi_svm_path[0], hidden_1_mutex, 4096)
    vi_svm_layer13_p = Visvm(hidden_queue, 'layer13', vi_svm_path[0], hidden_1_mutex, 1024)
    vi_svm_layer14_p = Visvm(hidden_queue, 'layer14', vi_svm_path[0], hidden_1_mutex, 1024)
    vi_svm_layer17_p = Visvm(hidden_queue, 'layer17', vi_svm_path[0], hidden_1_mutex, 4096)
    vi_svm_layer20_p = Visvm(hidden_queue, 'layer20', vi_svm_path[0], hidden_1_mutex, 4096)
	
    
    print('visvm start')

    vi_svm_layer5_p.start()
    vi_svm_layer6_p.start()
    vi_svm_layer8_p.start()
    vi_svm_layer10_p.start()
    vi_svm_layer12_p.start()
    vi_svm_layer13_p.start()
    vi_svm_layer14_p.start()
    vi_svm_layer17_p.start()
    vi_svm_layer20_p.start()
	


    vi_svm_layer5_pid = vi_svm_layer5_p.pid
    vi_svm_layer6_pid = vi_svm_layer6_p.pid
    vi_svm_layer8_pid = vi_svm_layer8_p.pid
    vi_svm_layer10_pid = vi_svm_layer10_p.pid
    vi_svm_layer12_pid = vi_svm_layer12_p.pid
    vi_svm_layer13_pid = vi_svm_layer13_p.pid
    vi_svm_layer14_pid = vi_svm_layer14_p.pid
    vi_svm_layer17_pid = vi_svm_layer17_p.pid
    vi_svm_layer20_pid = vi_svm_layer20_p.pid
	
    
	
	# WARM UP MODEL LOAD
    warm_model = AlexNet2()
    warm_model.load_state_dict(torch.load('/home/nvidia/joo/multi_process/AlexNet/warmalex2.pt'))
    warm_model.eval()
    warm_model.to(device)

	
    model = AlexNet()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/AlexNet/alexnet_dict_cifar10.pt'))
    model.eval()
    model.to(device)
    
    time.sleep(20)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)    

	# WARM UP
    warm_model(test_input)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, hidden_queue, vi_svm_layer5_pid, vi_svm_layer6_pid, vi_svm_layer8_pid, vi_svm_layer10_pid, vi_svm_layer12_pid, vi_svm_layer13_pid, vi_svm_layer14_pid, vi_svm_layer17_pid, vi_svm_layer20_pid, hidden_1_mutex)
    end.record()
    torch.cuda.synchronize()    
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)

	

    vi_svm_layer5_p.join()
    vi_svm_layer6_p.join()
    vi_svm_layer8_p.join()
    vi_svm_layer10_p.join()
    vi_svm_layer12_p.join()
    vi_svm_layer13_p.join()
    vi_svm_layer14_p.join()
    vi_svm_layer17_p.join()
    vi_svm_layer20_p.join()
	
	

    end_total.record()
    torch.cuda.synchronize()
    print("TOTAL TIME : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()



