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

    # def forward(self, x: torch.Tensor, hidden_queue, hidden_queue_2, derived_model_layer2_pid, vi_svm_layer3_pid, derived_model_layer3_pid, derived_model_layer5_pid, vi_svm_layer6_pid, derived_model_layer6_pid, vi_svm_layer8_pid, derived_model_layer8_pid, vi_svm_layer10_pid, derived_model_layer10_pid, vi_svm_layer12_pid, derived_model_layer12_pid, vi_svm_layer13_pid, derived_model_layer13_pid, vi_svm_layer14_pid, derived_model_layer14_pid, vi_svm_layer17_pid, derived_model_layer17_pid, vi_svm_layer20_pid, derived_model_layer20_pid, hidden_1_mutex, hidden_2_mutex) -> torch.Tensor:
    def forward(self, x: torch.Tensor, hidden_queue, vi_svm_layer3_pid, hidden_1_mutex) -> torch.Tensor:

        x = self.layer1(x)
        
        x = self.layer2(x)


        x = self.layer3(x)
        out_3 = x
        out_3 = out_3.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_3)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer3_pid, signal.SIGUSR1)
    

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
        # print(self.name, 'shape : ', self.visvm_input.shape)


        self.pre_result = self.model.predict_gpu(self.visvm_input)
        print(self.name, ' result : ', self.pre_result)



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


	

    # vi_svm_path = ['/home/nvidia/joo/models/AlexNet/VISVM/layer3_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer6_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer8_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer10_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer12_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer13_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer14_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer17_visvm.bin', '/home/nvidia/joo/models/AlexNet/VISVM/layer20_visvm.bin']
    # dm_path = ['/home/nvidia/joo/models/AlexNet/D_M/MP/layer2_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer3_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer5_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer6_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer8_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer10_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer12_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer13_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer14_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer17_derived_model_dict.pt', '/home/nvidia/joo/models/AlexNet/D_M/MP/layer20_derived_model_dict.pt']
    # pi_svm_path = ['/home/nvidia/joo/models/AlexNet/PISVM/layer2_layer3_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer3_layer5_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer5_layer6_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer6_layer8_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer8_layer10_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer10_layer12_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer12_layer13_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer13_layer14_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer14_layer17_pisvm.bin', '/home/nvidia/joo/models/AlexNet/PISVM/layer17_layer20_pisvm.bin']
    vi_svm_path = ['/home/nvidia/joo/models/AlexNet/VISVM/layer3_visvm.bin']




    derived_model_layer2 = derived_model.Derived_Model2()
    derived_model_layer3 = derived_model.Derived_Model3()
    derived_model_layer5 = derived_model.Derived_Model5()
    derived_model_layer6 = derived_model.Derived_Model6()

	
    vi_svm_layer3_p = Visvm(hidden_queue, 'layer3', vi_svm_path[0], hidden_1_mutex, 4096)



    
    print('visvm start')
    vi_svm_layer3_p.start()

	

    vi_svm_layer3_pid = vi_svm_layer3_p.pid


	
    print('warm target start')
	# WARM UP MODEL LOAD
    warm_model = AlexNet2()
    warm_model.load_state_dict(torch.load('/home/nvidia/joo/multi_process/AlexNet/warmalex2.pt'))
    warm_model.eval()
    warm_model.to(device)

	
    model = AlexNet()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/AlexNet/alexnet_dict_cifar10.pt'))
    model.eval()
    model.to(device)
    
    time.sleep(10)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)    

	# WARM UP
    warm_model(test_input)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    # result = model(test_input, hidden_queue, hidden_queue_2, derived_model_layer2_pid, vi_svm_layer3_pid, derived_model_layer3_pid, derived_model_layer5_pid, vi_svm_layer6_pid, derived_model_layer6_pid, vi_svm_layer8_pid, derived_model_layer8_pid, vi_svm_layer10_pid, derived_model_layer10_pid, vi_svm_layer12_pid, derived_model_layer12_pid, vi_svm_layer13_pid, derived_model_layer13_pid, vi_svm_layer14_pid, derived_model_layer14_pid, vi_svm_layer17_pid, derived_model_layer17_pid, vi_svm_layer20_pid, derived_model_layer20_pid, hidden_1_mutex, hidden_2_mutex)
    result = model(test_input, hidden_queue, vi_svm_layer3_pid, hidden_1_mutex)

    end.record()
    torch.cuda.synchronize()    
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)

	
    vi_svm_layer3_p.join()



	

    end_total.record()
    torch.cuda.synchronize()
    print("TOTAL TIME : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()




