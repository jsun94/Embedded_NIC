# from multiprocessing import Process, Value, Array
# import numpy as np

# """Array [:]로 복사되는 것 까지 확인 완료"""


# class FF(Process):
#     def __init__(self, n, a):
#         super(Process, self).__init__()
#         self.n = n
#         self.a = a

#     def run(self):
#         self.n.value = 3.1415927
#         # for i in range(len(self.a)):
#         #     self.a[i] = -self.a[i]
#         ss = np.array([1,2,3,1,2,3,1,2,3,1,2,3])
#         self.a[:] = ss[:]

# if __name__ == '__main__':
#     num = Value('d', 0.0)
#     arr = Array('i', 12)
#     print('arr : ', arr[:])

#     # p = Process(target=f, args=(num, arr))
#     p = FF(num,arr)
#     p.start()
#     p.join()

#     print(num.value)
#     print(arr[:])

#####################################################################
"""shared memory array로 visvm까지만 한 것"""
import torch
import torch.nn as nn
from typing import Any
import torchvision.models as models


from multiprocessing import Process, Queue, Lock, Array
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

    def forward(self, x: torch.Tensor, layer5_array, vi_svm_layer5_pid, layer6_array, vi_svm_layer6_pid, layer8_array, vi_svm_layer8_pid, layer10_array, vi_svm_layer10_pid, layer12_array, vi_svm_layer12_pid, layer13_array, vi_svm_layer13_pid, layer14_array, vi_svm_layer14_pid, layer17_array, vi_svm_layer17_pid, layer20_array, vi_svm_layer20_pid) -> torch.Tensor:
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        out_3 = x
        out_3 = out_3.detach().cpu().numpy()
        out_3 = np.reshape(out_3,(12288,))
        layer5_array[:] = out_3[:]
        os.kill(vi_svm_layer5_pid, signal.SIGUSR1)
        
        x = self.layer6(x)
        out_4 = x
        out_4 = out_4.detach().cpu().numpy()
        out_4 = np.reshape(out_4,(3072,))
        layer6_array[:] = out_4[:]
        os.kill(vi_svm_layer6_pid, signal.SIGUSR1)

        x = self.layer7(x)
        x = self.layer8(x)
        out_5 = x
        out_5 = out_5.detach().cpu().numpy()
        out_5 = np.reshape(out_5,(6144,))
        layer8_array[:] = out_5[:]
        os.kill(vi_svm_layer8_pid, signal.SIGUSR1)

        x = self.layer9(x)
        x = self.layer10(x)
        out_6 = x
        out_6 = out_6.detach().cpu().numpy()
        out_6 = np.reshape(out_6,(4096,))
        layer10_array[:] = out_6[:]
        os.kill(vi_svm_layer10_pid, signal.SIGUSR1)

        x = self.layer11(x)
        x = self.layer12(x)
        out_7 = x
        out_7 = out_7.detach().cpu().numpy()
        out_7 = np.reshape(out_7,(4096,))
        layer12_array[:] = out_7[:]
        os.kill(vi_svm_layer12_pid, signal.SIGUSR1)

        x = self.layer13(x)
        out_8 = x
        out_8 = out_8.detach().cpu().numpy()
        out_8 = np.reshape(out_8,(1024,))
        layer13_array[:] = out_8[:]
        os.kill(vi_svm_layer13_pid, signal.SIGUSR1)

        x = self.layer14(x)
        out_9 = x
        out_9 = out_9.detach().cpu().numpy()
        out_9 = np.reshape(out_9,(1024,))
        layer14_array[:] = out_9[:]
        os.kill(vi_svm_layer14_pid, signal.SIGUSR1)

        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        out_10 = x
        out_10 = out_10.detach().cpu().numpy()
        out_10 = np.reshape(out_10,(4096,))
        layer17_array[:] = out_10[:]
        os.kill(vi_svm_layer17_pid, signal.SIGUSR1)

        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        out_11 = x
        out_11 = out_11.detach().cpu().numpy()
        out_11 = np.reshape(out_11,(4096,))
        layer20_array[:] = out_11[:]
        os.kill(vi_svm_layer20_pid, signal.SIGUSR1)

        x = self.layer21(x)
        return x

class Visvm(Process):
    def __init__(self, hidden_array, name, visvm_path, input_shape):
        super(Process, self).__init__()
        self.hidden_array = hidden_array
        self.name = name
        self.visvm_path = visvm_path
        self.input_shape = input_shape
        
    def visvm_handler(self, signum, frame):
        # start_visvm = time.time()

        self.visvm_input = np.zeros((self.input_shape,))
        self.visvm_input[:] = self.hidden_array[:]

        self.visvm_input = np.reshape(self.visvm_input, (1,-1))
        print(self.name, 'shape : ', self.visvm_input.shape)


        self.pre_result = self.model.predict_gpu(self.visvm_input)



    def run(self):
        signal.signal(signal.SIGUSR1, self.visvm_handler) # 시그널 링크 
        self.model = OneClassSVM()
        self.model.load_from_file(self.visvm_path)
        # os.kill(os.getppid(), signal.SIGUSR1)
        print(self.name, ' VISVM START')
        signal.pause()	#	target model에서부터 signal 받기 전까지는 pause
        print(self.name, ' VISVM DONE')


    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')


    layer5_array = Array('i', 12288)
    layer6_array = Array('i', 3072)
    layer8_array = Array('i', 6144)
    layer10_array = Array('i', 4096)
    layer12_array = Array('i', 4096)
    layer13_array = Array('i', 1024)
    layer14_array = Array('i', 1024)
    layer17_array = Array('i', 4096)
    layer20_array = Array('i', 4096)

	
    vi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']

    vi_svm_layer5_p = Visvm(layer5_array, 'layer5', vi_svm_path[0], 12288)
    vi_svm_layer6_p = Visvm(layer6_array, 'layer6', vi_svm_path[0], 3072)
    vi_svm_layer8_p = Visvm(layer8_array, 'layer8', vi_svm_path[0], 6144)
    vi_svm_layer10_p = Visvm(layer10_array, 'layer10', vi_svm_path[0], 4096)
    vi_svm_layer12_p = Visvm(layer12_array, 'layer12', vi_svm_path[0], 4096)
    vi_svm_layer13_p = Visvm(layer13_array, 'layer13', vi_svm_path[0], 1024)
    vi_svm_layer14_p = Visvm(layer14_array, 'layer14', vi_svm_path[0], 1024)
    vi_svm_layer17_p = Visvm(layer17_array, 'layer17', vi_svm_path[0], 4096)
    vi_svm_layer20_p = Visvm(layer20_array, 'layer20', vi_svm_path[0], 4096)


	
    
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
    
    time.sleep(3)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)    

	# WARM UP
    warm_model(test_input)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, layer5_array, vi_svm_layer5_pid, layer6_array, vi_svm_layer6_pid, layer8_array, vi_svm_layer8_pid, layer10_array, vi_svm_layer10_pid, layer12_array, vi_svm_layer12_pid, layer13_array, vi_svm_layer13_pid, layer14_array, vi_svm_layer14_pid, layer17_array, vi_svm_layer17_pid, layer20_array, vi_svm_layer20_pid)
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




