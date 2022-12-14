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

    def forward(self, x: torch.Tensor, hidden_queue, hidden_queue_2, vi_svm_layer2_pid, derived_model_layer2_pid, vi_svm_layer3_pid, derived_model_layer3_pid, vi_svm_layer5_pid, derived_model_layer5_pid, vi_svm_layer6_pid, derived_model_layer6_pid, vi_svm_layer8_pid, derived_model_layer8_pid, vi_svm_layer10_pid, derived_model_layer10_pid, vi_svm_layer12_pid, derived_model_layer12_pid, vi_svm_layer13_pid, derived_model_layer13_pid, vi_svm_layer14_pid, derived_model_layer14_pid, vi_svm_layer17_pid, derived_model_layer17_pid, vi_svm_layer20_pid, derived_model_layer20_pid, hidden_1_mutex, hidden_2_mutex) -> torch.Tensor:
        x = self.layer1(x)
        
        x = self.layer2(x)
        out_1 = x
        out_1 = out_1.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_1)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer2_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_1)
        hidden_2_mutex.release()
        os.kill(derived_model_layer2_pid, signal.SIGUSR1)
        
        x = self.layer3(x)
        out_2 = x
        out_2 = out_2.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_2)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer3_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_2)
        hidden_2_mutex.release()
        os.kill(derived_model_layer3_pid, signal.SIGUSR1)
        
        x = self.layer4(x)
        x = self.layer5(x)
        out_3 = x
        out_3 = out_3.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_3)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer5_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_3)
        hidden_2_mutex.release()
        os.kill(derived_model_layer5_pid, signal.SIGUSR1)
        
        x = self.layer6(x)
        out_4 = x
        out_4 = out_4.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_4)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer6_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_4)
        hidden_2_mutex.release()
        os.kill(derived_model_layer6_pid, signal.SIGUSR1)
                
        x = self.layer7(x)
        x = self.layer8(x)
        out_5 = x
        out_5 = out_5.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_5)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer8_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_5)
        hidden_2_mutex.release()
        os.kill(derived_model_layer8_pid, signal.SIGUSR1)
        
        x = self.layer9(x)
        x = self.layer10(x)
        out_6 = x
        out_6 = out_6.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_6)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer10_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_6)
        hidden_2_mutex.release()
        os.kill(derived_model_layer10_pid, signal.SIGUSR1)
        
        x = self.layer11(x)
        x = self.layer12(x)
        out_7 = x
        out_7 = out_7.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_7)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer12_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_7)
        hidden_2_mutex.release()
        os.kill(derived_model_layer12_pid, signal.SIGUSR1)
        
        x = self.layer13(x)
        out_8 = x
        out_8 = out_8.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_8)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer13_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_8)
        hidden_2_mutex.release()
        os.kill(derived_model_layer13_pid, signal.SIGUSR1)
        
        x = self.layer14(x)
        out_9 = x
        out_9 = out_9.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_9)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer14_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_9)
        hidden_2_mutex.release()
        os.kill(derived_model_layer14_pid, signal.SIGUSR1)
        
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        out_10 = x
        out_10 = out_10.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_10)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer17_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_10)
        hidden_2_mutex.release()
        os.kill(derived_model_layer17_pid, signal.SIGUSR1)
        
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        out_11 = x
        out_11 = out_11.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_11)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer20_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_11)
        hidden_2_mutex.release()
        os.kill(derived_model_layer20_pid, signal.SIGUSR1)
        
        x = self.layer21(x)
        return x

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
        print(self.name, 'shape : ', self.visvm_input.shape)

        self.pre_result = self.model.predict_gpu(self.visvm_input)


    def run(self):
        signal.signal(signal.SIGUSR1, self.visvm_handler) # ????????? ?????? 
        self.model = OneClassSVM()
        self.model.load_from_file(self.visvm_path)
        # os.kill(os.getppid(), signal.SIGUSR1)
        print(self.name, ' VISVM START')
        signal.pause()	#	target model???????????? signal ?????? ???????????? pause
        print(self.name, ' VISVM DONE')

class DerivedModel(Process):		

    def __init__(self, name, index, checker, hidden_queue, derived_model, output_queue, mutex, pisvm_pid, hidden_2_mutex, dm_path):
        super(Process, self).__init__()
        self.index = index
        self.checker = checker
        self.name = name
        self.hidden_queue = hidden_queue
        self.derived_model = derived_model
        self.output_queue = output_queue
        self.mutex = mutex
        self.pisvm_pid = pisvm_pid
        self.hidden_2_mutex = hidden_2_mutex
        self.dm_path = dm_path


        
    def derived_model_handler(self, signum, frame):
        # start_dm = time.time()

        self.hidden_2_mutex.acquire()
        # print(self.name, ' DM QSIZE : ', self.hidden_queue.qsize())
        self.derived_model_input = self.hidden_queue.get()
        self.hidden_2_mutex.release()
        # os.kill(os.getppid(), signal.SIGUSR2)
        self.derived_model_input = np.reshape(self.derived_model_input, (1,-1))
        # print(self.name, ' SHAPE : ', self.derived_model_input.shape)
        self.derived_model_input = torch.Tensor(self.derived_model_input)
        self.derived_model_input = self.derived_model_input.to(device)
        
        self.derived_model_output = self.derived_model(self.derived_model_input)
        self.derived_model_output = self.derived_model_output.detach().cpu().numpy()
        # derived model output queue??? ?????? ??????????????? -> pisvm?????? get()??? ?????? ?????? ?????????
        """derived model?????? pisvm?????? ???????????? ??? ?????? ???????????? visvm ??????????????? ????????? ?????????"""
        if(self.index == 0 or self.index == 10):		#pisvm?????? ????????? ???????????? ????????? ????????? ??????
            self.output_queue.put(self.derived_model_output)
        else:
            self.output_queue.put(self.derived_model_output)
            self.output_queue.put(self.derived_model_output)

        
        self.mutex.acquire()
        self.checkbox = self.checker.get()
        self.checkbox[self.index] = 1
        self.checker.put(self.checkbox)
        self.mutex.release()
        
        if(self.index != 0):
            flag = True
            while(flag):
                self.checking = self.checker.get()
                self.checked = self.checking
                self.checker.put(self.checking)
                if(self.checked[self.index -1] != 0):
                    flag = False
            os.kill(self.pisvm_pid, signal.SIGUSR1)

        # end_dm = time.time()
        # print(self.name, f" DM time : {end_dm - start_dm:.3f} sec")


        # print(self.name, 'RESLUT : ', self.derived_model_output.shape)
            
    def run(self):
        signal.signal(signal.SIGUSR1, self.derived_model_handler)
        self.derived_model.load_state_dict(torch.load(self.dm_path, map_location ='cuda:0'))
        self.derived_model.eval()
        self.derived_model = self.derived_model.to(device)
        print(self.name, ' DERIVED process start')
        signal.pause()
        print(self.name, ' DERIVED process done')



class Pisvm(Process):
    """
    pisvm load, derived model check array, derived model output(shared memory ??????, ????????? ??????)->concatenate
    """
    def __init__(self, pisvm, checker, name, index, output_queue_1, output_queue_2, mutex):
        super(Process, self).__init__()
        self.pisvm = pisvm
        self.checker = checker
        self.name = name
        self.index = index
        self.mutex = mutex
        self.output_queue_1 = output_queue_1
        self.output_queue_2 = output_queue_2

    def pisvm_handler(self, signum, frame):
        # start_pisvm = time.time()
        # print('!!!!!here')
        self.pi_svm_input_1 = self.output_queue_1.get()
        self.pi_svm_input_2 = self.output_queue_2.get()
        self.pi_svm_input = np.concatenate((self.pi_svm_input_1, self.pi_svm_input_2), axis =1)
        print(self.name, ' pi svm input shape : ', self.pi_svm_input.shape)
        self.pisvm_result = self.model.predict_cpu(self.pi_svm_input)

        # end_pisvm = time.time()
        # print(self.name, f" PISVM time : {end_pisvm - start_pisvm:.3f} sec")



    def run(self):
        signal.signal(signal.SIGUSR1, self.pisvm_handler)
        self.model = OneClassSVM()
        self.model.load_from_file(self.pisvm)
        print(self.name, ' PISVM START')
        signal.pause()
        print(self.name, ' PISVM DONE')
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    mutex = Lock()
    hidden_1_mutex = Lock()
    hidden_2_mutex = Lock()

    hidden_queue = Queue()
    hidden_queue_2 = Queue()
    d_m_checker = np.array([0,0,0,0,0,0,0,0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)

    d_m_layer2_queue = Queue()
    d_m_layer3_queue = Queue()
    d_m_layer5_queue = Queue()
    d_m_layer6_queue = Queue()
    d_m_layer8_queue = Queue()
    d_m_layer10_queue = Queue()
    d_m_layer12_queue = Queue()
    d_m_layer13_queue = Queue()
    d_m_layer14_queue = Queue()
    d_m_layer17_queue = Queue()
    d_m_layer20_queue = Queue()
	
    vi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
    dm_path = ['/home/nvidia/joo/models/AlexNet/D_M/alex_dict_16384', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_4096', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_12288', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_3072', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_6144', '/home/nvidia/joo/models/AlexNet/D_M/alex_dict_1024']
    pi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
	
    derived_model_layer2 = derived_model.Derived_Model1()
    derived_model_layer3 = derived_model.Derived_Model2()
    derived_model_layer5 = derived_model.Derived_Model3()
    derived_model_layer6 = derived_model.Derived_Model4()
    derived_model_layer8 = derived_model.Derived_Model5()
    derived_model_layer10 = derived_model.Derived_Model2()
    derived_model_layer12 = derived_model.Derived_Model2()
    derived_model_layer13 = derived_model.Derived_Model6()
    derived_model_layer14 = derived_model.Derived_Model6()
    derived_model_layer17 = derived_model.Derived_Model2()
    derived_model_layer20 = derived_model.Derived_Model2()
	

    vi_svm_layer2_p = Visvm(hidden_queue, 'layer2', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer3_p = Visvm(hidden_queue, 'layer3', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer5_p = Visvm(hidden_queue, 'layer5', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer6_p = Visvm(hidden_queue, 'layer6', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer8_p = Visvm(hidden_queue, 'layer8', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer10_p = Visvm(hidden_queue, 'layer10', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer12_p = Visvm(hidden_queue, 'layer12', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer13_p = Visvm(hidden_queue, 'layer13', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer14_p = Visvm(hidden_queue, 'layer14', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer17_p = Visvm(hidden_queue, 'layer17', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer20_p = Visvm(hidden_queue, 'layer20', vi_svm_path[0], hidden_1_mutex)
	

    pi_svm_layer2_layer3_p = Pisvm(pi_svm_path[0], d_m_check_queue, '2&3', 0, d_m_layer2_queue, d_m_layer3_queue, mutex)
    pi_svm_layer3_layer5_p = Pisvm(pi_svm_path[0], d_m_check_queue, '3&5', 1, d_m_layer3_queue, d_m_layer5_queue, mutex)
    pi_svm_layer5_layer6_p = Pisvm(pi_svm_path[0], d_m_check_queue, '5&6', 2, d_m_layer5_queue, d_m_layer6_queue, mutex)
    pi_svm_layer6_layer8_p = Pisvm(pi_svm_path[0], d_m_check_queue, '6&8', 3, d_m_layer6_queue, d_m_layer8_queue, mutex)
    pi_svm_layer8_layer10_p = Pisvm(pi_svm_path[0], d_m_check_queue, '8&10', 4, d_m_layer8_queue, d_m_layer10_queue, mutex)
    pi_svm_layer10_layer12_p = Pisvm(pi_svm_path[0], d_m_check_queue, '10&12', 5, d_m_layer10_queue, d_m_layer12_queue, mutex)
    pi_svm_layer12_layer13_p = Pisvm(pi_svm_path[0], d_m_check_queue, '12&13', 6, d_m_layer12_queue, d_m_layer13_queue, mutex)
    pi_svm_layer13_layer14_p = Pisvm(pi_svm_path[0], d_m_check_queue, '13&14', 7, d_m_layer13_queue, d_m_layer14_queue, mutex)
    pi_svm_layer14_layer17_p = Pisvm(pi_svm_path[0], d_m_check_queue, '14&17', 8, d_m_layer14_queue, d_m_layer17_queue, mutex)
    pi_svm_layer17_layer20_p = Pisvm(pi_svm_path[0], d_m_check_queue, '17&20', 9, d_m_layer17_queue, d_m_layer20_queue, mutex)
    
    print('visvm start')
    vi_svm_layer2_p.start()
    vi_svm_layer3_p.start()
    vi_svm_layer5_p.start()
    vi_svm_layer6_p.start()
    vi_svm_layer8_p.start()
    vi_svm_layer10_p.start()
    vi_svm_layer12_p.start()
    vi_svm_layer13_p.start()
    vi_svm_layer14_p.start()
    vi_svm_layer17_p.start()
    vi_svm_layer20_p.start()
	
    print('pisvm start')
    pi_svm_layer2_layer3_p.start()
    pi_svm_layer3_layer5_p.start()
    pi_svm_layer5_layer6_p.start()
    pi_svm_layer6_layer8_p.start()
    pi_svm_layer8_layer10_p.start()
    pi_svm_layer10_layer12_p.start()
    pi_svm_layer12_layer13_p.start()
    pi_svm_layer13_layer14_p.start()
    pi_svm_layer14_layer17_p.start()
    pi_svm_layer17_layer20_p.start()

    vi_svm_layer2_pid = vi_svm_layer2_p.pid
    vi_svm_layer3_pid = vi_svm_layer3_p.pid
    vi_svm_layer5_pid = vi_svm_layer5_p.pid
    vi_svm_layer6_pid = vi_svm_layer6_p.pid
    vi_svm_layer8_pid = vi_svm_layer8_p.pid
    vi_svm_layer10_pid = vi_svm_layer10_p.pid
    vi_svm_layer12_pid = vi_svm_layer12_p.pid
    vi_svm_layer13_pid = vi_svm_layer13_p.pid
    vi_svm_layer14_pid = vi_svm_layer14_p.pid
    vi_svm_layer17_pid = vi_svm_layer17_p.pid
    vi_svm_layer20_pid = vi_svm_layer20_p.pid
	
    pi_svm_layer2_layer3_pid = pi_svm_layer2_layer3_p.pid 
    pi_svm_layer3_layer5_pid = pi_svm_layer3_layer5_p.pid
    pi_svm_layer5_layer6_pid = pi_svm_layer5_layer6_p.pid
    pi_svm_layer6_layer8_pid = pi_svm_layer6_layer8_p.pid
    pi_svm_layer8_layer10_pid = pi_svm_layer8_layer10_p.pid
    pi_svm_layer10_layer12_pid = pi_svm_layer10_layer12_p.pid
    pi_svm_layer12_layer13_pid = pi_svm_layer12_layer13_p.pid
    pi_svm_layer13_layer14_pid = pi_svm_layer13_layer14_p.pid
    pi_svm_layer14_layer17_pid = pi_svm_layer14_layer17_p.pid
    pi_svm_layer17_layer20_pid = pi_svm_layer17_layer20_p.pid
    
    derived_model_layer2_p = DerivedModel('DM layer2', 0, d_m_check_queue, hidden_queue_2, derived_model_layer2, d_m_layer2_queue, mutex, pi_svm_layer2_layer3_pid, hidden_2_mutex, dm_path[0])
    derived_model_layer3_p = DerivedModel('DM layer3', 1, d_m_check_queue, hidden_queue_2, derived_model_layer3, d_m_layer3_queue, mutex, pi_svm_layer2_layer3_pid, hidden_2_mutex, dm_path[1])
    derived_model_layer5_p = DerivedModel('DM layer5', 2, d_m_check_queue, hidden_queue_2, derived_model_layer5, d_m_layer5_queue, mutex, pi_svm_layer3_layer5_pid, hidden_2_mutex, dm_path[2])
    derived_model_layer6_p = DerivedModel('DM layer6', 3, d_m_check_queue, hidden_queue_2, derived_model_layer6, d_m_layer6_queue, mutex, pi_svm_layer5_layer6_pid, hidden_2_mutex, dm_path[3])
    derived_model_layer8_p = DerivedModel('DM layer8', 4, d_m_check_queue, hidden_queue_2, derived_model_layer8, d_m_layer8_queue, mutex, pi_svm_layer6_layer8_pid, hidden_2_mutex, dm_path[4])
    derived_model_layer10_p = DerivedModel('DM layer10', 5, d_m_check_queue, hidden_queue_2, derived_model_layer10, d_m_layer10_queue, mutex, pi_svm_layer8_layer10_pid, hidden_2_mutex, dm_path[1])
    derived_model_layer12_p = DerivedModel('DM layer12', 6, d_m_check_queue, hidden_queue_2, derived_model_layer12, d_m_layer12_queue, mutex, pi_svm_layer10_layer12_pid, hidden_2_mutex, dm_path[1])
    derived_model_layer13_p = DerivedModel('DM layer13', 7, d_m_check_queue, hidden_queue_2, derived_model_layer13, d_m_layer13_queue, mutex, pi_svm_layer12_layer13_pid, hidden_2_mutex, dm_path[5])
    derived_model_layer14_p = DerivedModel('DM layer14', 8, d_m_check_queue, hidden_queue_2, derived_model_layer14, d_m_layer14_queue, mutex, pi_svm_layer13_layer14_pid, hidden_2_mutex, dm_path[5])
    derived_model_layer17_p = DerivedModel('DM layer17', 9, d_m_check_queue, hidden_queue_2, derived_model_layer17, d_m_layer17_queue, mutex, pi_svm_layer14_layer17_pid, hidden_2_mutex, dm_path[1])
    derived_model_layer20_p = DerivedModel('DM layer20', 10, d_m_check_queue, hidden_queue_2, derived_model_layer20, d_m_layer20_queue, mutex, pi_svm_layer17_layer20_pid, hidden_2_mutex, dm_path[1])
	
    derived_model_layer2_p.start()
    derived_model_layer3_p.start()
    derived_model_layer5_p.start()
    derived_model_layer6_p.start()
    derived_model_layer8_p.start()
    derived_model_layer10_p.start()
    derived_model_layer12_p.start()
    derived_model_layer13_p.start()
    derived_model_layer14_p.start()
    derived_model_layer17_p.start()
    derived_model_layer20_p.start()
	
    derived_model_layer2_pid = derived_model_layer2_p.pid
    derived_model_layer3_pid = derived_model_layer3_p.pid
    derived_model_layer5_pid = derived_model_layer5_p.pid
    derived_model_layer6_pid = derived_model_layer6_p.pid
    derived_model_layer8_pid = derived_model_layer8_p.pid
    derived_model_layer10_pid = derived_model_layer10_p.pid
    derived_model_layer12_pid = derived_model_layer12_p.pid
    derived_model_layer13_pid = derived_model_layer13_p.pid
    derived_model_layer14_pid = derived_model_layer14_p.pid
    derived_model_layer17_pid = derived_model_layer17_p.pid
    derived_model_layer20_pid = derived_model_layer20_p.pid
	

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


    warm_model(test_input)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, hidden_queue, hidden_queue_2, vi_svm_layer2_pid, derived_model_layer2_pid, vi_svm_layer3_pid, derived_model_layer3_pid, vi_svm_layer5_pid, derived_model_layer5_pid, vi_svm_layer6_pid, derived_model_layer6_pid, vi_svm_layer8_pid, derived_model_layer8_pid, vi_svm_layer10_pid, derived_model_layer10_pid, vi_svm_layer12_pid, derived_model_layer12_pid, vi_svm_layer13_pid, derived_model_layer13_pid, vi_svm_layer14_pid, derived_model_layer14_pid, vi_svm_layer17_pid, derived_model_layer17_pid, vi_svm_layer20_pid, derived_model_layer20_pid, hidden_1_mutex, hidden_2_mutex)
    end.record()
    torch.cuda.synchronize()    
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)

	
    vi_svm_layer2_p.join()
    vi_svm_layer3_p.join()
    vi_svm_layer5_p.join()
    vi_svm_layer6_p.join()
    vi_svm_layer8_p.join()
    vi_svm_layer10_p.join()
    vi_svm_layer12_p.join()
    vi_svm_layer13_p.join()
    vi_svm_layer14_p.join()
    vi_svm_layer17_p.join()
    vi_svm_layer20_p.join()
	
    pi_svm_layer2_layer3_p.join()
    pi_svm_layer3_layer5_p.join()
    pi_svm_layer5_layer6_p.join()
    pi_svm_layer6_layer8_p.join()
    pi_svm_layer8_layer10_p.join()
    pi_svm_layer10_layer12_p.join()
    pi_svm_layer12_layer13_p.join()
    pi_svm_layer13_layer14_p.join()
    pi_svm_layer14_layer17_p.join()
    pi_svm_layer17_layer20_p.join()
	
    derived_model_layer2_p.join()
    derived_model_layer3_p.join()
    derived_model_layer5_p.join()
    derived_model_layer6_p.join()
    derived_model_layer8_p.join()
    derived_model_layer10_p.join()
    derived_model_layer12_p.join()
    derived_model_layer13_p.join()
    derived_model_layer14_p.join()
    derived_model_layer17_p.join()
    derived_model_layer20_p.join()
	

    end_total.record()
    torch.cuda.synchronize()
    print("TOTAL TIME : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()
    hidden_queue_2.close()
    d_m_check_queue.close()

    d_m_layer2_queue.close()
    d_m_layer3_queue.close()
    d_m_layer5_queue.close()
    d_m_layer6_queue.close()
    d_m_layer8_queue.close()
    d_m_layer10_queue.close()
    d_m_layer12_queue.close()
    d_m_layer13_queue.close()
    d_m_layer14_queue.close()
    d_m_layer17_queue.close()
    d_m_layer20_queue.close()