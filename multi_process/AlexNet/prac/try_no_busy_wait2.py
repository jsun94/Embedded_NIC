# """7.285 sec"""

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

    def forward(self, x: torch.Tensor, hidden_queue_2, derived_model_layer5_pid, derived_model_layer6_pid, derived_model_layer8_pid, derived_model_layer10_pid, derived_model_layer12_pid, derived_model_layer13_pid, derived_model_layer14_pid, derived_model_layer17_pid, derived_model_layer20_pid, hidden_2_mutex) -> torch.Tensor:
        x = self.layer1(x)
        
        x = self.layer2(x)
        
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        out_3 = x
        out_3 = out_3.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_3)
        hidden_2_mutex.release()
        os.kill(derived_model_layer5_pid, signal.SIGUSR1)
        
        x = self.layer6(x)
        out_4 = x
        out_4 = out_4.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_4)
        hidden_2_mutex.release()
        os.kill(derived_model_layer6_pid, signal.SIGUSR1)
                
        x = self.layer7(x)
        x = self.layer8(x)
        out_5 = x
        out_5 = out_5.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_5)
        hidden_2_mutex.release()
        os.kill(derived_model_layer8_pid, signal.SIGUSR1)
        
        x = self.layer9(x)
        x = self.layer10(x)
        out_6 = x
        out_6 = out_6.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_6)
        hidden_2_mutex.release()
        os.kill(derived_model_layer10_pid, signal.SIGUSR1)
        
        x = self.layer11(x)
        x = self.layer12(x)
        out_7 = x
        out_7 = out_7.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_7)
        hidden_2_mutex.release()
        os.kill(derived_model_layer12_pid, signal.SIGUSR1)
        
        x = self.layer13(x)
        out_8 = x
        out_8 = out_8.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_8)
        hidden_2_mutex.release()
        os.kill(derived_model_layer13_pid, signal.SIGUSR1)
        
        x = self.layer14(x)
        out_9 = x
        out_9 = out_9.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_9)
        hidden_2_mutex.release()
        os.kill(derived_model_layer14_pid, signal.SIGUSR1)
        
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        out_10 = x
        out_10 = out_10.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_10)
        hidden_2_mutex.release()
        os.kill(derived_model_layer17_pid, signal.SIGUSR1)
        
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        out_11 = x
        out_11 = out_11.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_11)
        hidden_2_mutex.release()
        os.kill(derived_model_layer20_pid, signal.SIGUSR1)
        
        x = self.layer21(x)
        return x


class DerivedModel(Process):		

    def __init__(self, name, index, pid_queue, checker, hidden_queue, derived_model, before_output_queue, output_queue, mutex, hidden_2_mutex, dm_path, pisvm_path, dummy_shape):
        super(Process, self).__init__()
        self.index = index
        self.checker = checker
        self.name = name
        self.hidden_queue = hidden_queue
        self.derived_model = derived_model
        self.output_queue = output_queue
        self.mutex = mutex
        self.hidden_2_mutex = hidden_2_mutex
        self.dm_path = dm_path
        self.dummy_shape = dummy_shape
        self.before_output_queue = before_output_queue
        self.pisvm_path = pisvm_path
        self.pid_queue = pid_queue


        
    def derived_model_handler(self, signum, frame):

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

        # derived model output queue에 두번 삽입해야함 -> pisvm에서 get()을 두번 해야 하므로
        """derived model에서 pisvm으로 시그널은 잘 가는 상태이고 visvm 인퍼런스가 안되는 상태임"""

        self.output_queue.put(self.derived_model_output)
        print(self.name, ' here')
        if(self.index != 8 and self.index != 0):
            os.kill(self.before_pid, signal.SIGUSR2)
            os.kill(self.current_pid, signal.SIGUSR1)
        elif(self.index == 0):
            os.kill(self.current_pid, signal.SIGUSR1)
        else:
            os.kill(self.current_pid,signal.SIGUSR2)
            
            
    def run(self):
        signal.signal(signal.SIGUSR1, self.derived_model_handler)

		# 다음 derived model pid를 가져오기 위한 code
        # self.pid_checking = self.pid_queue.get()
        # self.pid_checking[self.index] = os.getpid()
        # self.pid_queue.put(self.pid_checking)
        if(self.index != 8 and self.index != 0):
            pid_flag = True
            while(pid_flag):
                self.pid_checking = self.pid_queue.get()
                self.pid_checked = self.pid_checking
                self.pid_queue.put(self.pid_checking)
                if(self.pid_checked[self.index - 1] != 0 and self.pid_checked[self.index] != 0):
                    pid_flag = False
            self.before_pid = self.pid_checked[self.index -1]		# self.next_pid = 다음 derived model pid
            self.current_pid = self.pid_checked[self.index]
            print(self.name, self.before_pid, self.current_pid)
        elif(self.index == 0):
            pid_flag = True
            while(pid_flag):
                self.pid_checking = self.pid_queue.get()
                self.pid_checked = self.pid_checking
                self.pid_queue.put(self.pid_checking)
                if(self.pid_checked[self.index] != 0):
                    pid_flag = False
                    self.current_pid = self.pid_checked[self.index]
        else:
            pid_flag = True
            while(pid_flag):
                self.pid_checking = self.pid_queue.get()
                self.pid_checked = self.pid_checking
                self.pid_queue.put(self.pid_checking)
                if(self.pid_checked[self.index -1] != 0):
                    pid_flag = False
                    self.current_pid = self.pid_checked[self.index-1]

        self.derived_model.load_state_dict(torch.load(self.dm_path, map_location ='cuda:0'))
        self.derived_model.eval()
        self.derived_model = self.derived_model.to(device)
        #derived model WARM UP
        dummy_input = torch.ones(1,self.dummy_shape).to(device)
        self.derived_model(dummy_input)
        self.pi_svm_model = OneClassSVM()
        self.pi_svm_model.load_from_file(self.pisvm_path)

        print(self.name, ' DERIVED process start')
        signal.pause()
        print(self.name, ' DERIVED process done')


class Pisvm(Process):
    def __init__(self, name, index, pid_queue, pisvm_path):
        super(Process, self).__init__()
        self.name = name
        self.index =index
        self.pid_queue = pid_queue
        self.pisvm_path = pisvm_path


    def pisvm_handler(self, signum, frame):
        # print(self.name, ' HI!')
        time.sleep(1)

    def pisvm_handler2(self, signum, frame):
        print(self.index, ' BYE!')

    def run(self):
        signal.signal(signal.SIGUSR1, self.pisvm_handler2)
        signal.signal(signal.SIGUSR2, self.pisvm_handler)
        # 자기 pid를 pid_queue에 넣음
        self.pid_checking = self.pid_queue.get()
        self.pid_checking[self.index] = os.getpid()
        self.pid_queue.put(self.pid_checking)
        print(self.name, ' PISVM START')
        signal.pause()
        signal.pause()
        print(self.name, ' PISVM DONE')       

    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    mutex = Lock()
    hidden_2_mutex = Lock()

    hidden_queue_2 = Queue()
    d_m_checker = np.array([0,0,0,0,0,0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)
    
    pid_queue = Queue()
    pid_checker = np.array([0,0,0,0,0,0,0,0])
    pid_queue.put(pid_checker)


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
	

    derived_model_layer5 = derived_model.Derived_Model3()
    derived_model_layer6 = derived_model.Derived_Model4()
    derived_model_layer8 = derived_model.Derived_Model5()
    derived_model_layer10 = derived_model.Derived_Model2()
    derived_model_layer12 = derived_model.Derived_Model2()
    derived_model_layer13 = derived_model.Derived_Model6()
    derived_model_layer14 = derived_model.Derived_Model6()
    derived_model_layer17 = derived_model.Derived_Model2()
    derived_model_layer20 = derived_model.Derived_Model2()
	
    pisvm_layer5_layer6_p = Pisvm('5/6', 0, pid_queue, pi_svm_path[0])
    pisvm_layer6_layer8_p = Pisvm('6/8', 1, pid_queue, pi_svm_path[0])	
    pisvm_layer8_layer10_p = Pisvm('8/10', 2, pid_queue, pi_svm_path[0])	
    pisvm_layer10_layer12_p = Pisvm('10/12', 3, pid_queue, pi_svm_path[0])	
    pisvm_layer12_layer13_p = Pisvm('12/13', 4, pid_queue, pi_svm_path[0])	
    pisvm_layer13_layer14_p = Pisvm('13/14', 5, pid_queue, pi_svm_path[0])	
    pisvm_layer14_layer17_p = Pisvm('14/17', 6, pid_queue, pi_svm_path[0])	
    pisvm_layer17_layer20_p = Pisvm('17/20', 7, pid_queue, pi_svm_path[0])	

    pisvm_layer5_layer6_p.start()
    pisvm_layer6_layer8_p.start()
    pisvm_layer8_layer10_p.start()
    pisvm_layer10_layer12_p.start()
    pisvm_layer12_layer13_p.start()
    pisvm_layer13_layer14_p.start()
    pisvm_layer14_layer17_p.start()
    pisvm_layer17_layer20_p.start()


    pisvm_layer5_layer6_pid = pisvm_layer5_layer6_p.pid
    pisvm_layer6_layer8_pid = pisvm_layer6_layer8_p.pid
    pisvm_layer8_layer10_pid = pisvm_layer8_layer10_p.pid
    pisvm_layer10_layer12_pid = pisvm_layer10_layer12_p.pid	
    pisvm_layer12_layer13_pid = pisvm_layer12_layer13_p.pid	
    pisvm_layer13_layer14_pid = pisvm_layer13_layer14_p.pid
    pisvm_layer14_layer17_pid = pisvm_layer14_layer17_p.pid
    pisvm_layer17_layer20_pid = pisvm_layer17_layer20_p.pid

    
    

    derived_model_layer5_p = DerivedModel('DM layer5', 0, pid_queue,d_m_check_queue, hidden_queue_2, derived_model_layer5, d_m_layer5_queue, d_m_layer5_queue, mutex, hidden_2_mutex, dm_path[2], pi_svm_path[0], 12288)
    derived_model_layer6_p = DerivedModel('DM layer6', 1, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer6, d_m_layer5_queue, d_m_layer6_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 3072)
    derived_model_layer8_p = DerivedModel('DM layer8', 2, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer8, d_m_layer6_queue, d_m_layer8_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 6144)
    derived_model_layer10_p = DerivedModel('DM layer10', 3, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer10, d_m_layer8_queue, d_m_layer10_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 4096)
    derived_model_layer12_p = DerivedModel('DM layer12', 4, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer12, d_m_layer10_queue, d_m_layer12_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 4096)
    derived_model_layer13_p = DerivedModel('DM layer13', 5, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer13, d_m_layer12_queue, d_m_layer13_queue, mutex, hidden_2_mutex, dm_path[5], pi_svm_path[0], 1024)
    derived_model_layer14_p = DerivedModel('DM layer14', 6, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer14, d_m_layer13_queue, d_m_layer14_queue, mutex, hidden_2_mutex, dm_path[5], pi_svm_path[0], 1024)
    derived_model_layer17_p = DerivedModel('DM layer17', 7, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer17, d_m_layer14_queue, d_m_layer17_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 4096)
    derived_model_layer20_p = DerivedModel('DM layer20', 8, pid_queue, d_m_check_queue, hidden_queue_2, derived_model_layer20, d_m_layer17_queue, d_m_layer20_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 4096)
	

    derived_model_layer5_p.start()
    derived_model_layer6_p.start()
    derived_model_layer8_p.start()
    derived_model_layer10_p.start()
    derived_model_layer12_p.start()
    derived_model_layer13_p.start()
    derived_model_layer14_p.start()
    derived_model_layer17_p.start()
    derived_model_layer20_p.start()
	

    derived_model_layer5_pid = derived_model_layer5_p.pid
    derived_model_layer6_pid = derived_model_layer6_p.pid
    derived_model_layer8_pid = derived_model_layer8_p.pid
    derived_model_layer10_pid = derived_model_layer10_p.pid
    derived_model_layer12_pid = derived_model_layer12_p.pid
    derived_model_layer13_pid = derived_model_layer13_p.pid
    derived_model_layer14_pid = derived_model_layer14_p.pid
    derived_model_layer17_pid = derived_model_layer17_p.pid
    derived_model_layer20_pid = derived_model_layer20_p.pid

    print('layer5_pid : ', derived_model_layer5_pid, ' layer6_pid : ', derived_model_layer6_pid, ' layer8_pid : ', derived_model_layer8_pid, ' layer10_pid : ', derived_model_layer10_pid, ' layer12_pid : ', derived_model_layer12_pid, ' layer13_pid : ', derived_model_layer13_pid, ' layer14_pid : ', derived_model_layer14_pid, ' layer17_pid : ', derived_model_layer17_pid, ' layer20_pid : ', derived_model_layer20_pid)
	
	# WARM UP MODEL LOAD
    warm_model = AlexNet2()
    warm_model.load_state_dict(torch.load('/home/nvidia/joo/multi_process/AlexNet/warmalex2.pt'))
    warm_model.eval()
    warm_model.to(device)

	
    model = AlexNet()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/AlexNet/alexnet_dict_cifar10.pt'))
    model.eval()
    model.to(device)
    
    time.sleep(30)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)    

	# WARM UP
    warm_model(test_input)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, hidden_queue_2, derived_model_layer5_pid, derived_model_layer6_pid, derived_model_layer8_pid, derived_model_layer10_pid, derived_model_layer12_pid, derived_model_layer13_pid, derived_model_layer14_pid, derived_model_layer17_pid, derived_model_layer20_pid, hidden_2_mutex)
    end.record()
    torch.cuda.synchronize()    
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)

    pisvm_layer5_layer6_p.join()
    pisvm_layer6_layer8_p.join()
    pisvm_layer8_layer10_p.join()
    pisvm_layer10_layer12_p.join()
    pisvm_layer12_layer13_p.join()
    pisvm_layer13_layer14_p.join()
    pisvm_layer14_layer17_p.join()
    pisvm_layer17_layer20_p.join()	

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

    hidden_queue_2.close()
    d_m_check_queue.close()


    d_m_layer5_queue.close()
    d_m_layer6_queue.close()
    d_m_layer8_queue.close()
    d_m_layer10_queue.close()
    d_m_layer12_queue.close()
    d_m_layer13_queue.close()
    d_m_layer14_queue.close()
    d_m_layer17_queue.close()
    d_m_layer20_queue.close()

"""7.285 sec"""