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


    def forward(self, x, hidden_queue, hidden_queue_2, derived_model_relu_2_pid, derived_model_relu_4_pid, derived_model_maxpool2d_5_pid, derived_model_relu_7_pid, vi_svm_maxpool2d_10_pid, vi_svm_flatten_11_pid, vi_svm_relu_13_pid, vi_svm_relu_15_pid, hidden_1_mutex, hidden_2_mutex):
        out = self.layer1(x)

        out = self.layer2(out)	#
        out_1 = out
        out_1 = out_1.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_1)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_2_pid, signal.SIGUSR1)

        out = self.layer3(out)
                
        out = self.layer4(out)   #
        out_2 = out
        out_2 = out_2.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_2)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_4_pid, signal.SIGUSR1) 
          
        out = self.layer5(out)   #
        out_3 = out
        out_3 = out_3.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_3)
        hidden_2_mutex.release()
        os.kill(derived_model_maxpool2d_5_pid, signal.SIGUSR1)
        
        out = self.layer6(out)   

        out = self.layer7(out)	 #
        out_4 = out
        out_4 = out_4.detach().cpu().numpy()
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_4)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_7_pid, signal.SIGUSR1)
        
        out = self.layer8(out)       
        
        out = self.layer9(out)   #

        out = self.layer10(out)  #
        out_6 = out
        out_6 = out_6.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_6)
        hidden_1_mutex.release()
        os.kill(vi_svm_maxpool2d_10_pid, signal.SIGUSR1)

        out = self.layer11(out)  #
        out_7 = out
        out_7 = out_7.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_7)
        hidden_1_mutex.release()
        os.kill(vi_svm_flatten_11_pid, signal.SIGUSR1)

        out = self.layer12(out)
        
        out = self.layer13(out)  #
        out_8 = out
        out_8 = out_8.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_8)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_13_pid, signal.SIGUSR1)

        out = self.layer14(out)
        
        out = self.layer15(out)  #
        out_9 = out
        out_9 = out_9.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_9)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_15_pid, signal.SIGUSR1)

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

class DerivedModel(Process):		

    def __init__(self, name, index, checker, hidden_queue, derived_model, before_output_queue, output_queue, mutex, hidden_2_mutex, dm_path, pisvm_path, dummy_shape):
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

        # derived model output queue에 두번 삽입해야함 -> pisvm에서 get()을 두번 해야 하므로
        """derived model에서 pisvm으로 시그널은 잘 가는 상태이고 visvm 인퍼런스가 안되는 상태임"""
        # if(self.index == 0 or self.index == 8):		#pisvm에서 문제가 생긴다면 여기가 문제일 것임
        #     self.output_queue.put(self.derived_model_output)
        # else:
        #     self.output_queue.put(self.derived_model_output)
        #     self.output_queue.put(self.derived_model_output)
        self.output_queue.put(self.derived_model_output)
        """Alexnet 현재 이 부분 다른 것들이랑 순서 다름"""	
        if(self.index != 0):
            flag = True
            while(flag):
                self.checking = self.checker.get()
                self.checked = self.checking
                self.checker.put(self.checking)
                if(self.checked[self.index -1] != 0):
                    flag = False
            self.before_index_output = self.before_output_queue.get()
            # self.corrent_index_output = self.output_queue.get()
            # self.pi_svm_input = np.concatenate((self.before_index_output, self.corrent_index_output), axis = 1)
            self.pi_svm_input = np.concatenate((self.before_index_output, self.derived_model_output), axis = 1)
            # print(self.name, ' PI SVM input shape : ', self.pi_svm_input)
            self.pisvm_result = self.pi_svm_model.predict_cpu(self.pi_svm_input)


        
        self.mutex.acquire()
        self.checkbox = self.checker.get()
        self.checkbox[self.index] = 1
        self.checker.put(self.checkbox)
        self.mutex.release()
		#index = 0이 가장 먼저 끝날 경우 before_output_queue의 값이 없을 경우도 있음
        

        # end_dm = time.time()
        # print(self.name, f" DM time : {end_dm - start_dm:.3f} sec")


        # print(self.name, 'RESLUT : ', self.derived_model_output.shape)
            
    def run(self):
        signal.signal(signal.SIGUSR1, self.derived_model_handler)
        self.derived_model.load_state_dict(torch.load(self.dm_path, map_location ='cuda:0'))
        self.derived_model.eval()
        self.derived_model = self.derived_model.to(device)
        #derived model WARM UP
        dummy_input = torch.ones(1,self.dummy_shape).to(device)
        self.derived_model(dummy_input)
        self.pi_svm_model = OneClassSVM()
        self.pi_svm_model.load_from_file(self.pisvm_path)
        #pisvm WARM UP
        self.warm_input = np.zeros((1,1))
        self.pi_svm_model.predict_cpu(self.warm_input)

        print(self.name, ' DERIVED process start')
        signal.pause()
        print(self.name, ' DERIVED process done')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    mutex = Lock()
    hidden_1_mutex = Lock()
    hidden_2_mutex = Lock()

    hidden_queue = Queue()
    hidden_queue_2 = Queue()
    d_m_checker = np.array([0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)

    d_m_relu_2_queue = Queue()
    d_m_relu_4_queue = Queue()
    d_m_maxpool2d_5_queue = Queue()
    d_m_relu_7_queue = Queue()


    vi_svm_path = ['/home/nvidia/joo/models/Carlini/VISVM/layer10_visvm.bin', '/home/nvidia/joo/models/Carlini/VISVM/layer11_visvm.bin', '/home/nvidia/joo/models/Carlini/VISVM/layer13_visvm.bin', '/home/nvidia/joo/models/Carlini/VISVM/layer15_visvm.bin']
    dm_path = ['/home/nvidia/joo/models/Carlini/D_M/MP/layer2_derived_model_dict.pt', '/home/nvidia/joo/models/Carlini/D_M/MP/layer4_derived_model_dict.pt', '/home/nvidia/joo/models/Carlini/D_M/MP/layer5_derived_model_dict.pt', '/home/nvidia/joo/models/Carlini/D_M/MP/layer7_derived_model_dict.pt']
    pi_svm_path = ['/home/nvidia/joo/models/Carlini/PISVM/layer2_layer4_pisvm.bin', '/home/nvidia/joo/models/Carlini/PISVM/layer4_layer5_pisvm.bin', '/home/nvidia/joo/models/Carlini/PISVM/layer5_layer7_pisvm.bin']
    print('derived model load start')
    derived_model_relu_2 = derived_model.Derived_Model2()
    derived_model_relu_4 = derived_model.Derived_Model4()
    derived_model_maxpool2d_5 = derived_model.Derived_Model5()
    derived_model_relu_7 = derived_model.Derived_Model7()

    print('derived model load done')

	# create vi svm process

    vi_svm_maxpool2d_10_p = Visvm(hidden_queue, 'MAX10', vi_svm_path[0], hidden_1_mutex)
    vi_svm_flatten_11_p = Visvm(hidden_queue, 'FLAT11', vi_svm_path[1], hidden_1_mutex)
    vi_svm_relu_13_p = Visvm(hidden_queue, 'RELU13', vi_svm_path[2], hidden_1_mutex)
    vi_svm_relu_15_p = Visvm(hidden_queue, 'RELU15', vi_svm_path[3], hidden_1_mutex)

	# visvm process start
    print('visvm process start')

    vi_svm_maxpool2d_10_p.start()
    vi_svm_flatten_11_p.start()
    vi_svm_relu_13_p.start()
    vi_svm_relu_15_p.start()

	# get visvm process pid

    vi_svm_maxpool2d_10_pid = vi_svm_maxpool2d_10_p.pid
    vi_svm_flatten_11_pid = vi_svm_flatten_11_p.pid
    vi_svm_relu_13_pid = vi_svm_relu_13_p.pid
    vi_svm_relu_15_pid = vi_svm_relu_15_p.pid

    derived_model_relu_2_p = DerivedModel('DM RELU2', 0, d_m_check_queue, hidden_queue_2, derived_model_relu_2, d_m_relu_2_queue, d_m_relu_2_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 57600)
    derived_model_relu_4_p = DerivedModel('DM RELU4', 1, d_m_check_queue, hidden_queue_2, derived_model_relu_4, d_m_relu_2_queue, d_m_relu_4_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 50176)
    derived_model_maxpool2d_5_p = DerivedModel('DM MAX5', 2, d_m_check_queue, hidden_queue_2, derived_model_maxpool2d_5, d_m_relu_4_queue, d_m_maxpool2d_5_queue, mutex, hidden_2_mutex, dm_path[2], pi_svm_path[1], 12544)
    derived_model_relu_7_p = DerivedModel('DM RELU7', 3, d_m_check_queue, hidden_queue_2, derived_model_relu_7, d_m_maxpool2d_5_queue, d_m_relu_7_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[2], 18432)


    print('derived model process start')
    derived_model_relu_2_p.start()
    derived_model_relu_4_p.start()
    derived_model_maxpool2d_5_p.start()
    derived_model_relu_7_p.start()



    derived_model_relu_2_pid = derived_model_relu_2_p.pid
    derived_model_relu_4_pid = derived_model_relu_4_p.pid
    derived_model_maxpool2d_5_pid = derived_model_maxpool2d_5_p.pid
    derived_model_relu_7_pid = derived_model_relu_7_p.pid



    # WARM UP MODEL LOAD
    warm_model = CarliniModel2()
    warm_model.load_state_dict(torch.load('./warmcarlini_cifar10.pt'))
    warm_model.eval()
    warm_model.to(device)

    model = CarliniModel()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/Carlini/carlini_cifar10_dict.pt'))
    model.eval()
    model.to(device)

    time.sleep(30)
    test_data = torch.rand(1,3,32,32).to(device)

    warm_model(test_data)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()

    model(test_data,hidden_queue, hidden_queue_2, derived_model_relu_2_pid, derived_model_relu_4_pid, derived_model_maxpool2d_5_pid, derived_model_relu_7_pid, vi_svm_maxpool2d_10_pid, vi_svm_flatten_11_pid, vi_svm_relu_13_pid, vi_svm_relu_15_pid, hidden_1_mutex, hidden_2_mutex)
    end.record()
    torch.cuda.synchronize()
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)



    vi_svm_maxpool2d_10_p.join()
    vi_svm_flatten_11_p.join()
    vi_svm_relu_13_p.join()
    vi_svm_relu_15_p.join()

    derived_model_relu_2_p.join()
    derived_model_relu_4_p.join()
    derived_model_maxpool2d_5_p.join()
    derived_model_relu_7_p.join()


    end_total.record()
    torch.cuda.synchronize()
    print("TOTAL TIME : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()
    hidden_queue_2.close()
    d_m_check_queue.close()


    d_m_relu_2_queue.close()
    d_m_relu_4_queue.close()
    d_m_maxpool2d_5_queue.close()
    d_m_relu_7_queue.close()
