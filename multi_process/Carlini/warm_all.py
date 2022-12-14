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


    def forward(self, x, hidden_queue, hidden_queue_2, vi_svm_relu_2_pid, derived_model_relu_2_pid, vi_svm_relu_4_pid, derived_model_relu_4_pid, vi_svm_maxpool2d_5_pid, derived_model_maxpool2d_5_pid, vi_svm_relu_7_pid, derived_model_relu_7_pid, vi_svm_relu_9_pid, derived_model_relu_9_pid, vi_svm_maxpool2d_10_pid, derived_model_maxpool2d_10_pid, vi_svm_flatten_11_pid, derived_model_flatten_11_pid, vi_svm_relu_13_pid, derived_model_relu_13_pid, vi_svm_relu_15_pid, derived_model_relu_15_pid, hidden_1_mutex, hidden_2_mutex):
        out = self.layer1(x)

        out = self.layer2(out)	#
        out_1 = out
        out_1 = out_1.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_1)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_2_pid, signal.SIGUSR1)	# visvm process??? signal ??????
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_1)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_2_pid, signal.SIGUSR1)

        out = self.layer3(out)
                
        out = self.layer4(out)   # 
        out_2 = out
        out_2 = out_2.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_2)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_4_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_2)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_4_pid, signal.SIGUSR1)
          
        out = self.layer5(out)   #
        out_3 = out
        out_3 = out_3.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_3)
        hidden_1_mutex.release()
        os.kill(vi_svm_maxpool2d_5_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_3)
        hidden_2_mutex.release()
        os.kill(derived_model_maxpool2d_5_pid, signal.SIGUSR1)
        
        out = self.layer6(out)   

        out = self.layer7(out)	 #
        out_4 = out
        out_4 = out_4.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_4)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_7_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_4)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_7_pid, signal.SIGUSR1)
        
        out = self.layer8(out)       
        
        out = self.layer9(out)   #
        out_5 = out
        out_5 = out_5.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_5)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_9_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_5)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_9_pid, signal.SIGUSR1)

        out = self.layer10(out)  #
        out_6 = out
        out_6 = out_6.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_6)
        hidden_1_mutex.release()
        os.kill(vi_svm_maxpool2d_10_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_6)
        hidden_2_mutex.release()
        os.kill(derived_model_maxpool2d_10_pid, signal.SIGUSR1)

        out = self.layer11(out)  #
        out_7 = out
        out_7 = out_7.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_7)
        hidden_1_mutex.release()
        os.kill(vi_svm_flatten_11_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_7)
        hidden_2_mutex.release()
        os.kill(derived_model_flatten_11_pid, signal.SIGUSR1)

        out = self.layer12(out)
        
        out = self.layer13(out)  #
        out_8 = out
        out_8 = out_8.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_8)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_13_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_8)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_13_pid, signal.SIGUSR1)

        out = self.layer14(out)
        
        out = self.layer15(out)  #
        out_9 = out
        out_9 = out_9.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_9)
        hidden_1_mutex.release()
        os.kill(vi_svm_relu_15_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_9)
        hidden_2_mutex.release()
        os.kill(derived_model_relu_15_pid, signal.SIGUSR1)

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
        signal.signal(signal.SIGUSR1, self.visvm_handler) # ????????? ?????? 
        self.model = OneClassSVM()
        self.model.load_from_file(self.visvm_path)
        # visvm WARM UP
        self.warm_input = np.zeros((1,1))
        self.model.predict_gpu(self.warm_input)
        # os.kill(os.getppid(), signal.SIGUSR1)
        print(self.name, ' VISVM START')
        signal.pause()	#	target model???????????? signal ?????? ???????????? pause
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

        # derived model output queue??? ?????? ??????????????? -> pisvm?????? get()??? ?????? ?????? ?????????
        """derived model?????? pisvm?????? ???????????? ??? ?????? ???????????? visvm ??????????????? ????????? ?????????"""
        if(self.index == 0 or self.index == 8):		#pisvm?????? ????????? ???????????? ????????? ????????? ??????
            self.output_queue.put(self.derived_model_output)
        else:
            self.output_queue.put(self.derived_model_output)
            self.output_queue.put(self.derived_model_output)
			
        """Alexnet ?????? ??? ?????? ?????? ???????????? ?????? ??????"""	
        if(self.index != 0):
            flag = True
            while(flag):
                self.checking = self.checker.get()
                self.checked = self.checking
                self.checker.put(self.checking)
                if(self.checked[self.index -1] != 0):
                    flag = False
            self.before_index_output = self.before_output_queue.get()
            self.corrent_index_output = self.output_queue.get()
            self.pi_svm_input = np.concatenate((self.before_index_output, self.corrent_index_output), axis = 1)
            print(self.name, ' PI SVM input shape : ', self.pi_svm_input)
            self.pisvm_result = self.pi_svm_model.predict_cpu(self.pi_svm_input)


        
        self.mutex.acquire()
        self.checkbox = self.checker.get()
        self.checkbox[self.index] = 1
        self.checker.put(self.checkbox)
        self.mutex.release()
		#index = 0??? ?????? ?????? ?????? ?????? before_output_queue??? ?????? ?????? ????????? ??????
        

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
    d_m_checker = np.array([0,0,0,0,0,0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)

    d_m_relu_2_queue = Queue()
    d_m_relu_4_queue = Queue()
    d_m_maxpool2d_5_queue = Queue()
    d_m_relu_7_queue = Queue()
    d_m_relu_9_queue = Queue()
    d_m_maxpool2d_10_queue = Queue()
    d_m_flatten_11_queue = Queue()
    d_m_relu_13_queue = Queue()
    d_m_relu_15_queue = Queue()

    vi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
    dm_path = ['/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_2_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_4_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/MaxPool2d_5_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_7_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_9_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/MaxPool2d_10_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/Flatten_11_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_13_derived_model.pt', '/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_15_derived_model.pt']
    pi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']

    print('derived model load start')
    derived_model_relu_2 = derived_model.Derived_Model1()
    derived_model_relu_4 = derived_model.Derived_Model2()
    derived_model_maxpool2d_5 = derived_model.Derived_Model3()
    derived_model_relu_7 = derived_model.Derived_Model4()
    derived_model_relu_9 = derived_model.Derived_Model5()
    derived_model_maxpool2d_10 = derived_model.Derived_Model6()
    derived_model_flatten_11 = derived_model.Derived_Model7()
    derived_model_relu_13 = derived_model.Derived_Model8()
    derived_model_relu_15 = derived_model.Derived_Model9()
    print('derived model load done')

	# create vi svm process
    vi_svm_relu_2_p = Visvm(hidden_queue, 'RELU2', vi_svm_path[0], hidden_1_mutex)
    vi_svm_relu_4_p = Visvm(hidden_queue, 'RELU4', vi_svm_path[0], hidden_1_mutex)	#original -> vi_svm_path[1]
    vi_svm_maxpool2d_5_p = Visvm(hidden_queue, 'MAX5', vi_svm_path[0], hidden_1_mutex)
    vi_svm_relu_7_p = Visvm(hidden_queue, 'RELU7', vi_svm_path[0], hidden_1_mutex)
    vi_svm_relu_9_p = Visvm(hidden_queue, 'RELU9', vi_svm_path[0], hidden_1_mutex)
    vi_svm_maxpool2d_10_p = Visvm(hidden_queue, 'MAX10', vi_svm_path[0], hidden_1_mutex)
    vi_svm_flatten_11_p = Visvm(hidden_queue, 'FLAT11', vi_svm_path[0], hidden_1_mutex)
    vi_svm_relu_13_p = Visvm(hidden_queue, 'RELU13', vi_svm_path[0], hidden_1_mutex)
    vi_svm_relu_15_p = Visvm(hidden_queue, 'RELU15', vi_svm_path[0], hidden_1_mutex)

	# visvm process start
    print('visvm process start')
    vi_svm_relu_2_p.start()
    vi_svm_relu_4_p.start()
    vi_svm_maxpool2d_5_p.start()
    vi_svm_relu_7_p.start()
    vi_svm_relu_9_p.start()
    vi_svm_maxpool2d_10_p.start()
    vi_svm_flatten_11_p.start()
    vi_svm_relu_13_p.start()
    vi_svm_relu_15_p.start()

	# get visvm process pid
    vi_svm_relu_2_pid = vi_svm_relu_2_p.pid
    vi_svm_relu_4_pid = vi_svm_relu_4_p.pid
    vi_svm_maxpool2d_5_pid = vi_svm_maxpool2d_5_p.pid
    vi_svm_relu_7_pid = vi_svm_relu_7_p.pid
    vi_svm_relu_9_pid = vi_svm_relu_9_p.pid
    vi_svm_maxpool2d_10_pid = vi_svm_maxpool2d_10_p.pid
    vi_svm_flatten_11_pid = vi_svm_flatten_11_p.pid
    vi_svm_relu_13_pid = vi_svm_relu_13_p.pid
    vi_svm_relu_15_pid = vi_svm_relu_15_p.pid

    derived_model_relu_2_p = DerivedModel('DM RELU2', 0, d_m_check_queue, hidden_queue_2, derived_model_relu_2, d_m_relu_2_queue, d_m_relu_2_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 21632)
    derived_model_relu_4_p = DerivedModel('DM RELU4', 1, d_m_check_queue, hidden_queue_2, derived_model_relu_4, d_m_relu_2_queue, d_m_relu_4_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 18432)
    derived_model_maxpool2d_5_p = DerivedModel('DM MAX5', 2, d_m_check_queue, hidden_queue_2, derived_model_maxpool2d_5, d_m_relu_4_queue, d_m_maxpool2d_5_queue, mutex, hidden_2_mutex, dm_path[2], pi_svm_path[0], 4608)
    derived_model_relu_7_p = DerivedModel('DM RELU7', 3, d_m_check_queue, hidden_queue_2, derived_model_relu_7, d_m_maxpool2d_5_queue, d_m_relu_7_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 6400)
    derived_model_relu_9_p = DerivedModel('DM RELU9', 4, d_m_check_queue, hidden_queue_2, derived_model_relu_9, d_m_relu_7_queue, d_m_relu_9_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 4096)
    derived_model_maxpool2d_10_p = DerivedModel('DM MAX10', 5, d_m_check_queue, hidden_queue_2, derived_model_maxpool2d_10, d_m_relu_9_queue, d_m_maxpool2d_10_queue, mutex, hidden_2_mutex, dm_path[5], pi_svm_path[0], 1024)
    derived_model_flatten_11_p = DerivedModel('DM FLAT11', 6, d_m_check_queue, hidden_queue_2, derived_model_flatten_11, d_m_maxpool2d_10_queue, d_m_flatten_11_queue, mutex, hidden_2_mutex, dm_path[6], pi_svm_path[0], 1024)
    derived_model_relu_13_p = DerivedModel('DM RELU13', 7, d_m_check_queue, hidden_queue_2, derived_model_relu_13, d_m_flatten_11_queue, d_m_relu_13_queue, mutex, hidden_2_mutex, dm_path[7], pi_svm_path[0], 200)
    derived_model_relu_15_p = DerivedModel('DM RELU15', 8, d_m_check_queue, hidden_queue_2, derived_model_relu_15, d_m_relu_13_queue, d_m_relu_15_queue, mutex, hidden_2_mutex, dm_path[8], pi_svm_path[0], 200)

    print('derived model process start')
    derived_model_relu_2_p.start()
    derived_model_relu_4_p.start()
    derived_model_maxpool2d_5_p.start()

    derived_model_relu_7_p.start()
    derived_model_relu_9_p.start()
    derived_model_maxpool2d_10_p.start()
    derived_model_flatten_11_p.start()
    derived_model_relu_13_p.start()
    derived_model_relu_15_p.start()

    derived_model_relu_2_pid = derived_model_relu_2_p.pid
    derived_model_relu_4_pid = derived_model_relu_4_p.pid
    derived_model_maxpool2d_5_pid = derived_model_maxpool2d_5_p.pid
    derived_model_relu_7_pid = derived_model_relu_7_p.pid
    derived_model_relu_9_pid = derived_model_relu_9_p.pid
    derived_model_maxpool2d_10_pid = derived_model_maxpool2d_10_p.pid
    derived_model_flatten_11_pid = derived_model_flatten_11_p.pid
    derived_model_relu_13_pid = derived_model_relu_13_p.pid
    derived_model_relu_15_pid = derived_model_relu_15_p.pid


    # WARM UP MODEL LOAD
    warm_model = CarliniModel2()
    warm_model.load_state_dict(torch.load('./warmcarlini.pt'))
    warm_model.eval()
    warm_model.to(device)

    model = CarliniModel()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/carlini_process_final.pt'))
    model.eval()
    model.to(device)

    time.sleep(30)
    test_data = torch.rand(1,1,28,28).to(device)

    warm_model(test_data)
    print('WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()

    model(test_data,hidden_queue, hidden_queue_2, vi_svm_relu_2_pid, derived_model_relu_2_pid, vi_svm_relu_4_pid, derived_model_relu_4_pid, vi_svm_maxpool2d_5_pid, derived_model_maxpool2d_5_pid, vi_svm_relu_7_pid, derived_model_relu_7_pid, vi_svm_relu_9_pid, derived_model_relu_9_pid, vi_svm_maxpool2d_10_pid, derived_model_maxpool2d_10_pid, vi_svm_flatten_11_pid, derived_model_flatten_11_pid, vi_svm_relu_13_pid, derived_model_relu_13_pid, vi_svm_relu_15_pid, derived_model_relu_15_pid, hidden_1_mutex, hidden_2_mutex)
    end.record()
    torch.cuda.synchronize()
    print("INFERENCE TIME : ", start.elapsed_time(end)/1000)


    vi_svm_relu_2_p.join()
    vi_svm_relu_4_p.join()
    vi_svm_maxpool2d_5_p.join()
    vi_svm_relu_7_p.join()
    vi_svm_relu_9_p.join()
    vi_svm_maxpool2d_10_p.join()
    vi_svm_flatten_11_p.join()
    vi_svm_relu_13_p.join()
    vi_svm_relu_15_p.join()

    derived_model_relu_2_p.join()
    derived_model_relu_4_p.join()
    derived_model_maxpool2d_5_p.join()
    derived_model_relu_7_p.join()
    derived_model_relu_9_p.join()
    derived_model_maxpool2d_10_p.join()
    derived_model_flatten_11_p.join()
    derived_model_relu_13_p.join()
    derived_model_relu_15_p.join()

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
    d_m_relu_9_queue.close()
    d_m_maxpool2d_10_queue.close()
    d_m_flatten_11_queue.close()
    d_m_relu_13_queue.close()
    d_m_relu_15_queue.close()