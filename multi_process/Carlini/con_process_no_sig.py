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



    def forward(self, x, hidden_queue_relu_2, hidden_queue_2_relu_2, vi_svm_relu_2_pid, derived_model_relu_2_pid, hidden_queue_relu_4, hidden_queue_2_relu_4, vi_svm_relu_4_pid, derived_model_relu_4_pid, hidden_queue_maxpool2d_5, hidden_queue_2_maxpool2d_5, vi_svm_maxpool2d_5_pid, derived_model_maxpool2d_5_pid, hidden_queue_relu_7, hidden_queue_2_relu_7, vi_svm_relu_7_pid, derived_model_relu_7_pid, hidden_queue_relu_9, hidden_queue_2_relu_9, vi_svm_relu_9_pid, derived_model_relu_9_pid, hidden_queue_maxpool2d_10, hidden_queue_2_maxpool2d_10, vi_svm_maxpool2d_10_pid, derived_model_maxpool2d_10_pid, hidden_queue_flatten_11, hidden_queue_2_flatten_11, vi_svm_flatten_11_pid, derived_model_flatten_11_pid, hidden_queue_relu_13, hidden_queue_2_relu_13, vi_svm_relu_13_pid, derived_model_relu_13_pid, hidden_queue_relu_15, hidden_queue_2_relu_15, vi_svm_relu_15_pid, derived_model_relu_15_pid):	# each out size of byte : 64
        start_forward = time.time()
        out = self.layer1(x)
    
        out = self.layer2(out)	#
        out_1 = out
        out_1 = out_1.detach().cpu().numpy()
        hidden_queue_relu_2.put(out_1)
        os.kill(vi_svm_relu_2_pid, signal.SIGUSR1)	# visvm process로 signal 전송
        hidden_queue_2_relu_2.put(out_1)
        os.kill(derived_model_relu_2_pid, signal.SIGUSR1)
        
        out = self.layer3(out)
                
        out = self.layer4(out)   # 
        out_2 = out
        out_2 = out_2.detach().cpu().numpy()
        hidden_queue_relu_4.put(out_2)
        os.kill(vi_svm_relu_4_pid, signal.SIGUSR1)
        hidden_queue_2_relu_4.put(out_2)
        os.kill(derived_model_relu_4_pid, signal.SIGUSR1)

        
        
        
        out = self.layer5(out)   #
        out_3 = out
        out_3 = out_3.detach().cpu().numpy()
        hidden_queue_maxpool2d_5.put(out_3)
        os.kill(vi_svm_maxpool2d_5_pid, signal.SIGUSR1)
        hidden_queue_2_maxpool2d_5.put(out_3)
        os.kill(derived_model_maxpool2d_5_pid, signal.SIGUSR1)


        
        out = self.layer6(out)   

        out = self.layer7(out)	 #
        out_4 = out
        out_4 = out_4.detach().cpu().numpy()
        hidden_queue_relu_7.put(out_4)
        os.kill(vi_svm_relu_7_pid, signal.SIGUSR1)
        hidden_queue_2_relu_7.put(out_4)
        os.kill(derived_model_relu_7_pid, signal.SIGUSR1)

        
        out = self.layer8(out)       
        
        out = self.layer9(out)   #
        out_5 = out
        out_5 = out_5.detach().cpu().numpy()
        hidden_queue_relu_9.put(out_5)
        os.kill(vi_svm_relu_9_pid, signal.SIGUSR1)
        hidden_queue_2_relu_9.put(out_5)
        os.kill(derived_model_relu_9_pid, signal.SIGUSR1)


        out = self.layer10(out)  #
        out_6 = out
        out_6 = out_6.detach().cpu().numpy()
        hidden_queue_maxpool2d_10.put(out_6)
        os.kill(vi_svm_maxpool2d_10_pid, signal.SIGUSR1)
        hidden_queue_2_maxpool2d_10.put(out_6)
        os.kill(derived_model_maxpool2d_10_pid, signal.SIGUSR1)

        out = self.layer11(out)  #
        out_7 = out
        out_7 = out_7.detach().cpu().numpy()
        hidden_queue_flatten_11.put(out_7)
        os.kill(vi_svm_flatten_11_pid, signal.SIGUSR1)
        hidden_queue_2_flatten_11.put(out_7)
        os.kill(derived_model_flatten_11_pid, signal.SIGUSR1)


        out = self.layer12(out)
        
        out = self.layer13(out)  #
        out_8 = out
        out_8 = out_8.detach().cpu().numpy()
        hidden_queue_relu_13.put(out_8)
        os.kill(vi_svm_relu_13_pid, signal.SIGUSR1)
        hidden_queue_2_relu_13.put(out_8)
        os.kill(derived_model_relu_13_pid, signal.SIGUSR1)


        out = self.layer14(out)
        
        out = self.layer15(out)  #
        out_9 = out
        out_9 = out_9.detach().cpu().numpy()
        hidden_queue_relu_15.put(out_9)
        os.kill(vi_svm_relu_15_pid, signal.SIGUSR1)
        hidden_queue_2_relu_15.put(out_9)
        os.kill(derived_model_relu_15_pid, signal.SIGUSR1)


        out = self.layer16(out)
        end_forward = time.time()
        print(f"target forward time : {end_forward - start_forward:.3f} sec")
        return out

class Visvm(Process):
    def __init__(self, hidden_queue, name, visvm_path):
        super(Process, self).__init__()
        self.hidden_queue = hidden_queue
        self.name = name
        self.visvm_path = visvm_path
        
    def visvm_handler(self, signum, frame):
        start_visvm = time.time()
        self.visvm_input = self.hidden_queue.get()	
        self.visvm_input = np.reshape(self.visvm_input, (1,-1))
        # print(self.name, 'shape : ', self.visvm_input.shape)
		#visvm inference 

        # self.pre_result = self.model.predict_gpu(self.visvm_input)
        self.model.predict_gpu(self.visvm_input)
        end_visvm = time.time()
        print(self.name, f" VISVM time : {end_visvm - start_visvm:.3f} sec")
        # print(self.name, ' predict result : ', self.pre_result)

    def run(self):
        signal.signal(signal.SIGUSR1, self.visvm_handler) # 시그널 링크 
        self.model = OneClassSVM()
        self.model.load_from_file(self.visvm_path)
        # os.kill(os.getppid(), signal.SIGUSR1)
        # print(self.name, ' VISVM START')
        signal.pause()	#	target model에서부터 signal 받기 전까지는 pause
        # print(self.name, ' VISVM DONE')


class DerivedModel(Process):		

    def __init__(self, name, index, checker, hidden_queue, derived_model, output_queue, mutex, pisvm_pid):
        super(Process, self).__init__()
        self.index = index
        self.checker = checker
        self.name = name
        self.hidden_queue = hidden_queue
        self.derived_model = derived_model
        self.output_queue = output_queue
        self.mutex = mutex
        self.pisvm_pid = pisvm_pid


        
    def derived_model_handler(self, signum, frame):
        start_dm = time.time()
        self.derived_model = self.derived_model.to(device)
        self.derived_model_input = self.hidden_queue.get()

        self.derived_model_input = np.reshape(self.derived_model_input, (1,-1))
        # print(self.name, ' SHAPE : ', self.derived_model_input.shape)
        self.derived_model_input = torch.Tensor(self.derived_model_input)
        self.derived_model_input = self.derived_model_input.to(device)
        
        self.derived_model_output = self.derived_model(self.derived_model_input)
        self.derived_model_output = self.derived_model_output.detach().cpu().numpy()
        # derived model output queue에 두번 삽입해야함 -> pisvm에서 get()을 두번 해야 하므로
        """derived model에서 pisvm으로 시그널은 잘 가는 상태이고 visvm 인퍼런스가 안되는 상태임"""
        if(self.index == 0 or self.index == 8):		#pisvm에서 문제가 생긴다면 여기가 문제일 것임
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
        end_dm = time.time()
        print(self.name,f" DM time : {end_dm - start_dm:.3f} sec")


        # print(self.name, 'RESLUT : ', self.derived_model_output.shape)
            
    def run(self):
        signal.signal(signal.SIGUSR1, self.derived_model_handler)
        # print(self.name, ' DERIVED process start')
        signal.pause()
        # print(self.name, ' DERIVED process done')

class Pisvm(Process):
    """
    pisvm load, derived model check array, derived model output(shared memory 생성, 시그널 처리)->concatenate
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
        start_pisvm = time.time()
        self.pi_svm_input_1 = self.output_queue_1.get()
        self.pi_svm_input_2 = self.output_queue_2.get()
        self.pi_svm_input = np.concatenate((self.pi_svm_input_1, self.pi_svm_input_2), axis =1)
        # print(self.name, ' pi svm input shape : ', self.pi_svm_input.shape)

        # self.pisvm_result = self.pisvm.predict_cpu(self.pi_svm_input)
        self.pisvm.predict_cpu(self.pi_svm_input)
        end_pisvm = time.time()
        print(self.name, f" PISVM time : {end_pisvm - start_pisvm:.3f} sec")

        # print(self.name, ' RESLUT : ', self.pisvm_result)


    def run(self):
        signal.signal(signal.SIGUSR1, self.pisvm_handler)

        signal.pause()
        # print(self.name, ' PISVM DONE')




if __name__ == '__main__':	
    # def main_handler(self, signum):
    #     print('MAIN GET SIGNAL')
    # signal.signal(signal.SIGUSR1, main_handler)
    print('main start')
    multiprocessing.set_start_method('spawn')
	# multiprocessing.set_start_method('fork')
    mutex = Lock()
	# hidden layer output queue for target->visvm
    hidden_queue_relu_2 = Queue()
    hidden_queue_relu_4 = Queue()
    hidden_queue_maxpool2d_5 = Queue()
    hidden_queue_relu_7 = Queue()
    hidden_queue_relu_9 = Queue()
    hidden_queue_maxpool2d_10 = Queue()
    hidden_queue_flatten_11 = Queue()
    hidden_queue_relu_13 = Queue()
    hidden_queue_relu_15 = Queue()

	# hidden layer output queue for target->derived model
    hidden_queue_2_relu_2 = Queue()
    hidden_queue_2_relu_4 = Queue()
    hidden_queue_2_maxpool2d_5 = Queue()
    hidden_queue_2_relu_7 = Queue()
    hidden_queue_2_relu_9 = Queue()
    hidden_queue_2_maxpool2d_10 = Queue()
    hidden_queue_2_flatten_11 = Queue()
    hidden_queue_2_relu_13 = Queue()
    hidden_queue_2_relu_15 = Queue()
	# check array when derived model done
    d_m_checker = np.array([0,0,0,0,0,0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)

	# derived model output queue (-> pisvm)
    d_m_relu_2_queue = Queue()
    d_m_relu_4_queue = Queue()
    d_m_maxpool2d_5_queue = Queue()
    d_m_relu_7_queue = Queue()
    d_m_relu_9_queue = Queue()
    d_m_maxpool2d_10_queue = Queue()
    d_m_flatten_11_queue = Queue()
    d_m_relu_13_queue = Queue()
    d_m_relu_15_queue = Queue()



	# visvm load
    print('visvm load start')
    vi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_2_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/ReLU_4_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/MaxPool2d_5_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/ReLU_7_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/ReLU_9_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/MaxPool2d_10_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/Flatten_11_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin','/home/nvidia/joo/models/VI_SVM/ReLU_15_VI_SVM_model.bin']

    print('visvm load done')

	# #derived_model load	
    print('derived model load start')
    derived_model_relu_2 = derived_model.Derived_Model1()
    derived_model_relu_2.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_2_derived_model.pt', map_location='cuda'))
    derived_model_relu_2.eval()
	# derived_model_relu_2.to(device)
    derived_model_relu_4 = derived_model.Derived_Model2()
    derived_model_relu_4.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_4_derived_model.pt', map_location='cuda'))
    derived_model_relu_4.eval()
	# derived_model_relu_4.to(device)

    derived_model_maxpool2d_5 = derived_model.Derived_Model3()
    derived_model_maxpool2d_5.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/MaxPool2d_5_derived_model.pt', map_location='cuda'))
    derived_model_maxpool2d_5.eval()
	# derived_model_maxpool2d_5.to(device)
    derived_model_relu_7 = derived_model.Derived_Model4()
    derived_model_relu_7.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_7_derived_model.pt', map_location='cuda'))
    derived_model_relu_7.eval()
	# derived_model_relu_7.to(device)
    derived_model_relu_9 = derived_model.Derived_Model5()
    derived_model_relu_9.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_9_derived_model.pt', map_location='cuda'))
    derived_model_relu_9.eval()
	# derived_model_relu_9.to(device)
    derived_model_maxpool2d_10 = derived_model.Derived_Model6()
    derived_model_maxpool2d_10.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/MaxPool2d_10_derived_model.pt', map_location='cuda'))
    derived_model_maxpool2d_10.eval()
	# derived_model_maxpool2d_10.to(device)
    derived_model_flatten_11 = derived_model.Derived_Model7()
    derived_model_flatten_11.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/Flatten_11_derived_model.pt', map_location='cuda'))
    derived_model_flatten_11.eval()
	# derived_model_flatten_11.to(device)
    derived_model_relu_13 = derived_model.Derived_Model8()
    derived_model_relu_13.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_13_derived_model.pt', map_location='cuda'))
    derived_model_relu_13.eval()
	# derived_model_relu_13.to(device)
    derived_model_relu_15 = derived_model.Derived_Model9()
    derived_model_relu_15.load_state_dict(torch.load('/home/nvidia/joo/models/D_M/process/Derived_Model/ReLU_15_derived_model.pt', map_location='cuda'))
    derived_model_relu_15.eval()
	# derived_model_relu_15.to(device)
    print('derived model load done')

    print('pisvm load start')
    pi_relu_2_relu_4 = OneClassSVM()
    pi_relu_2_relu_4.load_from_file('/home/nvidia/joo/models/PI_SVM/ReLU_2_ReLU_4_PI_SVM_model.bin')
    print('pisvm load done')

	# create vi svm process
    vi_svm_relu_2_p = Visvm(hidden_queue_relu_2, 'RELU2', vi_svm_path[0])
    vi_svm_relu_4_p = Visvm(hidden_queue_relu_4, 'RELU4', vi_svm_path[0])	# original -> vi_svm_path[1]
    vi_svm_maxpool2d_5_p = Visvm(hidden_queue_maxpool2d_5, 'MAX5', vi_svm_path[2])
    vi_svm_relu_7_p = Visvm(hidden_queue_relu_7, 'RELU7', vi_svm_path[3])
    vi_svm_relu_9_p = Visvm(hidden_queue_relu_9, 'RELU9', vi_svm_path[4])
    vi_svm_maxpool2d_10_p = Visvm(hidden_queue_maxpool2d_10, 'MAX10', vi_svm_path[5])
    vi_svm_flatten_11_p = Visvm(hidden_queue_flatten_11, 'FLAT11', vi_svm_path[6])
    vi_svm_relu_13_p = Visvm(hidden_queue_relu_13, 'RELU13', vi_svm_path[7])
    vi_svm_relu_15_p = Visvm(hidden_queue_relu_15, 'RELU15', vi_svm_path[8])

    pi_svm_relu_2_relu_4_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'RE2RE4', 0, d_m_relu_2_queue, d_m_relu_4_queue, mutex)
    pi_svm_relu_4_maxpool2d_5_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'RE4MX5', 1, d_m_relu_4_queue, d_m_maxpool2d_5_queue, mutex)
    pi_svm_maxpool2d_5_relu_7_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'MX5RE7', 2, d_m_maxpool2d_5_queue, d_m_relu_7_queue, mutex)
    pi_svm_relu_7_relu_9_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'RE7RE9', 3, d_m_relu_7_queue, d_m_relu_9_queue, mutex)
    pi_svm_relu_9_maxpool2d_10_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'RE9MX10', 4, d_m_relu_9_queue, d_m_maxpool2d_10_queue, mutex)
    pi_svm_maxpool2d_10_flatten_11_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'MX10FL11', 5, d_m_maxpool2d_10_queue, d_m_flatten_11_queue, mutex)
    pi_svm_flatten_11_relu_13_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'FL11RE13', 6, d_m_flatten_11_queue, d_m_relu_13_queue, mutex)
    pi_svm_relu_13_relu_15_p = Pisvm(pi_relu_2_relu_4, d_m_check_queue, 'RE13RE15', 7, d_m_relu_13_queue, d_m_relu_15_queue, mutex)


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

	# time.sleep(3)
	
    print('pisvm process start')
    pi_svm_relu_2_relu_4_p.start()
    pi_svm_relu_4_maxpool2d_5_p.start()
    pi_svm_maxpool2d_5_relu_7_p.start()
    pi_svm_relu_7_relu_9_p.start()
    pi_svm_relu_9_maxpool2d_10_p.start()
    pi_svm_maxpool2d_10_flatten_11_p.start()
    pi_svm_flatten_11_relu_13_p.start()
    pi_svm_relu_13_relu_15_p.start()

	
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


	#pid of pisvm process
    pi_svm_relu_2_relu_4_pid = pi_svm_relu_2_relu_4_p.pid
    pi_svm_relu_4_maxpool2d_5_pid = pi_svm_relu_4_maxpool2d_5_p.pid
    pi_svm_maxpool2d_5_relu_7_pid = pi_svm_maxpool2d_5_relu_7_p.pid
    pi_svm_relu_7_relu_9_pid = pi_svm_relu_7_relu_9_p.pid
    pi_svm_relu_9_maxpool2d_10_pid = pi_svm_relu_9_maxpool2d_10_p.pid
    pi_svm_maxpool2d_10_flatten_11_pid = pi_svm_maxpool2d_10_flatten_11_p.pid
    pi_svm_flatten_11_relu_13_pid = pi_svm_flatten_11_relu_13_p.pid
    pi_svm_relu_13_relu_15_pid = pi_svm_relu_13_relu_15_p.pid



    derived_model_relu_2_p = DerivedModel('DM RELU2', 0, d_m_check_queue, hidden_queue_2_relu_2, derived_model_relu_2, d_m_relu_2_queue, mutex, pi_svm_relu_2_relu_4_pid)
    derived_model_relu_4_p = DerivedModel('DM RELU4', 1, d_m_check_queue, hidden_queue_2_relu_4, derived_model_relu_4, d_m_relu_4_queue, mutex, pi_svm_relu_2_relu_4_pid)
    derived_model_maxpool2d_5_p = DerivedModel('DM MAX5', 2, d_m_check_queue, hidden_queue_2_maxpool2d_5, derived_model_maxpool2d_5, d_m_maxpool2d_5_queue, mutex, pi_svm_relu_4_maxpool2d_5_pid)
    derived_model_relu_7_p = DerivedModel('DM RELU7', 3, d_m_check_queue, hidden_queue_2_relu_7, derived_model_relu_7, d_m_relu_7_queue, mutex, pi_svm_maxpool2d_5_relu_7_pid)
    derived_model_relu_9_p = DerivedModel('DM RELU9', 4, d_m_check_queue, hidden_queue_2_relu_9, derived_model_relu_9, d_m_relu_9_queue, mutex, pi_svm_relu_7_relu_9_pid)
    derived_model_maxpool2d_10_p = DerivedModel('DM MAX10', 5, d_m_check_queue, hidden_queue_2_maxpool2d_10, derived_model_maxpool2d_10, d_m_maxpool2d_10_queue, mutex, pi_svm_relu_9_maxpool2d_10_pid)
    derived_model_flatten_11_p = DerivedModel('DM FLAT11', 6, d_m_check_queue, hidden_queue_2_flatten_11, derived_model_flatten_11, d_m_flatten_11_queue, mutex, pi_svm_maxpool2d_10_flatten_11_pid)
    derived_model_relu_13_p = DerivedModel('DM RELU13', 7, d_m_check_queue, hidden_queue_2_relu_13, derived_model_relu_13, d_m_relu_13_queue, mutex, pi_svm_flatten_11_relu_13_pid)
    derived_model_relu_15_p = DerivedModel('DM RELU15', 8, d_m_check_queue, hidden_queue_2_relu_15, derived_model_relu_15, d_m_relu_15_queue, mutex, pi_svm_relu_13_relu_15_pid)

    print('derived model process start')
    derived_model_relu_2_p.start()

    derived_model_relu_4_p.start()
	# print('here2')
    derived_model_maxpool2d_5_p.start()
	# print('here3')

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

    model = CarliniModel()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/carlini_process_final.pt'))
    model.eval()
    model.to(device)
    test_data = torch.rand(1,1,28,28).to(device)

    # print('going to wait signal')
    # signal.pause()
    # signal.pause()
    time.sleep(20)
    print('wake up')
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    start.record()
    model(test_data, hidden_queue_relu_2, hidden_queue_2_relu_2, vi_svm_relu_2_pid, derived_model_relu_2_pid, hidden_queue_relu_4, hidden_queue_2_relu_4, vi_svm_relu_4_pid, derived_model_relu_4_pid, hidden_queue_maxpool2d_5, hidden_queue_2_maxpool2d_5, vi_svm_maxpool2d_5_pid, derived_model_maxpool2d_5_pid, hidden_queue_relu_7, hidden_queue_2_relu_7, vi_svm_relu_7_pid, derived_model_relu_7_pid, hidden_queue_relu_9, hidden_queue_2_relu_9, vi_svm_relu_9_pid, derived_model_relu_9_pid, hidden_queue_maxpool2d_10, hidden_queue_2_maxpool2d_10, vi_svm_maxpool2d_10_pid, derived_model_maxpool2d_10_pid, hidden_queue_flatten_11, hidden_queue_2_flatten_11, vi_svm_flatten_11_pid, derived_model_flatten_11_pid, hidden_queue_relu_13, hidden_queue_2_relu_13, vi_svm_relu_13_pid, derived_model_relu_13_pid, hidden_queue_relu_15, hidden_queue_2_relu_15, vi_svm_relu_15_pid, derived_model_relu_15_pid)
	# model(test_data,hidden_queue, hidden_queue_2, vi_svm_relu_2_pid, derived_model_relu_2_pid, vi_svm_relu_4_pid, derived_model_relu_4_pid, derived_model_maxpool2d_5_pid, dm_mutex)



    vi_svm_relu_2_p.join()
    vi_svm_relu_4_p.join()
    vi_svm_maxpool2d_5_p.join()
    vi_svm_relu_7_p.join()
    vi_svm_relu_9_p.join()
    vi_svm_maxpool2d_10_p.join()
    vi_svm_flatten_11_p.join()
    vi_svm_relu_13_p.join()
    vi_svm_relu_15_p.join()

    pi_svm_relu_2_relu_4_p.join()
    pi_svm_relu_4_maxpool2d_5_p.join()
    pi_svm_maxpool2d_5_relu_7_p.join()
    pi_svm_relu_7_relu_9_p.join()
    pi_svm_relu_9_maxpool2d_10_p.join()
    pi_svm_maxpool2d_10_flatten_11_p.join()
    pi_svm_flatten_11_relu_13_p.join()
    pi_svm_relu_13_relu_15_p.join()



    derived_model_relu_2_p.join()
    derived_model_relu_4_p.join()
    derived_model_maxpool2d_5_p.join()
    derived_model_relu_7_p.join()
    derived_model_relu_9_p.join()
    derived_model_maxpool2d_10_p.join()
    derived_model_flatten_11_p.join()
    derived_model_relu_13_p.join()
    derived_model_relu_15_p.join()
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end) / 1000, "sec")
	


    hidden_queue_relu_2.close()
    hidden_queue_relu_4.close()
    hidden_queue_maxpool2d_5.close()
    hidden_queue_relu_7.close()
    hidden_queue_relu_9.close()
    hidden_queue_maxpool2d_10.close()
    hidden_queue_flatten_11.close()
    hidden_queue_relu_13.close()
    hidden_queue_relu_15.close()

	# hidden layer output queue for target->derived model
    hidden_queue_2_relu_2.close()
    hidden_queue_2_relu_4.close()
    hidden_queue_2_maxpool2d_5.close()
    hidden_queue_2_relu_7.close()
    hidden_queue_2_relu_9.close()
    hidden_queue_2_maxpool2d_10.close()
    hidden_queue_2_flatten_11.close()
    hidden_queue_2_relu_13.close()
    hidden_queue_2_relu_15.close()

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