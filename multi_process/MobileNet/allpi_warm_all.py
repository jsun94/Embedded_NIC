import torch
import torch.nn as nn


from multiprocessing import Process, Queue, Lock
import multiprocessing
import numpy as np
import time
import os
from thundersvm import OneClassSVM
import signal
import derived_model

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


""" WARM UP MODEL """

class MobileNetv1_2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetv1_2, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 2)
        
        #conv2
        self.layer2 = nn.Conv2d(32,32, kernel_size = 3, padding = 1, stride = 1)
        self.layer3 = nn.BatchNorm2d(32)
        self.layer4 = nn.ReLU(inplace = True)
        self.layer5 = nn.Conv2d(32,64, kernel_size = 1, stride = 1)
        self.layer6 = nn.BatchNorm2d(64)
        self.layer7 = nn.ReLU(inplace = True)
        
        #conv3
        self.layer8 = nn.Conv2d(64,64, kernel_size = 3, padding = 1, stride = 2)
        self.layer9 = nn.BatchNorm2d(64)
        self.layer10 = nn.ReLU(inplace = True)
        self.layer11 = nn.Conv2d(64,128, kernel_size = 1, stride = 1)
        self.layer12 = nn.BatchNorm2d(128)
        self.layer13 = nn.ReLU(inplace = True)
        
        #conv4
        self.layer14 = nn.Conv2d(128,128, kernel_size = 3, padding = 1, stride = 1)
        self.layer15 = nn.BatchNorm2d(128)
        self.layer16 = nn.ReLU(inplace = True)
        self.layer17 = nn.Conv2d(128,128, kernel_size = 1, stride = 1)
        self.layer18 = nn.BatchNorm2d(128)
        self.layer19 = nn.ReLU(inplace = True)
        
        #conv5
        self.layer20 = nn.Conv2d(128,128, kernel_size = 3, padding = 1, stride = 2)
        self.layer21 = nn.BatchNorm2d(128)
        self.layer22 = nn.ReLU(inplace = True)
        self.layer23 = nn.Conv2d(128,256, kernel_size = 1, stride = 1)
        self.layer24 = nn.BatchNorm2d(256)
        self.layer25 = nn.ReLU(inplace = True)
        
        #conv6
        self.layer26 = nn.Conv2d(256,256, kernel_size = 3, padding = 1, stride = 1)
        self.layer27 = nn.BatchNorm2d(256)
        self.layer28 = nn.ReLU(inplace = True)
        self.layer29 = nn.Conv2d(256,256, kernel_size = 1, stride = 1)
        self.layer30 = nn.BatchNorm2d(256)
        self.layer31 = nn.ReLU(inplace = True)        

		#conv7
        self.layer32 = nn.Conv2d(256,256, kernel_size = 3, padding = 1, stride = 2)
        self.layer33 = nn.BatchNorm2d(256)
        self.layer34 = nn.ReLU(inplace = True)
        self.layer35 = nn.Conv2d(256,512, kernel_size = 1, stride = 1)
        self.layer36 = nn.BatchNorm2d(512)
        self.layer37 = nn.ReLU(inplace = True)

		#conv8
        self.layer38 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer39 = nn.BatchNorm2d(512)
        self.layer40 = nn.ReLU(inplace = True)
        self.layer41 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU(inplace = True)

		#conv9
        self.layer44 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer45 = nn.BatchNorm2d(512)
        self.layer46 = nn.ReLU(inplace = True)
        self.layer47 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer48 = nn.BatchNorm2d(512)
        self.layer49 = nn.ReLU(inplace = True)
        
		#conv10
        self.layer50 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer51 = nn.BatchNorm2d(512)
        self.layer52 = nn.ReLU(inplace = True)
        self.layer53 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer54 = nn.BatchNorm2d(512)
        self.layer55 = nn.ReLU(inplace = True)
        
		#conv11
        self.layer56 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer57 = nn.BatchNorm2d(512)
        self.layer58 = nn.ReLU(inplace = True)
        self.layer59 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer60 = nn.BatchNorm2d(512)
        self.layer61 = nn.ReLU(inplace = True)
        
		#conv12
        self.layer62 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer63 = nn.BatchNorm2d(512)
        self.layer64 = nn.ReLU(inplace = True)
        self.layer65 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer66 = nn.BatchNorm2d(512)
        self.layer67 = nn.ReLU(inplace = True)
        
		#conv13
        self.layer68 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 2)
        self.layer69 = nn.BatchNorm2d(512)
        self.layer70 = nn.ReLU(inplace = True)
        self.layer71 = nn.Conv2d(512,1024, kernel_size = 1, stride = 1)
        self.layer72 = nn.BatchNorm2d(1024)
        self.layer73 = nn.ReLU(inplace = True)
        
		#conv14
        self.layer74 = nn.Conv2d(1024,1024, kernel_size = 3, padding = 1, stride = 2)
        self.layer75 = nn.BatchNorm2d(1024)
        self.layer76 = nn.ReLU(inplace = True)
        self.layer77 = nn.Conv2d(1024,1024, kernel_size = 1, stride = 1)
        self.layer78 = nn.BatchNorm2d(1024)
        self.layer79 = nn.ReLU(inplace = True)

		#avgpool
        self.layer80 = nn.AvgPool2d(2)
        
        #view from forward
        self.layer81 = nn.Flatten()
        
        #fc
        self.layer82 = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
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
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.layer26(x)
        x = self.layer27(x)
        x = self.layer28(x)
        x = self.layer29(x)
        x = self.layer30(x)
        x = self.layer31(x)
        x = self.layer32(x)
        x = self.layer33(x)
        x = self.layer34(x)
        x = self.layer35(x)
        x = self.layer36(x)
        x = self.layer37(x)
        x = self.layer38(x)
        x = self.layer39(x)
        x = self.layer40(x)
        x = self.layer41(x)
        x = self.layer42(x)
        x = self.layer43(x)
        x = self.layer44(x)
        x = self.layer45(x)
        x = self.layer46(x)
        x = self.layer47(x)
        x = self.layer48(x)
        x = self.layer49(x)
        x = self.layer50(x)
        x = self.layer51(x)
        x = self.layer52(x)
        x = self.layer53(x)
        x = self.layer54(x)
        x = self.layer55(x)
        x = self.layer56(x)
        x = self.layer57(x)    
        x = self.layer58(x)
        x = self.layer59(x)
        x = self.layer60(x)    
        x = self.layer61(x)      
        x = self.layer62(x)
        x = self.layer63(x)
        x = self.layer64(x)        
        x = self.layer65(x)
        x = self.layer66(x)
        x = self.layer67(x)        
        x = self.layer68(x)
        x = self.layer69(x)
        x = self.layer70(x)        
        x = self.layer71(x)
        x = self.layer72(x)
        x = self.layer73(x)       
        x = self.layer74(x)
        x = self.layer75(x)
        x = self.layer76(x)      
        x = self.layer77(x)
        x = self.layer78(x)
        x = self.layer79(x)      
        x = self.layer80(x)    
        x = self.layer81(x)
        x = self.layer82(x)
        return x

""" WARM UP MODEL """

class MobileNetv1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetv1, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 2)
        
        #conv2
        self.layer2 = nn.Conv2d(32,32, kernel_size = 3, padding = 1, stride = 1)
        self.layer3 = nn.BatchNorm2d(32)
        self.layer4 = nn.ReLU(inplace = True)
        self.layer5 = nn.Conv2d(32,64, kernel_size = 1, stride = 1)
        self.layer6 = nn.BatchNorm2d(64)
        self.layer7 = nn.ReLU(inplace = True)
        
        #conv3
        self.layer8 = nn.Conv2d(64,64, kernel_size = 3, padding = 1, stride = 2)
        self.layer9 = nn.BatchNorm2d(64)
        self.layer10 = nn.ReLU(inplace = True)
        self.layer11 = nn.Conv2d(64,128, kernel_size = 1, stride = 1)
        self.layer12 = nn.BatchNorm2d(128)
        self.layer13 = nn.ReLU(inplace = True)
        
        #conv4
        self.layer14 = nn.Conv2d(128,128, kernel_size = 3, padding = 1, stride = 1)
        self.layer15 = nn.BatchNorm2d(128)
        self.layer16 = nn.ReLU(inplace = True)
        self.layer17 = nn.Conv2d(128,128, kernel_size = 1, stride = 1)
        self.layer18 = nn.BatchNorm2d(128)
        self.layer19 = nn.ReLU(inplace = True)
        
        #conv5
        self.layer20 = nn.Conv2d(128,128, kernel_size = 3, padding = 1, stride = 2)
        self.layer21 = nn.BatchNorm2d(128)
        self.layer22 = nn.ReLU(inplace = True)
        self.layer23 = nn.Conv2d(128,256, kernel_size = 1, stride = 1)
        self.layer24 = nn.BatchNorm2d(256)
        self.layer25 = nn.ReLU(inplace = True)
        
        #conv6
        self.layer26 = nn.Conv2d(256,256, kernel_size = 3, padding = 1, stride = 1)
        self.layer27 = nn.BatchNorm2d(256)
        self.layer28 = nn.ReLU(inplace = True)
        self.layer29 = nn.Conv2d(256,256, kernel_size = 1, stride = 1)
        self.layer30 = nn.BatchNorm2d(256)
        self.layer31 = nn.ReLU(inplace = True)        

		#conv7
        self.layer32 = nn.Conv2d(256,256, kernel_size = 3, padding = 1, stride = 2)
        self.layer33 = nn.BatchNorm2d(256)
        self.layer34 = nn.ReLU(inplace = True)
        self.layer35 = nn.Conv2d(256,512, kernel_size = 1, stride = 1)
        self.layer36 = nn.BatchNorm2d(512)
        self.layer37 = nn.ReLU(inplace = True)

		#conv8
        self.layer38 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer39 = nn.BatchNorm2d(512)
        self.layer40 = nn.ReLU(inplace = True)
        self.layer41 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer42 = nn.BatchNorm2d(512)
        self.layer43 = nn.ReLU(inplace = True)

		#conv9
        self.layer44 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer45 = nn.BatchNorm2d(512)
        self.layer46 = nn.ReLU(inplace = True)
        self.layer47 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer48 = nn.BatchNorm2d(512)
        self.layer49 = nn.ReLU(inplace = True)
        
		#conv10
        self.layer50 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer51 = nn.BatchNorm2d(512)
        self.layer52 = nn.ReLU(inplace = True)
        self.layer53 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer54 = nn.BatchNorm2d(512)
        self.layer55 = nn.ReLU(inplace = True)
        
		#conv11
        self.layer56 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer57 = nn.BatchNorm2d(512)
        self.layer58 = nn.ReLU(inplace = True)
        self.layer59 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer60 = nn.BatchNorm2d(512)
        self.layer61 = nn.ReLU(inplace = True)
        
		#conv12
        self.layer62 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 1)
        self.layer63 = nn.BatchNorm2d(512)
        self.layer64 = nn.ReLU(inplace = True)
        self.layer65 = nn.Conv2d(512,512, kernel_size = 1, stride = 1)
        self.layer66 = nn.BatchNorm2d(512)
        self.layer67 = nn.ReLU(inplace = True)
        
		#conv13
        self.layer68 = nn.Conv2d(512,512, kernel_size = 3, padding = 1, stride = 2)
        self.layer69 = nn.BatchNorm2d(512)
        self.layer70 = nn.ReLU(inplace = True)
        self.layer71 = nn.Conv2d(512,1024, kernel_size = 1, stride = 1)
        self.layer72 = nn.BatchNorm2d(1024)
        self.layer73 = nn.ReLU(inplace = True)
        
		#conv14
        self.layer74 = nn.Conv2d(1024,1024, kernel_size = 3, padding = 1, stride = 2)
        self.layer75 = nn.BatchNorm2d(1024)
        self.layer76 = nn.ReLU(inplace = True)
        self.layer77 = nn.Conv2d(1024,1024, kernel_size = 1, stride = 1)
        self.layer78 = nn.BatchNorm2d(1024)
        self.layer79 = nn.ReLU(inplace = True)

		#avgpool
        self.layer80 = nn.AvgPool2d(2)
        
        #view from forward
        self.layer81 = nn.Flatten()
        
        #fc
        self.layer82 = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden_queue, hidden_queue_2, vi_svm_layer76_pid, derived_model_layer76_pid, vi_svm_layer79_pid, derived_model_layer79_pid, vi_svm_layer80_pid, derived_model_layer80_pid, vi_svm_layer81_pid, derived_model_layer81_pid, hidden_1_mutex, hidden_2_mutex):
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
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.layer24(x)
        x = self.layer25(x)
        x = self.layer26(x)
        x = self.layer27(x)
        x = self.layer28(x)
        x = self.layer29(x)
        x = self.layer30(x)
        x = self.layer31(x)
        x = self.layer32(x)
        x = self.layer33(x)
        x = self.layer34(x)
        x = self.layer35(x)
        x = self.layer36(x)
        x = self.layer37(x)
        x = self.layer38(x)
        x = self.layer39(x)
        x = self.layer40(x)
        x = self.layer41(x)
        x = self.layer42(x)
        x = self.layer43(x)
        x = self.layer44(x)
        x = self.layer45(x)
        x = self.layer46(x)
        x = self.layer47(x)
        x = self.layer48(x)
        x = self.layer49(x)
        x = self.layer50(x)
        x = self.layer51(x)
        x = self.layer52(x)
        x = self.layer53(x)
        x = self.layer54(x)
        x = self.layer55(x)
        x = self.layer56(x)
        x = self.layer57(x)
        
        x = self.layer58(x)
        
        x = self.layer59(x)
        x = self.layer60(x)
        
        x = self.layer61(x)
        
        x = self.layer62(x)
        x = self.layer63(x)
        x = self.layer64(x)
        
        x = self.layer65(x)
        x = self.layer66(x)
        x = self.layer67(x)
        
        x = self.layer68(x)
        x = self.layer69(x)
        x = self.layer70(x)
        
        x = self.layer71(x)
        x = self.layer72(x)
        x = self.layer73(x)
        
        x = self.layer74(x)
        x = self.layer75(x)
        x = self.layer76(x)
        out_7 = x
        out_7 = out_7.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_7)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer76_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_7)
        hidden_2_mutex.release()
        os.kill(derived_model_layer76_pid, signal.SIGUSR1)
        
        x = self.layer77(x)
        x = self.layer78(x)
        x = self.layer79(x)
        out_8 = x
        out_8 = out_8.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_8)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer79_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_8)
        hidden_2_mutex.release()
        os.kill(derived_model_layer79_pid, signal.SIGUSR1)
        
        x = self.layer80(x)
        out_9 = x
        out_9 = out_9.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_9)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer80_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_9)
        hidden_2_mutex.release()
        os.kill(derived_model_layer80_pid, signal.SIGUSR1)
        
        x = self.layer81(x)
        out_10 = x
        out_10 = out_10.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_10)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer81_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_10)
        hidden_2_mutex.release()
        os.kill(derived_model_layer81_pid, signal.SIGUSR1)

        x = self.layer82(x)
        return x

"""""""""""""""""""""""""""""""""""""""""""""VISVM CLASS"""""""""""""""""""""""""""""""""""""""""""""
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

"""""""""""""""""""""""""""""""""""""""""""""VISVM CLASS"""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""DM CLASS"""""""""""""""""""""""""""""""""""""""""""""
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
		####################################
        # if(self.index == 0 or self.index == 3):		#pisvm에서 문제가 생긴다면 여기가 문제일 것임
        #     self.output_queue.put(self.derived_model_output)
        # else:
        #     self.output_queue.put(self.derived_model_output)
        #     self.output_queue.put(self.derived_model_output)
        self.output_queue.put(self.derived_model_output)
        
        self.mutex.acquire()
        self.checkbox = self.checker.get()
        self.checkbox[self.index] = 1
        self.checker.put(self.checkbox)
        self.mutex.release()
		#index = 0이 가장 먼저 끝날 경우 before_output_queue의 값이 없을 경우도 있음
        
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
            print(self.name, ' PI SVM input shape : ', self.pi_svm_input)
            self.pisvm_result = self.pi_svm_model.predict_cpu(self.pi_svm_input)

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

"""""""""""""""""""""""""""""""""""""""""""""DM CLASS"""""""""""""""""""""""""""""""""""""""""""""

    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    mutex = Lock()
    hidden_1_mutex = Lock()
    hidden_2_mutex = Lock()

    hidden_queue = Queue()
    hidden_queue_2 = Queue()
    d_m_checker = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)
    
    d_m_layer4_queue = Queue()
    d_m_layer7_queue = Queue()
    d_m_layer10_queue = Queue()
    d_m_layer13_queue = Queue()
    d_m_layer16_queue = Queue()
    d_m_layer19_queue = Queue()
    d_m_layer22_queue = Queue()
    d_m_layer25_queue = Queue()
    d_m_layer28_queue = Queue()
    d_m_layer31_queue = Queue()
    d_m_layer34_queue = Queue()
    d_m_layer37_queue = Queue()
    d_m_layer40_queue = Queue()
    d_m_layer43_queue = Queue()
    d_m_layer46_queue = Queue()
    d_m_layer49_queue = Queue()
    d_m_layer52_queue = Queue()
    d_m_layer55_queue = Queue()
    d_m_layer58_queue = Queue()
    d_m_layer61_queue = Queue()
    d_m_layer64_queue = Queue()
    d_m_layer67_queue = Queue()
    d_m_layer70_queue = Queue()
    d_m_layer73_queue = Queue()
    d_m_layer76_queue = Queue()
    d_m_layer79_queue = Queue()
    d_m_layer80_queue = Queue()
    d_m_layer81_queue = Queue()

    
    # vi_svm_path = ['/home/nvidia/joo/models/MobileNet/VISVM/layer76_visvm.bin', '/home/nvidia/joo/models/MobileNet/VISVM/layer79_visvm.bin', '/home/nvidia/joo/models/MobileNet/VISVM/layer80_visvm.bin', '/home/nvidia/joo/models/MobileNet/VISVM/layer81_visvm.bin']
    # pi_svm_path = ['/home/nvidia/joo/models/PI_SVM/Flatten_11_ReLU_13_PI_SVM_model.bin']
    # dm_path = ['/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_12800', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_9216', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_4608', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_4096', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_1024']
    vi_svm_path = ['/home/nvidia/joo/models/MobileNet/VISVM/layer34_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer70_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer73_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer76_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer76_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer79_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer80_visvm.bin','/home/nvidia/joo/models/MobileNet/VISVM/layer81_visvm.bin']
    pi_svm_path = ['/home/nvidia/joo/models/MobileNet/PISVM/layer4_layer7_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer7_layer10_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer10_layer13_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer13_layer16_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer16_layer19_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer19_layer22_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer22_layer25_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer25_layer28_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer28_layer31_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer31_layer34_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer34_layer37_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer37_layer40_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer40_layer43_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer43_layer46_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer46_layer49_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer49_layer52_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer52_layer55_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer55_layer58_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer58_layer61_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer61_layer64_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer64_layer67_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer67_layer70_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer70_layer73_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer73_layer76_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer76_layer79_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer79_layer80_pisvm.bin','/home/nvidia/joo/models/MobileNet/PISVM/layer80_layer81_pisvm.bin']
    dm_path = ['/home/nvidia/joo/models/MobileNet/D_M/MP/layer4_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer7_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer10_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer13_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer16_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer19_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer22_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer25_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer28_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer31_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer34_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer37_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer40_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer43_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer46_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer49_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer52_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer55_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer58_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer61_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer64_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer67_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer70_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer73_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer76_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer79_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer80_derived_model_dict.pt','/home/nvidia/joo/models/MobileNet/D_M/MP/layer81_derived_model_dict.pt']


    derived_model_layer4 = derived_model.Derived_Model4()
    derived_model_layer7 = derived_model.Derived_Model7()
    derived_model_layer10 = derived_model.Derived_Model10()
    derived_model_layer13 = derived_model.Derived_Model13()
    derived_model_layer16 = derived_model.Derived_Model16()
    derived_model_layer19 = derived_model.Derived_Model19()
    derived_model_layer22 = derived_model.Derived_Model22()
    derived_model_layer25 = derived_model.Derived_Model25()
    derived_model_layer28 = derived_model.Derived_Model28()
    derived_model_layer31 = derived_model.Derived_Model31()
    derived_model_layer34 = derived_model.Derived_Model34()
    derived_model_layer37 = derived_model.Derived_Model37()
    derived_model_layer40 = derived_model.Derived_Model40()
    derived_model_layer43 = derived_model.Derived_Model43()
    derived_model_layer46 = derived_model.Derived_Model46()
    derived_model_layer49 = derived_model.Derived_Model49()
    derived_model_layer52 = derived_model.Derived_Model52()
    derived_model_layer55 = derived_model.Derived_Model55()
    derived_model_layer58 = derived_model.Derived_Model58()
    derived_model_layer61 = derived_model.Derived_Model61()
    derived_model_layer64 = derived_model.Derived_Model64()
    derived_model_layer67 = derived_model.Derived_Model67()
    derived_model_layer70 = derived_model.Derived_Model70()
    derived_model_layer73 = derived_model.Derived_Model73()
    derived_model_layer76 = derived_model.Derived_Model76()
    derived_model_layer79 = derived_model.Derived_Model79()
    derived_model_layer80 = derived_model.Derived_Model80()
    derived_model_layer81 = derived_model.Derived_Model81()


    vi_svm_layer22_p = Visvm(hidden_queue, 'layer22', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer34_p = Visvm(hidden_queue, 'layer34', vi_svm_path[1], hidden_1_mutex)
    vi_svm_layer70_p = Visvm(hidden_queue, 'layer70', vi_svm_path[2], hidden_1_mutex)
    vi_svm_layer73_p = Visvm(hidden_queue, 'layer73', vi_svm_path[3], hidden_1_mutex)
    vi_svm_layer76_p = Visvm(hidden_queue, 'layer76', vi_svm_path[4], hidden_1_mutex)
    vi_svm_layer79_p = Visvm(hidden_queue, 'layer79', vi_svm_path[5], hidden_1_mutex)
    vi_svm_layer80_p = Visvm(hidden_queue, 'layer80', vi_svm_path[6], hidden_1_mutex)
    vi_svm_layer81_p = Visvm(hidden_queue, 'layer81', vi_svm_path[7], hidden_1_mutex)
    
    print('VISVM PROCESS START')
    vi_svm_layer22_p.start()
    vi_svm_layer34_p.start()
    vi_svm_layer70_p.start()
    vi_svm_layer73_p.start()
    vi_svm_layer76_p.start()
    vi_svm_layer79_p.start()
    vi_svm_layer80_p.start()
    vi_svm_layer81_p.start()

    vi_svm_layer22_pid = vi_svm_layer22_p.pid
    vi_svm_layer34_pid = vi_svm_layer34_p.pid
    vi_svm_layer70_pid = vi_svm_layer70_p.pid
    vi_svm_layer73_pid = vi_svm_layer73_p.pid    
    vi_svm_layer76_pid = vi_svm_layer76_p.pid
    vi_svm_layer79_pid = vi_svm_layer79_p.pid
    vi_svm_layer80_pid = vi_svm_layer80_p.pid
    vi_svm_layer81_pid = vi_svm_layer81_p.pid
    

    derived_model_layer4_p = DerivedModel('DM layer4', 0, d_m_check_queue, hidden_queue_2, derived_model_layer4, d_m_layer4_queue, d_m_layer4_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 36992)
    derived_model_layer7_p = DerivedModel('DM layer7', 1, d_m_check_queue, hidden_queue_2, derived_model_layer7, d_m_layer4_queue, d_m_layer7_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 73984)
    derived_model_layer10_p = DerivedModel('DM layer10', 2, d_m_check_queue, hidden_queue_2, derived_model_layer10, d_m_layer7_queue, d_m_layer10_queue, mutex, hidden_2_mutex, dm_path[2], pi_svm_path[1], 18496)
    derived_model_layer13_p = DerivedModel('DM layer13', 3, d_m_check_queue, hidden_queue_2, derived_model_layer13, d_m_layer10_queue, d_m_layer13_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[2], 36992)

    derived_model_layer16_p = DerivedModel('DM layer16', 4, d_m_check_queue, hidden_queue_2, derived_model_layer16, d_m_layer13_queue, d_m_layer16_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[3], 36992)
    derived_model_layer19_p = DerivedModel('DM layer19', 5, d_m_check_queue, hidden_queue_2, derived_model_layer19, d_m_layer16_queue, d_m_layer19_queue, mutex, hidden_2_mutex, dm_path[5], pi_svm_path[4], 36992)
    derived_model_layer22_p = DerivedModel('DM layer22', 6, d_m_check_queue, hidden_queue_2, derived_model_layer22, d_m_layer19_queue, d_m_layer22_queue, mutex, hidden_2_mutex, dm_path[6], pi_svm_path[5], 10368)
    derived_model_layer25_p = DerivedModel('DM layer25', 7, d_m_check_queue, hidden_queue_2, derived_model_layer25, d_m_layer22_queue, d_m_layer25_queue, mutex, hidden_2_mutex, dm_path[7], pi_svm_path[6], 20736)

    derived_model_layer28_p = DerivedModel('DM layer28', 8, d_m_check_queue, hidden_queue_2, derived_model_layer28, d_m_layer22_queue, d_m_layer28_queue, mutex, hidden_2_mutex, dm_path[8], pi_svm_path[7], 20736)
    derived_model_layer31_p = DerivedModel('DM layer31', 9, d_m_check_queue, hidden_queue_2, derived_model_layer31, d_m_layer28_queue, d_m_layer31_queue, mutex, hidden_2_mutex, dm_path[9], pi_svm_path[8], 20736)
    derived_model_layer34_p = DerivedModel('DM layer34', 10, d_m_check_queue, hidden_queue_2, derived_model_layer34, d_m_layer31_queue, d_m_layer34_queue, mutex, hidden_2_mutex, dm_path[10], pi_svm_path[9], 6400)
    derived_model_layer37_p = DerivedModel('DM layer37', 11, d_m_check_queue, hidden_queue_2, derived_model_layer37, d_m_layer34_queue, d_m_layer37_queue, mutex, hidden_2_mutex, dm_path[11], pi_svm_path[10], 12800)

    derived_model_layer40_p = DerivedModel('DM layer40', 12, d_m_check_queue, hidden_queue_2, derived_model_layer40, d_m_layer37_queue, d_m_layer40_queue, mutex, hidden_2_mutex, dm_path[12], pi_svm_path[11], 12800)
    derived_model_layer43_p = DerivedModel('DM layer43', 13, d_m_check_queue, hidden_queue_2, derived_model_layer43, d_m_layer40_queue, d_m_layer43_queue, mutex, hidden_2_mutex, dm_path[13], pi_svm_path[12], 12800)
    derived_model_layer46_p = DerivedModel('DM layer46', 14, d_m_check_queue, hidden_queue_2, derived_model_layer46, d_m_layer43_queue, d_m_layer46_queue, mutex, hidden_2_mutex, dm_path[14], pi_svm_path[13], 12800)
    derived_model_layer49_p = DerivedModel('DM layer49', 15, d_m_check_queue, hidden_queue_2, derived_model_layer49, d_m_layer46_queue, d_m_layer49_queue, mutex, hidden_2_mutex, dm_path[15], pi_svm_path[14], 12800)

    # derived_model_layer76_p = DerivedModel('DM layer76', 0, d_m_check_queue, hidden_queue_2, derived_model_layer76, d_m_layer76_queue, d_m_layer76_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    # derived_model_layer79_p = DerivedModel('DM layer79', 1, d_m_check_queue, hidden_queue_2, derived_model_layer79, d_m_layer76_queue, d_m_layer79_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    # derived_model_layer80_p = DerivedModel('DM layer80', 2, d_m_check_queue, hidden_queue_2, derived_model_layer80, d_m_layer79_queue, d_m_layer80_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)
    # derived_model_layer81_p = DerivedModel('DM layer81', 3, d_m_check_queue, hidden_queue_2, derived_model_layer81, d_m_layer80_queue, d_m_layer81_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)

    # derived_model_layer76_p = DerivedModel('DM layer76', 0, d_m_check_queue, hidden_queue_2, derived_model_layer76, d_m_layer76_queue, d_m_layer76_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    # derived_model_layer79_p = DerivedModel('DM layer79', 1, d_m_check_queue, hidden_queue_2, derived_model_layer79, d_m_layer76_queue, d_m_layer79_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    # derived_model_layer80_p = DerivedModel('DM layer80', 2, d_m_check_queue, hidden_queue_2, derived_model_layer80, d_m_layer79_queue, d_m_layer80_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)
    # derived_model_layer81_p = DerivedModel('DM layer81', 3, d_m_check_queue, hidden_queue_2, derived_model_layer81, d_m_layer80_queue, d_m_layer81_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)


    # derived_model_layer76_p = DerivedModel('DM layer76', 0, d_m_check_queue, hidden_queue_2, derived_model_layer76, d_m_layer76_queue, d_m_layer76_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    # derived_model_layer79_p = DerivedModel('DM layer79', 1, d_m_check_queue, hidden_queue_2, derived_model_layer79, d_m_layer76_queue, d_m_layer79_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    # derived_model_layer80_p = DerivedModel('DM layer80', 2, d_m_check_queue, hidden_queue_2, derived_model_layer80, d_m_layer79_queue, d_m_layer80_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)
    # derived_model_layer81_p = DerivedModel('DM layer81', 3, d_m_check_queue, hidden_queue_2, derived_model_layer81, d_m_layer80_queue, d_m_layer81_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)


    derived_model_layer76_p.start()
    derived_model_layer79_p.start()
    derived_model_layer80_p.start()
    derived_model_layer81_p.start()


    derived_model_layer76_pid = derived_model_layer76_p.pid
    derived_model_layer79_pid = derived_model_layer79_p.pid
    derived_model_layer80_pid = derived_model_layer80_p.pid
    derived_model_layer81_pid = derived_model_layer81_p.pid

    # WARM UP MODEL LOAD
    warm_model = MobileNetv1_2()
    warm_model.load_state_dict(torch.load('/home/nvidia/joo/multi_process/MobileNet/warmmobile.pt'))
    warm_model.eval()
    warm_model.to(device)
    
    model = MobileNetv1()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/MobileNet/mobilenet_dict_cifar10.pt'))
    model.eval()
    model.to(device)
    
    time.sleep(20)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)

    # WARM UP
    warm_model(test_input)
    print(' WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, hidden_queue, hidden_queue_2, vi_svm_layer76_pid, derived_model_layer76_pid, vi_svm_layer79_pid, derived_model_layer79_pid, vi_svm_layer80_pid, derived_model_layer80_pid, vi_svm_layer81_pid, derived_model_layer81_pid, hidden_1_mutex, hidden_2_mutex)
    end.record()
    torch.cuda.synchronize()
    print("MobileNet inference time : ", start.elapsed_time(end)/1000)
    	


    vi_svm_layer76_p.join()
    vi_svm_layer79_p.join()
    vi_svm_layer80_p.join()
    vi_svm_layer81_p.join()
    

    derived_model_layer76_p.join()
    derived_model_layer79_p.join()
    derived_model_layer80_p.join()
    derived_model_layer81_p.join()

    end_total.record()
    torch.cuda.synchronize()
    print("MobileNet total time : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()
    hidden_queue_2.close()
    d_m_check_queue.close()
    

    d_m_layer76_queue.close()
    d_m_layer79_queue.close()
    d_m_layer80_queue.close()
    d_m_layer81_queue.close()
