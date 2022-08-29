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

    def forward(self, x, hidden_queue, hidden_queue_2, vi_svm_layer58_pid, derived_model_layer58_pid, vi_svm_layer61_pid, derived_model_layer61_pid, vi_svm_layer64_pid, derived_model_layer64_pid, vi_svm_layer67_pid, derived_model_layer67_pid, vi_svm_layer70_pid, derived_model_layer70_pid, vi_svm_layer73_pid, derived_model_layer73_pid, vi_svm_layer76_pid, derived_model_layer76_pid, vi_svm_layer79_pid, derived_model_layer79_pid, vi_svm_layer80_pid, derived_model_layer80_pid, hidden_1_mutex, hidden_2_mutex):
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
        out_1 = x
        out_1 = out_1.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_1)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer58_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_1)
        hidden_2_mutex.release()
        os.kill(derived_model_layer58_pid, signal.SIGUSR1)
        
        x = self.layer59(x)
        x = self.layer60(x)
        
        x = self.layer61(x)
        out_2 = x
        out_2 = out_2.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_2)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer61_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_2)
        hidden_2_mutex.release()
        os.kill(derived_model_layer61_pid, signal.SIGUSR1)
        
        x = self.layer62(x)
        x = self.layer63(x)
        x = self.layer64(x)
        out_3 = x
        out_3 = out_3.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_3)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer64_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_3)
        hidden_2_mutex.release()
        os.kill(derived_model_layer64_pid, signal.SIGUSR1)
        
        x = self.layer65(x)
        x = self.layer66(x)
        x = self.layer67(x)
        out_4 = x
        out_4 = out_4.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_4)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer67_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_4)
        hidden_2_mutex.release()
        os.kill(derived_model_layer67_pid, signal.SIGUSR1)
        
        x = self.layer68(x)
        x = self.layer69(x)
        x = self.layer70(x)
        out_5 = x
        out_5 = out_5.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_5)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer70_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_5)
        hidden_2_mutex.release()
        os.kill(derived_model_layer70_pid, signal.SIGUSR1)
        
        x = self.layer71(x)
        x = self.layer72(x)
        x = self.layer73(x)
        out_6 = x
        out_6 = out_6.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_6)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer73_pid, signal.SIGUSR1)
        hidden_2_mutex.acquire()
        hidden_queue_2.put(out_6)
        hidden_2_mutex.release()
        os.kill(derived_model_layer73_pid, signal.SIGUSR1)
        
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
            self.corrent_index_output = self.output_queue.get()
            self.pi_svm_input = np.concatenate((self.before_index_output, self.corrent_index_output), axis = 1)
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
    d_m_checker = np.array([0,0,0,0,0,0,0,0,0])
    d_m_check_queue = Queue()
    d_m_check_queue.put(d_m_checker)
    
    d_m_layer58_queue = Queue()
    d_m_layer61_queue = Queue()
    d_m_layer64_queue = Queue()
    d_m_layer67_queue = Queue()
    d_m_layer70_queue = Queue()
    d_m_layer73_queue = Queue()
    d_m_layer76_queue = Queue()
    d_m_layer79_queue = Queue()
    d_m_layer80_queue = Queue()

    
    vi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
    pi_svm_path = ['/home/nvidia/joo/models/VI_SVM/ReLU_13_VI_SVM_model.bin']
    dm_path = ['/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_12800', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_9216', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_4608', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_4096', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_1024']

    derived_model_layer58 = derived_model.Derived_Model1()
    derived_model_layer61 = derived_model.Derived_Model1()
    derived_model_layer64 = derived_model.Derived_Model1()
    derived_model_layer67 = derived_model.Derived_Model1()
    derived_model_layer70 = derived_model.Derived_Model3()
    derived_model_layer73 = derived_model.Derived_Model2()
    derived_model_layer76 = derived_model.Derived_Model4()
    derived_model_layer79 = derived_model.Derived_Model4()
    derived_model_layer80 = derived_model.Derived_Model5()


    
    vi_svm_layer58_p = Visvm(hidden_queue, 'layer58', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer61_p = Visvm(hidden_queue, 'layer61', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer64_p = Visvm(hidden_queue, 'layer64', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer67_p = Visvm(hidden_queue, 'layer67', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer70_p = Visvm(hidden_queue, 'layer70', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer73_p = Visvm(hidden_queue, 'layer73', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer76_p = Visvm(hidden_queue, 'layer76', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer79_p = Visvm(hidden_queue, 'layer79', vi_svm_path[0], hidden_1_mutex)
    vi_svm_layer80_p = Visvm(hidden_queue, 'layer80', vi_svm_path[0], hidden_1_mutex)

    
    print('VISVM PROCESS START')
    vi_svm_layer58_p.start()
    vi_svm_layer61_p.start()
    vi_svm_layer64_p.start()
    vi_svm_layer67_p.start()
    vi_svm_layer70_p.start()
    vi_svm_layer73_p.start()
    vi_svm_layer76_p.start()
    vi_svm_layer79_p.start()
    vi_svm_layer80_p.start()

    
    vi_svm_layer58_pid = vi_svm_layer58_p.pid
    vi_svm_layer61_pid = vi_svm_layer61_p.pid
    vi_svm_layer64_pid = vi_svm_layer64_p.pid
    vi_svm_layer67_pid = vi_svm_layer67_p.pid
    vi_svm_layer70_pid = vi_svm_layer70_p.pid
    vi_svm_layer73_pid = vi_svm_layer73_p.pid
    vi_svm_layer76_pid = vi_svm_layer76_p.pid
    vi_svm_layer79_pid = vi_svm_layer79_p.pid
    vi_svm_layer80_pid = vi_svm_layer80_p.pid
    

    derived_model_layer58_p = DerivedModel('DM layer58', 0, d_m_check_queue, hidden_queue_2, derived_model_layer58, d_m_layer58_queue, d_m_layer58_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 12800)
    derived_model_layer61_p = DerivedModel('DM layer61', 1, d_m_check_queue, hidden_queue_2, derived_model_layer61, d_m_layer58_queue, d_m_layer61_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 12800)
    derived_model_layer64_p = DerivedModel('DM layer64', 2, d_m_check_queue, hidden_queue_2, derived_model_layer64, d_m_layer61_queue, d_m_layer64_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 12800)
    derived_model_layer67_p = DerivedModel('DM layer67', 3, d_m_check_queue, hidden_queue_2, derived_model_layer67, d_m_layer64_queue, d_m_layer67_queue, mutex, hidden_2_mutex, dm_path[0], pi_svm_path[0], 12800)
    derived_model_layer70_p = DerivedModel('DM layer70', 4, d_m_check_queue, hidden_queue_2, derived_model_layer70, d_m_layer67_queue, d_m_layer70_queue, mutex, hidden_2_mutex, dm_path[2], pi_svm_path[0], 4608)
    derived_model_layer73_p = DerivedModel('DM layer73', 5, d_m_check_queue, hidden_queue_2, derived_model_layer73, d_m_layer70_queue, d_m_layer73_queue, mutex, hidden_2_mutex, dm_path[1], pi_svm_path[0], 9216)
    derived_model_layer76_p = DerivedModel('DM layer76', 6, d_m_check_queue, hidden_queue_2, derived_model_layer76, d_m_layer73_queue, d_m_layer76_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    derived_model_layer79_p = DerivedModel('DM layer79', 7, d_m_check_queue, hidden_queue_2, derived_model_layer79, d_m_layer76_queue, d_m_layer79_queue, mutex, hidden_2_mutex, dm_path[3], pi_svm_path[0], 4096)
    derived_model_layer80_p = DerivedModel('DM layer80', 8, d_m_check_queue, hidden_queue_2, derived_model_layer80, d_m_layer79_queue, d_m_layer80_queue, mutex, hidden_2_mutex, dm_path[4], pi_svm_path[0], 1024)

    derived_model_layer58_p.start()
    derived_model_layer61_p.start()
    derived_model_layer64_p.start()
    derived_model_layer67_p.start()
    derived_model_layer70_p.start()
    derived_model_layer73_p.start()
    derived_model_layer76_p.start()
    derived_model_layer79_p.start()
    derived_model_layer80_p.start()
    
    derived_model_layer58_pid = derived_model_layer58_p.pid
    derived_model_layer61_pid = derived_model_layer61_p.pid
    derived_model_layer64_pid = derived_model_layer64_p.pid
    derived_model_layer67_pid = derived_model_layer67_p.pid
    derived_model_layer70_pid = derived_model_layer70_p.pid
    derived_model_layer73_pid = derived_model_layer73_p.pid
    derived_model_layer76_pid = derived_model_layer76_p.pid
    derived_model_layer79_pid = derived_model_layer79_p.pid
    derived_model_layer80_pid = derived_model_layer80_p.pid

    # WARM UP MODEL LOAD
    warm_model = MobileNetv1_2()
    warm_model.load_state_dict(torch.load('/home/nvidia/joo/multi_process/MobileNet/warmmobile.pt'))
    warm_model.eval()
    warm_model.to(device)
    
    model = MobileNetv1()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/MobileNet/mobilenet_dict_cifar10.pt'))
    model.eval()
    model.to(device)
    
    time.sleep(30)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)

    # WARM UP
    warm_model(test_input)
    print(' WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, hidden_queue, hidden_queue_2, vi_svm_layer58_pid, derived_model_layer58_pid, vi_svm_layer61_pid, derived_model_layer61_pid, vi_svm_layer64_pid, derived_model_layer64_pid, vi_svm_layer67_pid, derived_model_layer67_pid, vi_svm_layer70_pid, derived_model_layer70_pid, vi_svm_layer73_pid, derived_model_layer73_pid, vi_svm_layer76_pid, derived_model_layer76_pid, vi_svm_layer79_pid, derived_model_layer79_pid, vi_svm_layer80_pid, derived_model_layer80_pid, hidden_1_mutex, hidden_2_mutex)
    end.record()
    torch.cuda.synchronize()
    print("MobileNet inference time : ", start.elapsed_time(end)/1000)
    	

    vi_svm_layer58_p.join()
    vi_svm_layer61_p.join()
    vi_svm_layer64_p.join()
    vi_svm_layer67_p.join()
    vi_svm_layer70_p.join()
    vi_svm_layer73_p.join()
    vi_svm_layer76_p.join()
    vi_svm_layer79_p.join()
    vi_svm_layer80_p.join()
    
    derived_model_layer58_p.join()
    derived_model_layer61_p.join()
    derived_model_layer64_p.join()
    derived_model_layer67_p.join()
    derived_model_layer70_p.join()
    derived_model_layer73_p.join()
    derived_model_layer76_p.join()
    derived_model_layer79_p.join()
    derived_model_layer80_p.join()
    
    end_total.record()
    torch.cuda.synchronize()
    print("MobileNet total time : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()
    hidden_queue_2.close()
    d_m_check_queue.close()
    
    d_m_layer58_queue.close()
    d_m_layer61_queue.close()
    d_m_layer64_queue.close()
    d_m_layer67_queue.close()
    d_m_layer70_queue.close()
    d_m_layer73_queue.close()
    d_m_layer76_queue.close()
    d_m_layer79_queue.close()
    d_m_layer80_queue.close()