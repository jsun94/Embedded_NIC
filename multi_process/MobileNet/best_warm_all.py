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

    def forward(self, x, hidden_queue, vi_svm_layer10_pid, hidden_1_mutex):
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
        out_10 = x
        out_10 = out_10.detach().cpu().numpy()
        hidden_1_mutex.acquire()
        hidden_queue.put(out_10)
        hidden_1_mutex.release()
        os.kill(vi_svm_layer10_pid, signal.SIGUSR1)

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



    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    mutex = Lock()
    hidden_1_mutex = Lock()


    hidden_queue = Queue()



    
    # vi_svm_path = ['/home/nvidia/joo/models/MobileNet/VISVM/layer76_visvm.bin', '/home/nvidia/joo/models/MobileNet/VISVM/layer79_visvm.bin', '/home/nvidia/joo/models/MobileNet/VISVM/layer80_visvm.bin', '/home/nvidia/joo/models/MobileNet/VISVM/layer81_visvm.bin']
    # pi_svm_path = ['/home/nvidia/joo/models/PI_SVM/Flatten_11_ReLU_13_PI_SVM_model.bin']
    # dm_path = ['/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_12800', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_9216', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_4608', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_4096', '/home/nvidia/joo/models/MobileNet/D_M/mobile_dict_1024']
    vi_svm_path = ['/home/nvidia/joo/models/MobileNet/VISVM/layer10_visvm.bin']





    vi_svm_layer10_p = Visvm(hidden_queue, 'layer10', vi_svm_path[0], hidden_1_mutex)

    
    print('VISVM PROCESS START')
    vi_svm_layer10_p.start()


    vi_svm_layer10_pid = vi_svm_layer10_p.pid

    








    # WARM UP MODEL LOAD
    warm_model = MobileNetv1_2()
    warm_model.load_state_dict(torch.load('/home/nvidia/joo/multi_process/MobileNet/warmmobile.pt'))
    warm_model.eval()
    warm_model.to(device)
    
    model = MobileNetv1()
    model.load_state_dict(torch.load('/home/nvidia/joo/models/MobileNet/mobilenet_dict_cifar10.pt'))
    model.eval()
    model.to(device)
    
    time.sleep(10)
    
    test_input = torch.ones(1,3,32,32)
    test_input = test_input.to(device)

    # WARM UP
    warm_model(test_input)
    print(' WARM DONE')

    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)
    end_total = torch.cuda.Event(enable_timing = True)
    start.record()
    result = model(test_input, hidden_queue, vi_svm_layer10_pid, hidden_1_mutex)
    end.record()
    torch.cuda.synchronize()
    print("MobileNet inference time : ", start.elapsed_time(end)/1000)
    	


    vi_svm_layer10_p.join()

    



    end_total.record()
    torch.cuda.synchronize()
    print("MobileNet total time : ", start.elapsed_time(end_total)/1000)

    hidden_queue.close()

    


