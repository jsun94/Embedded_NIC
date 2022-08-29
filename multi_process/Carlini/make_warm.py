import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# class CarliniModel2(nn.Module):
#     def __init__(self):
#         super(CarliniModel2, self).__init__()
#         self.layer1 = torch.nn.Conv2d(1,32,(3,3))
#         self.layer2 = torch.nn.ReLU()
#         self.layer3 = torch.nn.Conv2d(32,32,(3,3))
#         self.layer4 = torch.nn.ReLU()
#         self.layer5 = torch.nn.MaxPool2d(kernel_size=2)

#         self.layer6 = torch.nn.Conv2d(32,64,(3,3))
#         self.layer7 = torch.nn.ReLU()
#         self.layer8 = torch.nn.Conv2d(64,64,(3,3))
#         self.layer9 = torch.nn.ReLU()
#         self.layer10 = torch.nn.MaxPool2d(kernel_size=2)

#         self.layer11 = torch.nn.Flatten()
#         self.layer12 = torch.nn.Linear(1024,200)
#         self.layer13 = torch.nn.ReLU()
#         self.layer14 = torch.nn.Linear(200,200)
#         self.layer15 = torch.nn.ReLU()
#         self.layer16 = torch.nn.Linear(200,10)
#         #self.layer17 = torch.nn.Softmax() ###


#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)	#
#         out = self.layer3(out)       
#         out = self.layer4(out)   # 
#         out = self.layer5(out)   #
#         out = self.layer6(out)   
#         out = self.layer7(out)	 #
#         out = self.layer8(out)       
#         out = self.layer9(out)   #
#         out = self.layer10(out)  #
#         out = self.layer11(out)  #
#         out = self.layer12(out)
#         out = self.layer13(out)  #
#         out = self.layer14(out)    
#         out = self.layer15(out)  #
#         out = self.layer16(out)

#         return out

# model = CarliniModel2()
# model.eval()
# model.to(device)

# torch.save(model.state_dict(), './warmcarlini.pt')

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

model = CarliniModel2()
model.eval()
model.to(device)

torch.save(model.state_dict(), './warmcarlini_cifar10.pt')