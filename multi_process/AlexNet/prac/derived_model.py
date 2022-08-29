# import torch.nn as nn

# class Derived_Model1(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model1, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(16384,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model2(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model2, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(4096,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model3(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model3, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(12288,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model4(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model4, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(3072,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model5(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model5, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(6144,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model6(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model6, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(1024,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)


import torch.nn as nn

class Derived_Model2(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model2, self).__init__()
        self.derived = nn.Linear(16384,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model3(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model3, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model5(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model5, self).__init__()
        self.derived = nn.Linear(12288,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model6(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model6, self).__init__()
        self.derived = nn.Linear(3072,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model8(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model8, self).__init__()
        self.derived = nn.Linear(6144,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model10(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model10, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model12(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model12, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model13(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model13, self).__init__()
        self.derived = nn.Linear(1024,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model14(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model14, self).__init__()
        self.derived = nn.Linear(1024,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model17(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model17, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model20(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model20, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x