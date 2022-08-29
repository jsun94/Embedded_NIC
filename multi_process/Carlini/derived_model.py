# import torch.nn as nn

# class Derived_Model1(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model1, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(21632,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)


# class Derived_Model2(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model2, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(18432,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)


# class Derived_Model3(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model3, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(4608,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)

# class Derived_Model4(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model4, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(6400,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)
    
# class Derived_Model5(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model5, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(4096,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)
    
# class Derived_Model6(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model6, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(1024,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)
    
# class Derived_Model7(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model7, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(1024,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)
    
# class Derived_Model8(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model8, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(200,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)
    
# class Derived_Model9(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model9, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(200,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x, dim =1)

import torch.nn as nn

class Derived_Model2(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model2, self).__init__()
        self.derived = nn.Linear(57600,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model4(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model4, self).__init__()
        self.derived = nn.Linear(50176,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model5(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model5, self).__init__()
        self.derived = nn.Linear(12544,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model7(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model7, self).__init__()
        self.derived = nn.Linear(18432,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x

class Derived_Model9(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model9, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model10(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model10, self).__init__()
        self.derived = nn.Linear(3200,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model11(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model11, self).__init__()
        self.derived = nn.Linear(3200,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model13(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model13, self).__init__()
        self.derived = nn.Linear(256,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model15(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model15, self).__init__()
        self.derived = nn.Linear(256,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x