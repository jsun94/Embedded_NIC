# import torch.nn as nn

# class Derived_Model1(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model1, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(12800,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model2(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model2, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(9216,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model3(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model3, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(4608,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model4(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model4, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(4096,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)
    
# class Derived_Model5(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
#     def __init__(self):
#         super(Derived_Model5, self).__init__()
#         # self.derived = nn.Sequential()
#         self.derived = nn.Linear(1024,10)
        
#     def forward(self,x):
#         x = self.derived(x)
#         return nn.functional.softmax(x,dim = 1)

import torch.nn as nn

class Derived_Model4(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model4, self).__init__()
        self.derived = nn.Linear(36992,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model7(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model7, self).__init__()
        self.derived = nn.Linear(73984,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model10(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model10, self).__init__()
        self.derived = nn.Linear(18496,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model13(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model13, self).__init__()
        self.derived = nn.Linear(36992,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model16(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model16, self).__init__()
        self.derived = nn.Linear(36992,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model19(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model19, self).__init__()
        self.derived = nn.Linear(36992,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model22(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model22, self).__init__()
        self.derived = nn.Linear(10368,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model25(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model25, self).__init__()
        self.derived = nn.Linear(20736,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model28(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model28, self).__init__()
        self.derived = nn.Linear(20736,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model31(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model31, self).__init__()
        self.derived = nn.Linear(20736,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model34(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model34, self).__init__()
        self.derived = nn.Linear(6400,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model37(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model37, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model40(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model40, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model43(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model43, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model46(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model46, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model49(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model49, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model52(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model52, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model55(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model55, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model58(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model58, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model61(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model61, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model64(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model64, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model67(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model67, self).__init__()
        self.derived = nn.Linear(12800,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model70(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model70, self).__init__()
        self.derived = nn.Linear(4608,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model73(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model73, self).__init__()
        self.derived = nn.Linear(9216,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model76(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model76, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model79(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model79, self).__init__()
        self.derived = nn.Linear(4096,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model80(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model80, self).__init__()
        self.derived = nn.Linear(1024,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x
    
class Derived_Model81(nn.Module):	#멀티 스레드용 -> 모델 전체 저장하므로
    def __init__(self):
        super(Derived_Model81, self).__init__()
        self.derived = nn.Linear(1024,10)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x):
        x = self.derived(x)
        x = self.softmax(x)
        return x