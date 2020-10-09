import numpy as np
import torch
from torch.nn import functional as F
import pickle
from copy import deepcopy as copy
from IPython.display import display, clear_output
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def set_num_threads(n):
#     torch.set_num_threads(n)
    
####################################
## Models
####################################


class _FCNN(torch.nn.Module):
    def __init__(self, nodes, activation, dropout_p=0.0):
        super(_FCNN,self).__init__()

        self.nodes = nodes
        self.activation = activation
        self.dropout_p = dropout_p
        
        self.seq = []
        for i in range(len(self.nodes)-2):
            if self.dropout_p > 0.0:
                self.seq = self.seq + [torch.nn.Linear(self.nodes[i],self.nodes[i+1]),torch.nn.Dropout(self.dropout_p), self.activation]
            else:
                self.seq = self.seq + [torch.nn.Linear(self.nodes[i],self.nodes[i+1]),                                  self.activation]
        self.seq = self.seq + [torch.nn.Linear(self.nodes[-2],self.nodes[-1])]

        self.nn = torch.nn.Sequential(*self.seq)

    def forward(self, x):
        return self.nn(x)
    

def FCNN(nodes, activation=torch.nn.ReLU(), dropout_p=0.0):
    model = _FCNN(nodes=nodes,activation=activation,dropout_p=dropout_p).to(device)
    return model



class FCNN_IdentityBlock(torch.nn.Module):
    def __init__(self, inout_feature, nodes, activation, dropout_p=0.0, trainable=False, initZeros=True):
        super(FCNN_IdentityBlock, self).__init__()
        
        self.nodes = nodes
        self.activation = activation
        self.dropout_p = dropout_p
        self.trainable = trainable
        self.initZeros = initZeros
        
        if self.dropout_p > 0.0:
            self.seq = [torch.nn.Linear(inout_feature,self.nodes[0]),torch.nn.Dropout(self.dropout_p), self.activation]
        else:
            self.seq = [torch.nn.Linear(inout_feature,self.nodes[0])                                 , self.activation]
        
        
        for i in range(len(self.nodes)-1):
            if self.dropout_p > 0.0:
                self.seq = self.seq + [torch.nn.Linear(self.nodes[i],self.nodes[i+1]),torch.nn.Dropout(self.dropout_p), self.activation]
            else:
                self.seq = self.seq + [torch.nn.Linear(self.nodes[i],self.nodes[i+1]),                                  self.activation]
        self.seq = self.seq + [torch.nn.Linear(self.nodes[-1],inout_feature)]

        self.IdentityBlock = torch.nn.Sequential(*self.seq)
        if initZeros:
            for p in self.IdentityBlock.parameters():
                p.data.fill_(0)
        if not trainable:
            for p in self.IdentityBlock.parameters():
                p.requires_grad  = False

    def forward(self, x):
        y  = self.IdentityBlock(x)
        y += x
        return y
    
    
class Linear_wResidualBlock(torch.nn.Module):
    def __init__(self, nodes, activation, dropout_p=0.0, trainable=False, initZeros=True):
        super(Linear_wResidualBlock, self).__init__()
        
        self.seq = []
        for i in range(len(self.nodes)-2):
            if dropout_p > 0.0:
                self.seq = self.seq + [torch.nn.Linear(nodes[i],nodes[i+1]),torch.nn.Dropout(dropout_p), activation]
            else:
                self.seq = self.seq + [torch.nn.Linear(nodes[i],nodes[i+1]),                             activation]
        self.seq = self.seq + [torch.nn.Linear(nodes[-2],nodes[-1]),activation]
        self.ResidualBlock = torch.nn.Sequential(*self.seq)
        if initZeros:
            for p in self.ResidualBlock.parameters():
                p.data.fill_(0)
        if not trainable:
            for p in self.ResidualBlock.parameters():
                p.requires_grad  = False
        
        if dropout_p > 0.0:
            self.nn = torch.nn.Sequential([torch.nn.Linear(nodes[0],nodes[-1]),torch.nn.Dropout(dropout_p), activation])
        else:
            self.nn = torch.nn.Sequential([torch.nn.Linear(nodes[0],nodes[-1]),                             activation])

    def forward(self, x):
        y  = self.nn(x)
        y += self.ResidualBlock(x)
        return y
    
    
class _resFCNN(torch.nn.Module):
    def __init__(self, nodes, activation, dropout_p=0.0, res_trainable=False, res_initZeros=True, identity_block_every_layer=True):
        super(_resFCNN, self).__init__()
        
        
        self.layers = []
        for i in range(len(nodes)-2):
            temp_nodes = [nodes[i],min(nodes[i],nodes[i+1]),min(nodes[i],nodes[i+1]),nodes[i+1]]
            self.layers.append(Linear_wResidualBlock(temp_nodes,activation,dropout_p,res_trainable,res_initZeros))
            if identity_block_every_layer:
                temp_nodes = [nodes[i+1],nodes[i+1]]
                self.layers.append(FCNN_IdentityBlock(temp_nodes,activation,dropout_p,res_trainable,res_initZeros))
               
        self.layers.append(torch.nn.Linear(nodes[-2],nodes[-1]))
                

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
def resFCNN(nodes, activation=torch.nn.ReLU(), dropout_p=0.0, res_trainable=False, res_initZeros=True, identity_block_every_layer=True):
    model = _resFCNN(nodes,activation,dropout_p,res_trainable,res_initZeros).to(device)
    return model

####################################
## Loss
####################################
_Loss = torch.nn.modules.loss._Loss

def relative_MSE(input, target, epsilon = 1.0e-6, size_average=None, reduce=None, reduction='sum'):
    loss = (input - target) ** 2 /(input**2+epsilon)
    return torch.sum(loss)


class RMSELoss(_Loss):
    def __init__(self, epsilon = 1.0e-6, size_average=None, reduce=None, reduction='sum'):
        super(RMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return relative_MSE(input, target, reduction=self.reduction)
    
    
def get_val_loss_supervised(model,val_data_loader,criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, outputs in val_data_loader:
            inputs = inputs.to(device)
            model_outputs = model(inputs)
            loss = criterion(outputs.to(device), model_outputs)
            val_loss += loss.item()
    val_loss /= len(val_data_loader)
    return val_loss
    
####################################
## Train
####################################      
def train_supervised(model,lr,epochs,
                     train_data_loader,test_data_loader=None,val_data_loader=None,
                     criterion=torch.nn.MSELoss(),
                     optimizer=torch.optim.Adam,
                     fname = None,
                     old_hist = None,
                     old_best_loss = None,
                     dispHead = 10,
                     dispTail = 10,
                     flagEvalMode = False,
                     args = None):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(filter(lambda p: p.requires_grad, net.parameters())),lr=lr)
    
    if old_hist == None:
        old_hist ={'train_loss':[],'test_loss' :[]}

    old_epochs = len(old_hist['train_loss'])
    hist = {'train_loss':np.zeros(old_epochs+epochs)}
           
    hist['train_loss'][:old_epochs] = old_hist['train_loss'][:]
    if 'test_loss' in old_hist:
        hist['test_loss'] = np.zeros(old_epochs+epochs)
        hist['test_loss'][:old_epochs] = old_hist['test_loss' ][:]
    
    for epoch in range(epochs):
        if flagEvalMode:
            model.eval()
        else:
            model.train()
        train_loss = 0
        for inputs, outputs in train_data_loader:
#             print('inputs.shape, outputs.shape=',inputs.shape, outputs.shape)
            opt.zero_grad()
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            model_outputs = model(inputs)
            loss = criterion(outputs, model_outputs)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_data_loader)
        hist['train_loss'][old_epochs+epoch] = train_loss

        
        if test_data_loader == None:
            test_loss = train_loss
        else:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for inputs, outputs in test_data_loader:
                    inputs = inputs.to(device)
                    outputs = outputs.to(device)
                    model_outputs = model(inputs)
                    loss = criterion(outputs, model_outputs)
                    test_loss += loss.item()
            test_loss /= len(test_data_loader)
            hist['test_loss' ][old_epochs+epoch] = test_loss

            
            
            
        flag_best = False
        if old_best_loss != None:
            if test_loss < old_best_loss:
                flag_best = True
                old_best_loss = test_loss
                checkpoint = {'epoch':old_epochs+epoch, 
                              'model':model,
                              'model_state_dict':model.state_dict(), 
                              'optimizer':opt,
                              'optimizer_state_dict':opt.state_dict()}
                if fname!=None:
                    torch.save(checkpoint, fname + '_best.checkpoint')
                else:
                    torch.save(checkpoint, '_best.checkpoint')

        
        # display the epoch training loss
        if epoch < dispHead  or epoch >= epochs -dispTail:
            end = '\n'
        else:
            end = '\r'
        if test_data_loader != None:
            print("epoch : {}/{}, train loss = {:.6f}, test loss = {:.6f}".format(
                  old_epochs +epoch, old_epochs +epochs, 
                  hist['train_loss'][old_epochs +epoch], 
                  hist['test_loss'][old_epochs +epoch]), end=end)
        else:
            print("epoch : {}/{}, train loss = {:.6f}".format(
                  old_epochs +epoch, old_epochs +epochs, 
                  hist['train_loss'][old_epochs +epoch]), end=end)

                
                
    checkpoint = {'epoch':old_epochs+epoch, 
                  'model':model,
                  'model_state_dict':model.state_dict(), 
                  'optimizer':opt,
                  'optimizer_state_dict':opt.state_dict()}
    if fname!=None:
        torch.save(checkpoint, fname + '_trainingEnd.checkpoint')
            
             
            
    if old_best_loss !=None and flag_best:
        if fname!=None:
            checkpoint = torch.load(fname + '_best.checkpoint')
        else:
            checkpoint = torch.load('_best.checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'])

    if val_data_loader != None:
        val_loss = get_val_loss_supervised(model,val_data_loader,criterion)
        print('validation_loss:',val_loss)
        
    return model,hist
    
        

###################################################
## Utils
###################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

       

def get_derivatives_scalar_input(n,func,x,create_graph=False):
    '''
    inputs
      n: order of derivatives
      func(x): scalar function with a scalar input x
    output
      df: represent df/dx
    '''
    xtmp = torch.Tensor([x])
    xtmp.requires_grad = True
    f = func(xtmp)
    df = [f]
    for i in range(n):
        df.append(torch.autograd.grad(df[-1], xtmp, create_graph=True)[0])
    if create_graph:
        return df
    else:
        return np.array([item.item() for item in df])
    
    

def get_derivatives_vector_input(n,func,x,create_graph=False):
    '''
    inputs
      n: order of derivatives (1<=n<=4)
      func(x): scalar function with a vector input x
    output
      df: list of matrices of derivatives (ex) [Jacobian (n), Hessian (n x n), 3rd derivatives (n x n x n)...]
    '''
    
    dim_x = len(x)
    xtmp = torch.Tensor(x)
    xtmp.requires_grad = True
    f = func(xtmp)
    df = torch.autograd.grad(f, xtmp, create_graph=True)[0]
    if n==1:
        if create_graph:
            return df
        else:
            return df.detach().numpy()
    
    ddf = []
    for d in range(dim_x):
        ddf.append(torch.autograd.grad(df[d], xtmp, create_graph=True)[0])
    ddf = torch.cat(ddf).reshape((dim_x,dim_x))
    if n==2:
        if create_graph:
            return df,ddf
        else:
            return df.detach().numpy(), ddf.detach().numpy()

    
    dddf = [0]*dim_x
    for d1 in range(dim_x):
        dddf[d1] = [0]*dim_x
        for d2 in range(dim_x):
            dddf[d1][d2] = torch.autograd.grad(ddf[d1,d2], xtmp, create_graph=True)[0]
        dddf[d1] = torch.cat(dddf[d1]).reshape((dim_x,dim_x))
    dddf = torch.cat(dddf).reshape((dim_x,dim_x,dim_x))
    if n==3:
        if create_graph:
            return df,ddf,dddf
        else:
            return df.detach().numpy(), ddf.detach().numpy(), dddf.detach().numpy()
        
    
    ddddf = [0]*dim_x
    for d1 in range(dim_x):
        ddddf[d1] = [0]*dim_x
        for d2 in range(dim_x):
            ddddf[d1][d2] = [0]*dim_x
            for d3 in range(dim_x):
                ddddf[d1][d2][d3] = torch.autograd.grad(dddf[d1,d2,d3], xtmp, create_graph=True)[0]
            ddddf[d1][d2] = torch.cat(ddddf[d1][d2]).reshape((dim_x,dim_x))
        ddddf[d1] = torch.cat(ddddf[d1]).reshape((dim_x,dim_x,dim_x))
    ddddf = torch.cat(ddddf).reshape((dim_x,dim_x,dim_x,dim_x))
    
        
    if create_graph:
        return df,ddf,dddf,ddddf
    else:
        return df.detach().numpy(), ddf.detach().numpy(), dddf.detach().numpy(), ddddf.detach().numpy()
    
    
    