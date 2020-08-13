import numpy as np
import torch
from torch.nn import functional as F
import pickle
from copy import deepcopy as copy
  
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
            if dropout_p > 0.0:
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







class _resFCNN_BasicBlock(torch.nn.Module):
    def __init__(self, n_nodes, n_layers, activation, dropout_p=0.0):
        super(_resFCNN_BasicBlock, self).__init__()
        
        self.node = n_nodes
        self.activation = activation
        self.dropout_p = dropout_p
        
        self.seq = []
        for i in range(n_layers):
          self.seq = self.seq + [torch.nn.Linear(n_nodes,n_nodes), torch.nn.Dropout(self.dropout_p), self.activation]
        self.resNN = torch.nn.Sequential(*self.seq)
        self.name = 'resFCNN_BasicBlock'

    def forward(self, x):
        y  = self.resNN(x)
        y += x
        return y
    
    
class _resFCNN_BottleNeckBlock(torch.nn.Module):
    def __init__(self, nodes, n_layers, activation, dropout_p=0.0):
        super(_resFCNN_BottleNeckBlock, self).__init__()
        
        self.nodes = nodes
        self.n_layers = n_layers
        self.activation = activation
        self.dropout_p = dropout_p
        
        d_node = int((nodes[1] - nodes[0])/n_layers)
        self.seq = []
        for i in range(n_layers-1):
            self.seq = self.seq + [torch.nn.Linear(nodes[0]+i*d_node,nodes[0]+(i+1)*d_node), 
                                   torch.nn.Dropout(dropout_p), 
                                   self.activation]
        self.seq = self.seq + [torch.nn.Linear(nodes[0]+(n_layers-1)*d_node,nodes[1]), 
                               torch.nn.Dropout(dropout_p), 
                               self.activation]
        
        self.resNN = torch.nn.Sequential(*self.seq)
        self.directNN = torch.nn.Sequential(torch.nn.Linear(nodes[0],nodes[1]), 
                                            torch.nn.Dropout(dropout_p), 
                                            self.activation)
        self.name = 'resFCNN_BottleNeckBlock'

    def forward(self, x):
        y  = self.resNN(x)
        y += self.directNN(x)
        return y
    
    
class _resFCNN(torch.nn.Module):
    def __init__(self, nodes, layers, activation, dropout_p=0.0):
        super(_resFCNN, self).__init__()
        
        self.nodes = nodes
        self.layers = layers
        self.activation = activation
        self.dropout_p = dropout_p
        assert len(nodes) == len(layers)+1
        
        self.seq = []
        for i in range(len(layers)):
            if layers[i]>0:
                if nodes[i][0]==nodes[i][1]:
                    self.seq = self.seq + [_resFCNN_BasicBlock(nodes[i][0], layers[i], activation, dropout_p)]
                else:
                    self.seq = self.seq + [_resFCNN_BottleNeckBlock(nodes[i], layers[i], activation, dropout_p)]
            else:
                self.seq = self.seq + [torch.nn.Linear(self.nodes[i][0],self.nodes[i][1]), activation, torch.nn.Dropout(dropout_p)]

        self.seq = self.seq + [torch.nn.Linear(self.nodes[-1][0],self.nodes[-1][1])]
                
        self.nn = torch.nn.Sequential(*self.seq)
        self.name = 'resFCNN'

    def forward(self, x):
        return self.nn(x)
    
def resFCNN(nodes, layers, activation=torch.nn.ReLU(), dropout_p=0.0):
    model = _resFCNN(nodes,layers,activation,dropout_p).to(device)
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
                     old_best_loss = 1e15,
                     get_best_model = False,
                     disp = True,
                     args = None):

    opt = torch.optim.Adam(model.parameters(),lr=lr)
    
    if old_hist == None:
        old_hist ={'train_loss':[],'test_loss' :[]}

    old_epochs = len(old_hist['train_loss'])
    hist = {'train_loss':np.zeros(old_epochs+epochs)}
           
    hist['train_loss'][:old_epochs] = old_hist['train_loss'][:]
    if 'test_loss' in old_hist:
        hist['test_loss'] = np.zeros(old_epochs+epochs)
        hist['test_loss'][:old_epochs] = old_hist['test_loss' ][:]
    
      
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, outputs in train_data_loader:
#             print('inputs.shape, outputs.shape=',inputs.shape, outputs.shape)
            opt.zero_grad()
            inputs = inputs.to(device)
            model_outputs = model(inputs)
            loss = criterion(outputs.to(device), model_outputs)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_data_loader)
        hist['train_loss'][old_epochs+epoch] = train_loss

        if test_data_loader != None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for inputs, outputs in test_data_loader:
                    inputs = inputs.to(device)
                    model_outputs = model(inputs)
                    loss = criterion(outputs.to(device), model_outputs)
                    test_loss += loss.item()
            test_loss /= len(test_data_loader)
            hist['test_loss' ][old_epochs+epoch] = test_loss

            if test_loss < old_best_loss:
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
        if disp:
            if test_data_loader != None:
                print("epoch : {}/{}, train loss = {:.6f}, test loss = {:.6f}".format(
                      old_epochs +epoch, old_epochs +epochs, 
                      hist['train_loss'][old_epochs +epoch], 
                      hist['test_loss'][old_epochs +epoch]))
            else:
                print("epoch : {}/{}, train loss = {:.6f}".format(
                      old_epochs +epoch, old_epochs +epochs, 
                      hist['train_loss'][old_epochs +epoch]))

    checkpoint = {'epoch':old_epochs+epoch, 
                  'model':model,
                  'model_state_dict':model.state_dict(), 
                  'optimizer':opt,
                  'optimizer_state_dict':opt.state_dict()}
    if fname!=None:
        torch.save(checkpoint, fname + '_trainingEnd.checkpoint')
            
            
    if get_best_model and test_data_loader!=None:
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


def get_derivatives(n,func,x,args):
    '''
    inputs
      n: order of derivatives
      func(x,**args): scalar function with scalar input x and constant arguments
    output
      df: list of derivatives from 0th to nth
    '''
    xtmp = torch.Tensor([x])
    xtmp.requires_grad = True
    f = func(xtmp,**args)
    df = [f]
    for i in range(n):
        df.append(torch.autograd.grad(df[-1], xtmp, create_graph=True))
    return [item[0].item() for item in df]