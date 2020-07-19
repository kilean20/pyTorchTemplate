import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
import pickle
from copy import deepcopy as copy
try:
    from torchsummary import summary
except:
    print('torchsummary package not available')
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



####################################
## Models
####################################


class FCNN(torch.nn.Module):
    def __init__(self, nodes, activation, dropout_p=0.0):
        super(FCNN,self).__init__()

        self.nodes = nodes
        self.activation = activation
        self.dropout_p = dropout_p

    if dropout_p > 0.0:
        self.seq = [torch.nn.Linear(12,self.nodes[0]),torch.nn.Dropout(self.dropout_p),self.activation]
    else:
        self.seq = [torch.nn.Linear(12,self.nodes[0]),self.activation]
    for i in range(len(self.nodes)-1):
    if dropout_p > 0.0:
        self.seq = self.seq + [torch.nn.Linear(self.nodes[i],self.nodes[i+1]),torch.nn.Dropout(self.dropout_p), self.activation]
    else:
        self.seq = self.seq + [torch.nn.Linear(self.nodes[i],self.nodes[i+1]),torch.nn.Dropout(self.dropout_p), self.activation]
    self.seq = self.seq + [torch.nn.Linear(self.nodes[i+1],1)]

    self.nn = torch.nn.Sequential(*self.seq)

    def forward(self, x):
        return self.nn(x)
    
    
####################################
## Train
####################################  
      
# def train(epochs,model,optimizer,criterion activation,dropout_p,lr):
#   model = FCNN(nodes=nodes,activation=activation,dropout_p=dropout_p).to(device)
#   # print(model.seq)
#   optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#   criterion = torch.nn.MSELoss()
#   hist = {'train_loss':np.zeros(epochs),
#           'test_loss' :np.zeros(epochs)}
#   for epoch in range(epochs):
#       model.train()
#       train_loss = 0
#       for data in train_data_loader:
#           inputs = data[:,:12]
#           outputs = data[:,-1].reshape(len(data[:,-1]),-1)
#           optimizer.zero_grad()
#           inputs = inputs.to(device)
#           # print('inputs.shape',inputs.shape)
#           model_outputs = model(inputs)
#           # print('model_outputs.shape',model_outputs.shape)
#           # print('outputs.shape',outputs.shape)
#           loss = criterion(outputs.to(device), model_outputs)
#           loss.backward()
#           optimizer.step()
#           train_loss += loss.item()

#       train_loss /= len(train_data_loader)


#       model.eval()
#       test_loss = 0
#       with torch.no_grad():
#           for data in test_data_loader:
#               inputs = data[:,:12]
#               outputs = data[:,-1].reshape(len(data[:,-1]),-1)
#               # inputs = inputs.float().to(device)
#               inputs = inputs.to(device)
#               model_outputs = model(inputs)
#               loss = criterion(outputs.to(device), model_outputs)
#               test_loss += loss.item()

#       test_loss /= len(test_data_loader)

#       hist['train_loss'][epoch] = train_loss
#       hist['test_loss'][epoch] = test_loss


#       if epoch == int(0.3*epochs):
#           best_loss = test_loss
#       if epoch > int(0.3*epochs):
#           if test_loss < best_loss:
#               best_loss = test_loss
#               checkpoint = {'epoch':epoch+1, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
#               torch.save(checkpoint, path+'data/[FCNN]best.checkpoint')
      
#       # display the epoch training loss
#       print("epoch : {}/{}, train loss = {:.6f}, test loss = {:.6f}".format(epoch + 1, epochs, hist['train_loss'][epoch], hist['test_loss'][epoch]))

#   checkpoint = torch.load(path+'data/[FCNN]best.checkpoint')
#   model.load_state_dict(checkpoint['state_dict'])
#   val_err = 0
#   with torch.no_grad():
#       for data in val_data_loader:
#           inputs = data[:,:12]
#           outputs = data[:,-1].reshape(len(data[:,-1]),-1)
#           inputs = inputs.float().to(device)
#           model_outpus = model(inputs)
#           val_err += F.mse_loss(model_outpus, outputs.to(device)).item()
#   val_err /= len(val_data_loader)

#   return model,hist,val_err