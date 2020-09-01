import numpy as np
import matplotlib.pyplot as plt

  
def history(history):
    flagTest = False
    plt.semilogy(history['train_loss'])
    if 'test_loss' in history:
        if len(history['test_loss']!=0):
            flagTest = True
            
    if flagTest:
        plt.semilogy(history['test_loss'])
        plt.legend(['train_loss','test_loss'])
    else:
        plt.legend(['train_loss'])