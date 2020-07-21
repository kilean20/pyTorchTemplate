import numpy as np
import matplotlib.pyplot as plt

  
def history(history):
    plt.semilogy(history['train_loss'])
    if 'test_loss' in history:
        plt.semilogy(history['test_loss'])
        plt.legend(['train_loss','test_loss'])
    else:
        plt.legend(['train_loss'])