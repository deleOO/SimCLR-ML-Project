import pickle
import tarfile
import numpy as np
import wget
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# function to save model
def save_model(model, optimizer, scheduler, current_epoch, name):
    out = os.path.join('saved_models/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)


# function to save loss
def save_loss(tr_loss, val_loss):
    import pickle
    with open('saved_models/tr_loss.pkl', 'wb') as f:
        pickle.dump(tr_loss, f)
    with open('saved_models/val_loss.pkl', 'wb') as f:
        pickle.dump(val_loss, f)

# function to save loss checkpoint
def save_loss_checkpoint(tr_loss, val_loss, current_epoch):
    import pickle
    with open('saved_models/tr_loss_{}.pkl'.format(current_epoch), 'wb') as f:
        pickle.dump(tr_loss, f)
    with open('saved_models/val_loss_{}.pkl'.format(current_epoch), 'wb') as f:
        pickle.dump(val_loss, f)


# Loss
def plot_loss(tr_loss,val_loss):
    # Plot training & validation loss values
    plt.plot(tr_loss)
    plt.plot(val_loss)
    plt.title('Model loss',fontsize=10)
    plt.ylabel('Loss',fontsize=10)
    plt.xlabel('Epoch',fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(['Train', 'Validation'], loc='upper left',prop={'size': 10})
    plt.savefig('saved_plots/loss_plot.png', transparent = True)
    plt.tight_layout()
    plt.show()


# Accuracy

def plot_accuracy(tr_acc,val_acc):
    # Plot training & validation accuracy values
    plt.plot(tr_acc)
    plt.plot(val_acc)
    plt.title('Model accuracy',fontsize=10)
    plt.ylabel('Accuracy',fontsize=10)
    plt.xlabel('Epoch',fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.legend(['Train', 'Validation'], loc='upper left',prop={'size': 10})
    plt.savefig('saved_plots/accuracy_plot.png', transparent = True)
    plt.tight_layout()
    plt.show()