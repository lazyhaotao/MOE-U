import numpy as np
import matplotlib.pyplot as plt
import os.path
META_OPTIMIZER_NAMES = ['LSTM optimizer', 'MOE-A','MOE-U']
NORM_OPTIMIZER_NAMES = ['Adam', 'RMSprop', 'AdaGrad', 'SGD', 'SGDMomentum']
# target dataset
DATASET = "PaviaC"
# source dataset
SOURCEDATASET = "KSC"
SOURCEDATASET_L = " " # LSTM optimizer loss results
SOURCEDATASET_A = " " # MOE-A loss results
SOURCEDATASET_E = " "# MOE-U optimizer loss results

# load the hand-designed loss
steps = 50
loss_data = np.zeros((10,50,8)) # Number of tests, step size, number of optimizers
path = "../NormResults/"+DATASET+"/"
for i in range(len(NORM_OPTIMIZER_NAMES)):
    file_name = "loss_10_"+NORM_OPTIMIZER_NAMES[i]+".npy"
    loss_data[:,:,i] = np.load(path+file_name)

# load the LSTM optimizer loss data
path = "../LstmOptimizerResults/"+DATASET+"/"+SOURCEDATASET_L+"/"
file_name = "sample10loss_.npy"
loss_data[:,:,5] = np.load(path+file_name)

# load the MetaOE-A loss data
path = "../LstmOpAverageResults/"+DATASET+"/"+SOURCEDATASET_A+"/"
file_name = "sample10loss_.npy"
loss_data[:,:,6] = np.load(path+file_name)
# load the MetaOE-L loss data
path = "../LstmOpEnsembleResults/"+DATASET+"/"+SOURCEDATASET_E+"/"
file_name = "sample10loss_.npy"
loss_data[:,:,7] = np.load(path+file_name)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
name_list = ['Adam', 'RMSProp', 'AdaGrad', 'SGD', 'SGD with Momentum', 'LSTM optimizer', 'MOE-A','MOE-U']
color_list = ['brown','blue','purple','orange','red','black', 'green', 'pink']
marker_list = ['o','x','v','^','p','s','*','d']

# Calculate the final converged mean and variance
for i in range(loss_data.shape[2]):
    print(name_list[i], "mean:",np.mean(loss_data[:,45:50,i]), "std:", np.std(loss_data[:,45:50,i]))
# draw loss
for i in range(len(name_list)):
    a = loss_data[:,:,i]
    b = np.mean(a,axis=0)
    plt.plot(list(range(0,steps)), b, color=color_list[i], linestyle='-',label=name_list[i], marker=marker_list[i], markevery=10, markersize=7)
plt.xlabel("Steps", font1)
plt.ylabel("Loss", font1)
plt.yticks(fontproperties = 'Times New Roman', size = 13)
plt.xticks(fontproperties = 'Times New Roman', size = 13)
plt.xlim(0,50)
plt.legend(loc='upper right',fontsize=9)
plt.tight_layout()
save_pth = "./Loss/" + DATASET +"/"
if not os.path.isdir(save_pth):
    os.makedirs(save_pth)
plt.savefig(save_pth+"loss_trained_on_"+SOURCEDATASET+".png")
plt.show()