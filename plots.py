import numpy as np
import matplotlib.pyplot as plt
import pickle

import seaborn as sns
sns.set_theme()

class result_struct():
    def __init__(self):
        self.loss = []
        self.acc_test = []
        self.f1_test = []
        self.rec_test = []
        self.prec_test = []

with open("results_both.txt", "rb") as fp:
    results_both = pickle.load(fp) 
with open("results_hmi.txt", "rb") as fp:
    results_hmi = pickle.load(fp) 

plt.plot(np.array(results_both.f1_test) , label = 'Hmi + Mdi')
plt.plot(np.array(results_hmi.f1_test) , label = 'Hmi')
plt.xlabel('Iteration')
#plt.ylim(0,100)
plt.ylabel('F1-score')
plt.legend()
plt.show()

plt.plot(np.array(results_both.acc_test) , label = 'Hmi + Mdi')
plt.plot(np.array(results_hmi.acc_test) , label = 'Hmi')
plt.xlabel('Iteration')
#plt.ylim(0.5,1)
plt.ylabel('Classification Accuracy')
plt.legend()
plt.show()