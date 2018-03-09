#%% task 1 Import
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from numpy import genfromtxt
import matplotlib.pyplot as plt
import statistics

#%% read and init
train_in = genfromtxt('./data/train_in.csv', delimiter=',')
train_out = genfromtxt('./data/train_out.csv', delimiter=',')
test_in = genfromtxt('./data/test_in.csv', delimiter=',')
test_out = genfromtxt('./data/test_out.csv', delimiter=',')


#%% task 3: Bayes Rule Classifier:
# lets compare the numbers 5 and 7
# We could consider the average intensity of the lower half 8x16 square
# we would expect the number 7 to have a lower average intensity than 5 in that area.
itemid = 420

image = train_in[itemid]
image.shape = (16,16)
subimage = image[8:16,0:16]

print(subimage.shape)

#plotting integers and their lower halves
im = plt.imshow(image, cmap='gray')
#plt.savefig('report/figures/integers/'+str(itemid)+'.png')
plt.show()

im2 = plt.imshow(subimage, cmap='gray')
#plt.savefig('report/figures/integers/lh-'+str(itemid)+'.png')
plt.show()



#%%

## empty array of lower half mean intensities
subimage = []

## indexes of 5s and 7s in train_out
idx_57 = []

lh_intensity_all = []
lh_intensity = {}

classify_57 = {}
correct_57 = {}

for i in range(len(train_in)):
    image = train_in[i]
    image.shape = (16,16)
    subimage.append(image[8:16,0:16])
    lh_intensity_all.append( float(format( subimage[i].mean() , '.2f')))

#get lower half intensities of 5's and 7's from training
for i in range(len(train_in)):
    if train_out[i] == 5 or train_out[i] == 7:
        idx_57.append(i)
        lh_intensity[i] = float(format(subimage[i].mean(), '.2f'))

values_57 = [lh_intensity[key] for key in lh_intensity]
mean_57 = float(format( statistics.mean(values_57), '.2f' ))

print(mean_57)

#%% classify from mean.. not taking overlap and probabilities into account
for key, value in lh_intensity.items():
    correct_57[key] = int(train_out[key])
    if value < mean_57:
        classify_57[key] = 7
    else:
        classify_57[key] = 5
print(correct_57)

a = np.array(list(correct_57.values()))
b = np.array(list(classify_57.values()))
print(confusion_matrix(list(correct_57.values()), list(classify_57.values())))

(a == b).sum() / len(a)


# %% histograms

#generate histograms of lower half mean intensities of 5s and 7s from training data
hist5 = {}
hist7 = {}

for key, value in correct_57.items():
    if value == 5:
        if(lh_intensity[key] in hist5.keys()):
            hist5[lh_intensity[key]] = hist5[lh_intensity[key]] + 1
        else:
            hist5[lh_intensity[key]] = 1
    elif value == 7:
        if(lh_intensity[key] in hist7.keys()):
            hist7[lh_intensity[key]] = hist7[lh_intensity[key]] + 1
        else:
            hist7[lh_intensity[key]] = 1

fives = plt.bar(list(hist5.keys()), hist5.values(), color='g', alpha=0.5,width=0.01, label='intensity of 5s')
sevens = plt.bar(list(hist7.keys()), hist7.values(), color='b', alpha=0.5,width=0.01, label='intensity of 7s')
plt.legend(handles=[fives,sevens])
plt.xlabel('intensity with 2 decimal space precision')
plt.ylabel('occurrence')
plt.title('Histogram of bottom half mean intensities for 5s and 7s')
#plt.savefig('report/figures/plots/57hist.png')
plt.show()


#%% bayes classification P(C_k|X)

values_57 = [lh_intensity[key] for key in lh_intensity]
mean_57 = float(format( statistics.mean(values_57), '.2f' ))
P_C5 = sum(hist5.values()) / (sum(hist7.values()) + sum(hist5.values()))
P_C7 = sum(hist7.values()) / (sum(hist7.values()) + sum(hist5.values()))

for key, value in lh_intensity.items():
    correct_57[key] = int(train_out[key])
    #in case of uncertainty, if the value could either represent a 5 or a 7, apply bayes theorem
    if(value in hist5.keys() and value in hist7.keys() ):
        P_X_C5 = hist5[value] / (hist7[value] + hist5[value])
        P_X_C7 = hist7[value] / (hist7[value] + hist5[value])
        P_X = P_X_C5 * P_C5 + P_X_C7 * P_C7
        P_C5_X = (P_X_C5 * P_C5) / P_X
        P_C7_X = (P_X_C7 * P_C7) / P_X
        if(np.random.rand() > P_C5_X):
            classify_57[key] = 7
        else:
            classify_57[key] = 5
    #fallback to classifying by mean.
    elif value < mean_57:
        classify_57[key] = 7
    else:
        classify_57[key] = 5


print(correct_57)
print(classify_57)

a = np.array(list(correct_57.values()))
b = np.array(list(classify_57.values()))

print(confusion_matrix(list(correct_57.values()), list(classify_57.values())))
(a == b).sum() / len(a)
