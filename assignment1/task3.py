import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import warnings
import statistics
from timeit import default_timer as timer
from scipy.spatial import distance

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(threshold=np.nan)

# start_time = timer()

# training data
train_in = np.genfromtxt("data/train_in.csv", delimiter=",")
test_in = np.genfromtxt("data/test_in.csv", delimiter=",")
# label values
train_out = np.genfromtxt("data/train_out.csv")
test_out = np.genfromtxt("data/test_out.csv")

def draw_digit(itemid):
    image = train_in[itemid]
    image.shape = (16, 16)
    subimage = image[8:16, 0:16]

    # plotting integers and their lower halves
    im = plt.imshow(image, cmap='gray')
    # plt.savefig('report/figures/integers/'+str(itemid)+'.png')
    plt.show()

    im2 = plt.imshow(subimage, cmap='gray')
    # plt.savefig('report/figures/integers/lh-'+str(itemid)+'.png')
    plt.show()


def extract_feature(data_in, data_out):
    # we chose to look at the lower halves of 5s and 7s, and measure the mean grey values (intensity)
    # we observed that the curve in lh of 5s should give a higher mean
    # intensity than the straight stem of a 7.
    lh_intensity_all = []
    lh_intensity = {}

    correct_57 = {}
    subimage = []

    for i in range(len(data_in)):
        image = data_in[i]
        image.shape = (16, 16)
        subimage.append(image[8:16, 0:16])
        if data_out[i] == 5 or data_out[i] == 7:
            lh_intensity[i] = float(format(subimage[i].mean(), '.2f'))

    for key, value in lh_intensity.items():
        correct_57[key] = int(data_out[key])

    return(lh_intensity, correct_57)


def get_mean(feature):
    values = [feature[key] for key in feature]
    mean = float(format(statistics.mean(values), '.2f'))
    return(mean)


def generate_histograms_from_training(lh_intensity, correct_57):
    # generate histograms of lower half mean intensities of 5s and 7s from
    # training data
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

    return (hist5, hist7)


def draw_histogram(hist5, hist7):
    fives = plt.bar(list(hist5.keys()), hist5.values(), color='g',
                    alpha=0.5, width=0.01, label='intensity of 5s')
    sevens = plt.bar(list(hist7.keys()), hist7.values(), color='b',
                     alpha=0.5, width=0.01, label='intensity of 7s')
    plt.legend(handles=[fives, sevens])
    plt.xlabel('intensity with 2 decimal space precision')
    plt.ylabel('occurrence')
    plt.title('Histogram of bottom half mean intensities for 5s and 7s')
    # plt.savefig('report/figures/plots/57hist.png')
    plt.show()


def bayes_classification(feature, correct, hist5, hist7):
    mean = get_mean(feature)
    # calculate priors
    P_C5 = sum(hist5.values()) / (sum(hist7.values()) + sum(hist5.values()))
    P_C7 = sum(hist7.values()) / (sum(hist7.values()) + sum(hist5.values()))

    classification = {}

    for key, value in feature.items():
        # in case of uncertainty, if the value could either represent a 5 or a
        # 7, apply bayes theorem
        if(value in hist5.keys() and value in hist7.keys()):
            # class-conditionals
            P_X_C5 = hist5[value] / (hist7[value] + hist5[value])
            P_X_C7 = hist7[value] / (hist7[value] + hist5[value])
            # scaling factor
            P_X = P_X_C5 * P_C5 + P_X_C7 * P_C7
            # posteriors
            P_C5_X = (P_X_C5 * P_C5) / P_X
            P_C7_X = (P_X_C7 * P_C7) / P_X
            if(np.random.rand() < P_C7_X):
                classification[key] = 7
            else:
                classification[key] = 5
        # fallback to classifying, using mean as decision boundary.
        elif value < mean:
            classification[key] = 7
        else:
            classification[key] = 5

    print(correct)
    print(classification)

    a = np.array(list(correct.values()))
    b = np.array(list(classification.values()))

    print(confusion_matrix(a, b))
    acc = (a == b).sum() / len(a)
    print(acc)


def main():
    draw_digit(420)
    draw_digit(3)
    lh_training, correct_training = extract_feature(train_in, train_out)
    hist5, hist7 = generate_histograms_from_training(
        lh_training, correct_training)
    draw_histogram(hist5, hist7)

    lh_testing, correct_testing = extract_feature(test_in, test_out)

    bayes_classification(lh_training, correct_training, hist5, hist7)
    # try classifying the test data, using the feature histograms from our
    # training data
    bayes_classification(lh_testing, correct_testing, hist5, hist7)


if __name__ == '__main__':
    main()
