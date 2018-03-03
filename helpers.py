import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    ply.yticks(tick_marks, classes)

    if normalize is True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusin matrix")
    else:
        print("Confusion matrix, without norm")

    print(cm)

    thres = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel("Predicted Label")
    filename = 'plots/plot_confusion_matrix.png'
    plt.save(filename)


def number_of():
    oredered_list = OrderedDict(("list_" + str(i), []) for i in range(10))
    for i in range(10):
        print("number of " + str(i) + "s: " +
              str(len(oredered_list["list_" + str(i)])))
