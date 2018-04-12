import matplotlib.pyplot as plt
import pickle as pic 

data_pic = "decoded_imgs.pickle"
decoded_imgs = pic.load(open(data_pic, 'rb'))


n = 10  # digits to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display orginal
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.savefig('plot.png')
