



# Task2 - take #2
train_pre = np.empty(len(train_out))
test_pre = np.empty(len(test_out))

# for i in range(len(train_in)):
# 	current_dist = pairwise_distances(centers, train_in[i], metric='cosine')
# 	train_pre = np.argmin(current_dist)

conf_matrix_train = confusion_matrix(train_out, train_pre)


# calculate the correctly classified digits
correct_rate_train = np.zeros(10)
for i in range(10):
    correct_rate_train[i] = float(
        conf_matrix_train[i, i]) / np.sum(conf_matrix_train[i, :])
print("Correct rate of training")
print(correct_rate_train)


# create CM for test data
for i in range(len(test_out)):
    current_dist = pairwise_distances(centers, test_in[i], metric='cosine')
    test_pre[i] = np.argmin(current_dist)

conf_matrix_test = confusion_matrix(test_out, test_pre)

correct_rate_test = np.zeros(10)
for i in range(10):
    correct_rate_test[i] = float(
        conf_matrix_test[i, i]) / np.sum(conf_matrix_test[i, :])
print("correct rate on testing data")

print(correct_rate_test)
