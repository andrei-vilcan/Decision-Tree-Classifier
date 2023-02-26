import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz

from scipy.stats import entropy

np.random.seed(1006012690)


def load_data():
    # load data in
    with open("clean_real.txt") as real_file:
        real = real_file.readlines()
    with open("clean_fake.txt") as fake_file:
        fake = fake_file.readlines()

    all_words = real + fake

    y = []
    for i in range(len(fake)):
        y.append(0)
    for i in range(len(real)):
        y.append(1)
    y = np.asarray(y)

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(all_words)
    np.asarray(x)

    x_train, x_test, x_validation, y_train, y_test, y_validation = split_data(x, y)
    return x_train, x_test, x_validation, y_train, y_test, y_validation, vectorizer.get_feature_names_out()


def split_data(x, y):
    train_ratio = 0.7
    n = x.shape[0]
    train_size = math.floor(n * train_ratio)

    leftover = n - train_size
    test_size = math.floor(leftover/2)
    validation_size = n - train_size - test_size

    assert train_size + test_size + validation_size == n

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_validation, x_test, y_validation, y_test = train_test_split(x_test, y_test)

    return x_train, x_test, x_validation, y_train, y_test, y_validation


def select_model(x_train, x_test, x_validation, y_train, y_test, y_validation):
    criterion = ["gini", "entropy", "log_loss"]
    depths = [i for i in range(3, 15)]
    gini_validation_accuracies = []
    entropy_validation_accuracies = []
    log_loss_validation_accuracies = []
    validation_accuracies = [gini_validation_accuracies, entropy_validation_accuracies, log_loss_validation_accuracies]
    best = {"accuracy": 0, "criterion": "", "max_depth": 0}
    for i in range(len(criterion)):
        for j in range(len(depths)):
            dtc = DecisionTreeClassifier(criterion=criterion[i], max_depth=depths[j])
            dtc.fit(x_train, y_train)
            accuracy = dtc.score(x_validation, y_validation)
            print("Accuracy:", str(accuracy), " |  Criterion:", criterion[i], " |  Max Depth: ", str(depths[j]))

            validation_accuracies[i].append(accuracy)

            if accuracy > best["accuracy"]:
                best["classifier"] = dtc
                best["accuracy"] = accuracy
                best["criterion"] = criterion[i]
                best["max_depth"] = depths[j]

    plt.plot(depths, validation_accuracies[0], label="gini")
    plt.plot(depths, validation_accuracies[1], label="entropy")
    plt.plot(depths, validation_accuracies[2], label="log_loss")
    plt.legend()
    plt.title('Max Depth vs. Validation Accuracies')
    plt.xlabel('Max Depth')
    plt.ylabel('Validation Accuracy')
    plt.show()
    return best


def model_test(best_model, x_train, y_train, x_test, y_test):
    the_model = DecisionTreeClassifier(criterion=best_model["criterion"], max_depth=best_model["max_depth"])
    the_model.fit(x_train, y_train)
    accuracy = the_model.score(x_test, y_test)
    print("\nThe best parameters are: Criterion:", best_model["criterion"], " |  Max Depth:", str(best_model["max_depth"]), "\nThe test accuracy of this model is:", str(accuracy))
    return best_model


def visualize_model(model, feature_names):
    dot_data = tree.export_graphviz(model["classifier"], max_depth=2, feature_names=feature_names)
    graph = graphviz.Source(dot_data)
    graph.render('graph_{}.png'.format(model["criterion"]))


def compute_information_gain(keyword, x_train, y_train, feature_names):
    # Information Gain:
    # IG(Y|X) = H(Y) − H(Y|X)

    total = y_train.shape[0]

    probability_whole = np.sum(y_train)/np.shape(y_train)

    def pre_entropy(p):
        return p * math.log(p, 2)

    def calculate_entropy(p):
        return -sum(pre_entropy(p), pre_entropy(1 - p))

    # get entropy of whole
    h_of_y = calculate_entropy(probability_whole)
    # print(h_of_y)

    # get entropy of each child and subtract both entropies from the whole
    # find rows of x_train the don't have keyword
    keyword_index = np.where(feature_names == keyword)[0]
    x_train_sub = x_train[:, keyword_index]
    yes = []
    for i in range(x_train.shape[0]):
        if x_train_sub[i, 0] == 0:
            yes.append(1)
        else:
            yes.append(0)

    # get entropy of left child
    left_sum = 0
    left_total = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 1 and x_train_sub[i, 0] == 0:
            left_sum += 1
        if x_train_sub[i, 0] == 0:
            left_total += 1

    probability_left = left_sum/left_total
    h_of_l_y = entropy([probability_left, 1 - probability_left], base=2)

    # get entropy of right child
    right_sum = 0
    right_total = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 1 and x_train_sub[i, 0] == 1:
            right_sum += 1
        if x_train_sub[i, 0] == 1:
            right_total += 1

    probability_right = right_sum/right_total
    h_of_r_y = entropy([probability_right, 1 - probability_right], base=2)

    return h_of_y - ((left_total / total) * h_of_l_y) - ((right_total / total) * h_of_r_y)


def main():
    x_train, x_test, x_validation, y_train, y_test, y_validation, feature_names = load_data()
    model = select_model(x_train, x_test, x_validation, y_train, y_test, y_validation)
    best_model = model_test(model, x_train, y_train, x_test, y_test)
    visualize_model(best_model, feature_names)
    print("Information Gain of the split on the word 'the':", compute_information_gain("the", x_train, y_train, feature_names))
    print("Information Gain of the split on the word 'hillary':", compute_information_gain("hillary", x_train, y_train, feature_names))
    print("Information Gain of the split on the word 'trumps':", compute_information_gain("trumps", x_train, y_train, feature_names))
    print("Information Gain of the split on the word 'donald':", compute_information_gain("donald", x_train, y_train, feature_names))
    print("Information Gain of the split on the word 'travel':", compute_information_gain("travel", x_train, y_train, feature_names))


if __name__ == "__main__":
    main()
