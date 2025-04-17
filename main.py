import pandas
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV


def decision_tree(CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels,
                  MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels):
    # GridSearchCV
    params_grid = {"max_depth": [3, 6, 9, 12],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'max_leaf_nodes': [5, 10, None]}
    grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), params_grid, cv=5, n_jobs=-1)
    grid_search.fit(CIFAR_train_images, CIFAR_train_labels)
    print("CIFAR-10 Best parameters:", grid_search.best_params_)
    print("CIFAR-10 Best cross-validation score:", grid_search.best_score_)
    results = pandas.DataFrame(grid_search.cv_results_)
    print(results[['params', 'mean_test_score', 'std_test_score']])

    grid_search.fit(MNIST_train_images, MNIST_train_labels)
    print("MNIST Best parameters:", grid_search.best_params_)
    print("MNIST Best cross-validation score:", grid_search.best_score_)
    results = pandas.DataFrame(grid_search.cv_results_)
    print(results[['params', 'mean_test_score', 'std_test_score']])


def SVM(CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels,
        MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels):
    # SVM
    params_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(svm.SVC(), params_grid, cv=5, n_jobs=-1)
    grid_search.fit(CIFAR_train_images, CIFAR_train_labels)
    print("CIFAR-10 Best parameters:", grid_search.best_params_)
    print("CIFAR-10 Best cross-validation score:", grid_search.best_score_)
    results = pandas.DataFrame(grid_search.cv_results_)
    print(results[['params', 'mean_test_score', 'std_test_score']])

    grid_search.fit(MNIST_train_images, MNIST_train_labels)
    print("MNIST Best parameters:", grid_search.best_params_)
    print("MNIST Best cross-validation score:", grid_search.best_score_)
    results = pandas.DataFrame(grid_search.cv_results_)
    print(results[['params', 'mean_test_score', 'std_test_score']])



def load_data():
    # CIFAR-10
    CIFAR_transform_train = transforms.Compose([transforms.ToTensor()])
    CIFAR_transform_test = transforms.Compose([transforms.ToTensor()])

    trainset_CIFAR = datasets.CIFAR10(root='./data', train=True, download=True, transform=CIFAR_transform_train)
    testset_CIFAR = datasets.CIFAR10(root='./data', train=False, download=True, transform=CIFAR_transform_test)
    CIFAR_train = DataLoader(trainset_CIFAR, batch_size=32, shuffle=True, num_workers=2)
    CIFAR_test = DataLoader(testset_CIFAR, batch_size=32, shuffle=False, num_workers=2)
    CIFAR_train_images = []
    CIFAR_train_labels = []
    for batch in CIFAR_train:
        images, labels = batch
        images_flat = images.view(images.shape[0], -1)
        CIFAR_train_images.append(images_flat.numpy())
        CIFAR_train_labels.append(labels.numpy())
    CIFAR_train_images = np.vstack(CIFAR_train_images)
    CIFAR_train_labels = np.concatenate(CIFAR_train_labels)

    CIFAR_test_images = []
    CIFAR_test_labels = []
    for batch in CIFAR_test:
        images, labels = batch
        images_flat = images.view(images.shape[0], -1)
        CIFAR_test_images.append(images_flat.numpy())
        CIFAR_test_labels.append(labels.numpy())
    CIFAR_test_images = np.vstack(CIFAR_test_images)
    CIFAR_test_labels = np.concatenate(CIFAR_test_labels)

    # MNIST
    mnist_train_transform = transforms.Compose([transforms.ToTensor()])
    mnist_test_transform = transforms.Compose([transforms.ToTensor()])

    trainset_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_train_transform)
    testset_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_test_transform)

    MNIST_train = DataLoader(trainset_mnist, batch_size=32, shuffle=True, num_workers=2)
    MNIST_test = DataLoader(testset_mnist, batch_size=32, shuffle=False, num_workers=2)

    MNIST_train_images = []
    MNIST_train_labels = []
    for batch in MNIST_train:
        images, labels = batch
        images_flat = images.view(images.shape[0], -1)
        MNIST_train_images.append(images_flat.numpy())
        MNIST_train_labels.append(labels.numpy())
    MNIST_train_images = np.vstack(MNIST_train_images)
    MNIST_train_labels = np.concatenate(MNIST_train_labels)

    MNIST_test_images = []
    MNIST_test_labels = []
    for batch in MNIST_test:
        images, labels = batch
        images_flat = images.view(images.shape[0], -1)
        MNIST_test_images.append(images_flat.numpy())
        MNIST_test_labels.append(labels.numpy())
    MNIST_test_images = np.vstack(MNIST_test_images)
    MNIST_test_labels = np.concatenate(MNIST_test_labels)
    return (CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels,
            MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels)


def main():
    CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels, \
        MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels = load_data()

    decision_tree(CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels,
                  MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels)

    SVM(CIFAR_train_images, CIFAR_train_labels, CIFAR_test_images, CIFAR_test_labels,
        MNIST_train_images, MNIST_train_labels, MNIST_test_images, MNIST_test_labels)


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
