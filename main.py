from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV


def main():
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

    from sklearn.tree import DecisionTreeClassifier
    for i in (3, 6, 9, 12):
        print("max_depth: {}".format(i))

        tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=i)
        tree.fit(CIFAR_train_images, CIFAR_train_labels)

        print("CIFAR-10 훈련 세트 정확도: {:.3f}".format(tree.score(CIFAR_train_images, CIFAR_train_labels)))
        print("CIFAR-10 테스트 세트 정확도: {:.3f}".format(tree.score(CIFAR_test_images, CIFAR_test_labels)))

        tree = DecisionTreeClassifier(criterion='entropy',
                                      max_depth=i)
        tree.fit(MNIST_train_images, MNIST_train_labels)
        print("MNIST 훈련 세트 정확도: {:.3f}".format(tree.score(MNIST_train_images, MNIST_train_labels)))
        print("MNIST 테스트 세트 정확도: {:.3f}".format(tree.score(MNIST_test_images, MNIST_test_labels)))

    params_grid = {'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'max_leaf_nodes': [5, 10, None]}
    grid_search = GridSearchCV(DecisionTreeClassifier(criterion='entropy', max_depth=3), params_grid, cv=5, n_jobs=-1)
    print(grid_search.fit(MNIST_train_images, MNIST_train_labels))


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()
