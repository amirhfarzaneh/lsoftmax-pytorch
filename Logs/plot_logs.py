import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)


# Load text files function
def load_log(file):
    test_acc = []
    with open(file) as f:
        for line in f:
            if 'Test' in line:
                line = line.split()
                acc = float(line[-2][0:4]) / 100
                test_acc.append(acc)
    return test_acc


def main():
    path = 'm=4/'
    filename = '-m4.txt'
    shows = {}
    shows['test_acc_1'] = load_log(path + 'log1' + filename)

    for key in sorted(shows.keys()):
        if 'acc' in key:
            epochs = np.arange(1, 1 + len(shows[key]))
            max = np.max(shows[key])
            plt.plot(epochs, shows[key], label='{}:{}'.format(key, max))
    plt.legend(loc='lower right')
    plt.title('Test Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
