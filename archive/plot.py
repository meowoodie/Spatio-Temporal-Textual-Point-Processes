import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == '__main__':
    precision = np.loadtxt("result/precision_beta2_from-5to10.txt", delimiter=',')
    recall    = np.loadtxt("result/recalls_beta2_from-5to10.txt", delimiter=',')

    precision = precision.mean(axis=1)
    recall    = recall.mean(axis=1)

    print(precision)
    print(recall)

    font = {
        # 'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
    }

    x = np.linspace(-5, 10, 51)
    y = precision
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages('result/precision_beta2_from-5to10.pdf') as pdf:
        fig = plt.figure()
        plt.plot(x, y, 'ro-')
        plt.title('Precision of 500 retrieval')
        plt.xlabel(r'$\log \beta_2$')
        plt.ylabel('precision')
        # plt.grid(color='gray', linestyle='-', linewidth=1)
        # plt.show()
        pdf.savefig(fig)

    x = np.linspace(-5, 10, 51)
    y = recall
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages('result/recall_beta2_from-5to10.pdf') as pdf:
        fig = plt.figure()
        plt.plot(x, y, 'bo-')
        plt.title('Recall of 500 retrieval')
        plt.xlabel(r'$\log \beta_2$')
        plt.ylabel('recall')
        # plt.grid(color='gray', linestyle='-', linewidth=1)
        # plt.show()
        pdf.savefig(fig)

    # 2 * p * r / (p + r)
    x = np.linspace(-5, 10, 51)
    y = 2 * precision * recall / (precision + recall)
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages('result/fscore_beta2_from-5to10.pdf') as pdf:
        fig = plt.figure()
        plt.plot(x, y, 'go-')
        plt.title(r'F'+'-score of 500 retrieval')
        plt.xlabel(r'$\log \beta_2$')
        plt.ylabel(r'F'+'-score')
        # plt.grid(color='gray', linestyle='-', linewidth=1)
        # plt.show()
        pdf.savefig(fig)
