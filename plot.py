import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

def brokenyaxes_figure(
        xspan, yrobbery, yburglary, yall, yscaler=1e+4,
        robbery_ylim=(1.6136e+5, 1.6143e+5), burglary_ylim=(3.5099e+5, 3.5110e+5), all_ylim=(5.3974e+5, 5.3980e+5),
        title='Data log-likelihood over iterations', xlabel='iteration', ylabel='log-likelihood',
        filename='result/comp_loglik_iter25.pdf', is_scientific=False):
    # scaling the dataset and ticks
    yrobbery  = np.array(yrobbery/yscaler).astype(np.float32)
    yburglary = np.array(yburglary/yscaler).astype(np.float32)
    yall      = np.array(yall/yscaler).astype(np.float32)
    robbery_ylim  = np.array(robbery_ylim)/yscaler
    burglary_ylim = np.array(burglary_ylim)/yscaler
    all_ylim      = np.array(all_ylim)/yscaler
    ylabel        = ylabel + r'$\times 10^{%d}$' % np.log10(yscaler)
    # font configuration
    font = {
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
    }
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages(filename) as pdf:
        f, (ax3, ax2, ax1) = plt.subplots(3, 1, sharex=True)
        # plot the same data on both axes
        line1 = ax1.plot(xspan, yrobbery, c='red', label='robbery cases')
        line2 = ax2.plot(xspan, yburglary, c='blue', label='burglary cases')
        line3 = ax3.plot(xspan, yall, c='grey', label='mixed cases')
        # plot horizontal maximum line
        ax1.axhline(y=yrobbery.max(), linestyle=':', c='red', linewidth=1, alpha=0.7)
        ax2.axhline(y=yburglary.max(), linestyle=':', c='blue', linewidth=1, alpha=0.7)
        ax3.axhline(y=yall.max(), linestyle=':', c='grey', linewidth=1, alpha=0.7)
        # plot vertical maximum line
        plt.axvline(x=xspan[yrobbery.argmax()], linestyle='-.', c='red', linewidth=1, alpha=0.7, ymin=0,ymax=3.4, zorder=0, clip_on=False)
        plt.axvline(x=xspan[yburglary.argmax()], linestyle='-.', c='blue', linewidth=1, alpha=0.7, ymin=0,ymax=3.4, zorder=0, clip_on=False)
        plt.axvline(x=xspan[yall.argmax()], linestyle='-.', c='grey', linewidth=1, alpha=0.7, ymin=0,ymax=3.4, zorder=0, clip_on=False)
        # scatter the maximum points
        ax1.scatter(xspan[yrobbery.argmax()], yrobbery.max(), c="red",zorder=2)
        ax2.scatter(xspan[yburglary.argmax()], yburglary.max(), c="blue",zorder=2)
        ax3.scatter(xspan[yall.argmax()], yall.max(), c="grey",zorder=2)
        # get handles and labels
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()
        # combine handles and labels
        handles = handles1 + handles2 + handles3
        labels  = labels1 + labels2 + labels3
        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(*robbery_ylim)  # robbery
        ax2.set_ylim(*burglary_ylim) # burglary
        ax3.set_ylim(*all_ylim)      # all mixed data
        # hide the spines between ax1 and ax3
        ax1.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax2.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        ax3.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        # plot broken slash
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)        # bottom-left diagonal
        ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        kwargs.update(transform=ax2.transAxes)        # switch to the bottom axes
        ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
        kwargs.update(transform=ax3.transAxes)        # switch to the bottom axes
        ax3.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
        # plot legend
        f.legend(handles, labels, loc=(0.64, 0.67))
        ax3.set_title(title)
        ax1.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        if is_scientific:
            fmt = FormatStrFormatter('%.1f')
            ax1.yaxis.set_major_formatter(fmt)
            ax2.yaxis.set_major_formatter(fmt)
            ax3.yaxis.set_major_formatter(fmt)
            # ax1.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
            # ax2.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
            # ax3.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pdf.savefig(f)

def lines_figure(
        xspan, yrobbery, yburglary, yall,
        title='Recall of 500 retrieval', xlabel=r'$\log \beta_2$', ylabel='recall',
        filename='result/comp_recall_beta2_from-5to10.pdf'):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots()
        line_a = ax.plot(xspan, yall, '-', c='gray', linewidth=2, label='mixed cases')
        line_b = ax.plot(xspan, yburglary, '-', c='blue', linewidth=2, label='burglary cases')
        line_r = ax.plot(xspan, yrobbery, '-', c='red', linewidth=2, label='robbery cases')
        plt.axvline(x=xspan[yall.argmax()], linestyle='-.', c='gray', linewidth=1)
        plt.axvline(x=xspan[yburglary.argmax()]+0.05, linestyle='-.', c='blue', linewidth=1)
        plt.axvline(x=xspan[yrobbery.argmax()], linestyle='-.', c='red', linewidth=1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        legend = ax.legend(loc='upper right')
        # Put a nicer background color on the legend.
        legend.get_frame()
        pdf.savefig(fig)

if __name__ == '__main__':

    # results of convergence analysis
    all_precision = np.loadtxt("result/other_precision_convergence.txt", delimiter=',')
    all_recall    = np.loadtxt("result/other_recalls_convergence.txt", delimiter=',')
    all_loglik    = np.loadtxt('result/other_loglik_convergence.txt', delimiter=',')
    all_lowerb    = np.loadtxt('result/other_lowerb_convergence.txt', delimiter=',')
    burglary_precision = np.loadtxt("result/burglary_precision_convergence.txt", delimiter=',')
    burglary_recall    = np.loadtxt("result/burglary_recalls_convergence.txt", delimiter=',')
    burglary_loglik    = np.loadtxt('result/burglary_loglik_convergence.txt', delimiter=',')
    burglary_lowerb    = np.loadtxt('result/burglary_lowerb_convergence.txt', delimiter=',')
    robbery_precision = np.loadtxt("result/robbery_precision_convergence.txt", delimiter=',')
    robbery_recall    = np.loadtxt("result/robbery_recalls_convergence.txt", delimiter=',')
    robbery_loglik    = np.loadtxt('result/robbery_loglik_convergence.txt', delimiter=',')
    robbery_lowerb    = np.loadtxt('result/robbery_lowerb_convergence.txt', delimiter=',')

    # plot loglikelihood over iterations
    brokenyaxes_figure(
        xspan=list(range(1, 26)), yrobbery=robbery_loglik, yburglary=burglary_loglik, yall=all_loglik,
        yscaler=1e+4, robbery_ylim=(16.136e+4, 16.143e+4), burglary_ylim=(35.099e+4, 35.110e+4), all_ylim=(53.974e+4, 53.980e+4),
        title='Data log-likelihood over iterations', xlabel='iteration', ylabel='log-likelihood',
        filename='result/comp_loglik_iter25.pdf', is_scientific=True)

    # plot lower bound of loglikelihood over iterations
    brokenyaxes_figure(
        xspan=list(range(1, 26)), yrobbery=robbery_lowerb, yburglary=burglary_lowerb, yall=all_lowerb,
        yscaler=1e+4, robbery_ylim=(7.111e+4, 7.114e+4), burglary_ylim=(25.552e+4, 25.554e+4), all_ylim=(45.572e+4, 45.574e+4),
        title='Lower bound of data log-likelihood over iterations', xlabel='iteration', ylabel='lower bound of log-likelihood',
        filename='result/comp_lowerb_iter25.pdf', is_scientific=True)

    # plot precision over iterations
    brokenyaxes_figure(
        xspan=list(range(1, 26)), yrobbery=robbery_precision, yburglary=burglary_precision, yall=all_precision,
        yscaler=1e-2, robbery_ylim=(1.5e-2, 6.5e-2), burglary_ylim=(2.5e-2, 7.5e-2), all_ylim=(9.1e-2, 14.1e-2),
        title='Precision over iterations', xlabel='iteration', ylabel='precision',
        filename='result/comp_precision_iter25.pdf', is_scientific=True)

    brokenyaxes_figure(
        xspan=list(range(1, 26)), yrobbery=robbery_recall, yburglary=burglary_recall, yall=all_recall,
        yscaler=1e-1, robbery_ylim=(2.0e-1, 2.8e-1), burglary_ylim=(0.7e-1, 1.5e-1), all_ylim=(1.5e-1, 2.3e-1),
        title='Recall over iterations', xlabel='iteration', ylabel='recall',
        filename='result/comp_recall_iter25.pdf', is_scientific=True)

    # result of spatio factor analysis
    all_precision      = np.loadtxt("result/other_precision_beta1_from-15to0.txt", delimiter=',')
    all_recall         = np.loadtxt("result/other_recalls_beta1_from-15to0.txt", delimiter=',')
    burglary_precision = np.loadtxt("result/burglary_precision_beta1_from-15to0.txt", delimiter=',')
    burglary_recall    = np.loadtxt("result/burglary_recalls_beta1_from-15to0.txt", delimiter=',')
    robbery_precision  = np.loadtxt("result/robbery_precision_beta1_from-15to0.txt", delimiter=',')
    robbery_recall     = np.loadtxt("result/robbery_recalls_beta1_from-15to0.txt", delimiter=',')

    all_precision      = all_precision.mean(axis=1)
    all_recall         = all_recall.mean(axis=1)
    burglary_precision = burglary_precision.mean(axis=1)
    burglary_recall    = burglary_recall.mean(axis=1)
    robbery_precision  = robbery_precision.mean(axis=1)
    robbery_recall     = robbery_recall.mean(axis=1)

    # precision
    lines_figure(
        np.linspace(0, -15, 51), robbery_precision, burglary_precision, all_precision,
        title='Precision of 500 retrieval', xlabel=r'$\log \beta_1$', ylabel='precision',
        filename='result/comp_precision_beta1_from-15to0.pdf')

    # recall
    lines_figure(
        np.linspace(0, -15, 51), robbery_recall, burglary_recall, all_recall,
        title='Recall of 500 retrieval', xlabel=r'$\log \beta_1$', ylabel='recall',
        filename='result/comp_recall_beta1_from-15to0.pdf')

    # f-score
    lines_figure(
        np.linspace(0, -15, 51),
        2 * (robbery_precision * robbery_recall) / (robbery_precision + robbery_recall),
        2 * (burglary_precision * burglary_recall) / (burglary_precision + burglary_recall),
        2 * (all_precision * all_recall) / (all_precision + all_recall),
        title=r'$F$-score of 500 retrieval', xlabel=r'$\log \beta_1$', ylabel=r'$F$-score',
        filename='result/comp_fscore_beta1_from-15to0.pdf')
