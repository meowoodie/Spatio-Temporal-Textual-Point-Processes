import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict

from gensim import corpora, models
from gensim.matutils import corpus2dense

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

def baselines_figure(
        xspan, ysttpp_rbm, yrbm, ysvd, yae, ylda, yrand,
        title='Recall of 500 retrieval', xlabel=r'$N$', ylabel='recall',
        filename='result/comp_burglary_fscore_N_from100to1000.pdf'):
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots()
        line6 = ax.plot(xspan, yrand, '-', c='black', linewidth=2, label='Random')
        line5 = ax.plot(xspan, ylda, '-', c='orange', linewidth=2, label='LDA')
        line4 = ax.plot(xspan, yae, '-', c='green', linewidth=2, label='Autoencoder')
        line3 = ax.plot(xspan, ysvd, '-', c='blue', linewidth=2, label='SVD')
        line2 = ax.plot(xspan, yrbm, '-', c='brown', linewidth=2, label='RegRBM')
        line1 = ax.plot(xspan, ysttpp_rbm, '-', c='red', linewidth=2, label='STTPP+RegRBM')
        
        plt.axvline(x=xspan[yrand.argmax()], linestyle='-.', c='black', linewidth=1)
        plt.axvline(x=xspan[ylda.argmax()], linestyle='-.', c='orange', linewidth=1)
        plt.axvline(x=xspan[yae.argmax()], linestyle='-.', c='green', linewidth=1)
        plt.axvline(x=xspan[ysvd.argmax()]+0.05, linestyle='-.', c='blue', linewidth=1)
        plt.axvline(x=xspan[yrbm.argmax()], linestyle='-.', c='brown', linewidth=1)
        plt.axvline(x=xspan[ysttpp_rbm.argmax()], linestyle='-.', c='red', linewidth=1)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        legend = ax.legend(loc='upper right')
        # Put a nicer background color on the legend.
        legend.get_frame()
        pdf.savefig(fig)

def corpus_histogram(corpus, dictionary, sort_by="weighted_sum", \
                     show=False, N=10, file_name="result/test.pdf", title=None):
    """
    Calculate the histogram for each of the ngrams that appears in the indicated
    corpus.
    """
    # target corpus with ngrams being sorted by their tfidf values
    sorted_corpus = [ sorted(doc, key=lambda x: -x[1]) for doc in corpus ]
    # sorted_corpus = [ doc[doc[:, 1].argsort()] for doc in corpus ]
    # distributions of each of ngrams in the corpus
    ngram_dist   = defaultdict(lambda: [])
    # build dict for distributions of each of ngrams (key=ngram_id, val=list of tfidf values)
    for doc in sorted_corpus:
    	for ngram_id, tfidf_val in doc:
    		ngram_dist[dictionary[ngram_id]].append(tfidf_val)
    # filter the ngrams which have less than one tfidf value
    lowfreq_ngrams = [ ngram
    	for ngram, tfidf_set in ngram_dist.items()
    	if len(tfidf_set) < 2 ]
    for ngram in lowfreq_ngrams:
    	ngram_dist.pop(ngram)
    # target of sorting indicated by input parameter
    sort_target = []
    if sort_by == "count":
    	# count of each of ngrams in the corpus
    	sort_target = [ [ ngram, len(tfidf_set) ]
    		for ngram, tfidf_set in ngram_dist.items() ]
    elif sort_by == "weighted_sum":
    	# weighted sum of each of ngrams in the corpus
    	sort_target = [ [ ngram, sum(tfidf_set) ]
    		for ngram, tfidf_set in ngram_dist.items() ]
    # sorted ngrams
    sorted_ngram = sorted(sort_target, key=lambda x: -x[1])[:N]
    # visualize the distributions for top N ngrams
    if show:
    	import seaborn as sns
    	with PdfPages(file_name) as pdf:
    		fig, ax = plt.subplots(1, 1)
    		sns.set(color_codes=True)
    		for ngram, value in sorted_ngram:
    			sns.distplot(ngram_dist[ngram],
    				hist=False, rug=False, ax=ax, label="%s (%f)" % (ngram, value))
    		ax.set(xlabel='tfidf value', ylabel='frequency (count)')
    		if title is not None:
    			ax.set_title(title, fontweight="bold")
    		ax.legend(frameon=False)
    		pdf.savefig(fig)
    return dict(ngram_dist), sorted_ngram

if __name__ == '__main__':

    dataset = 'robbery'
    sttpp_p = np.loadtxt("result/%s_precision_N_from100to1000.txt" % (dataset), delimiter=',')
    rbm_p   = np.loadtxt("result/gbrbm_%s_precision_N_from100to1000.txt" % (dataset), delimiter=',')
    svd_p   = np.loadtxt("result/svd1k_%s_precision_N_from100to1000.txt" % (dataset), delimiter=',')
    ae_p    = np.loadtxt("result/autoencoder_%s_precision_N_from100to1000.txt" % (dataset), delimiter=',')
    lda_p   = np.loadtxt("result/lda1k_%s_precision_N_from100to1000.txt" % (dataset), delimiter=',')
    rand_p  = np.loadtxt("result/random_%s_precision_N_from100to1000.txt" % (dataset), delimiter=',')

    sttpp_r = np.loadtxt("result/%s_recall_N_from100to1000.txt" % (dataset), delimiter=',')
    rbm_r   = np.loadtxt("result/gbrbm_%s_recalls_N_from100to1000.txt" % (dataset), delimiter=',')
    svd_r   = np.loadtxt("result/svd1k_%s_recall_N_from100to1000.txt" % (dataset), delimiter=',')
    ae_r    = np.loadtxt("result/autoencoder_%s_recall_N_from100to1000.txt" % (dataset), delimiter=',')
    lda_r   = np.loadtxt("result/lda1k_%s_recalls_N_from100to1000.txt" % (dataset), delimiter=',')
    rand_r  = np.loadtxt("result/random_%s_recalls_N_from100to1000.txt" % (dataset), delimiter=',')
    
    sttpp_p = sttpp_p.mean(axis=1)
    rbm_p   = rbm_p.mean(axis=1)
    svd_p   = svd_p.mean(axis=1)
    ae_p    = ae_p.mean(axis=1)
    lda_p   = lda_p.mean(axis=1)
    rand_p  = rand_p.mean(axis=1)

    sttpp_r = sttpp_r.mean(axis=1)
    rbm_r   = rbm_r.mean(axis=1)
    svd_r   = svd_r.mean(axis=1)
    ae_r    = ae_r.mean(axis=1)
    lda_r   = lda_r.mean(axis=1)
    rand_r  = rand_r.mean(axis=1)
    
    sttpp = 2 * (sttpp_p * sttpp_r) / (sttpp_p + sttpp_r + 1e+10)
    rbm   = 2 * (rbm_p * rbm_r) / (rbm_p + rbm_r + 1e+10)
    svd   = 2 * (svd_p * svd_r) / (svd_p + svd_r + 1e+10)
    ae    = 2 * (ae_p * ae_r) / (ae_p + ae_r + 1e+10)
    lda   = 2 * (lda_p * lda_r) / (lda_p + lda_r + 1e+10)
    rand  = 2 * (rand_p * rand_r) / (rand_p + rand_r + 1e+10)
    #     2 * (burglary_precision * burglary_recall) / (burglary_precision + burglary_recall),
    #     2 * (all_precision * all_recall) / (all_precision + all_recall),
    
    baselines_figure(
        np.linspace(100, 1000, 51).astype(np.int32),
        svd, rbm, sttpp, ae, lda, rand,
        title=r'$F_1$ score of retrieval for %s cases' % 'mixed', xlabel=r'$N$', ylabel=r'$F_1$ score',
        filename='result/comp_%s_%s_N_from100to1000.pdf' % (dataset, 'fscore'))

    baselines_figure(
        np.linspace(100, 1000, 51).astype(np.int32),
        svd_r, rbm_r, sttpp_r, ae_r, lda_r, rand_r,
        title='Recall of retrieval for %s cases' % dataset, xlabel=r'$N$', ylabel='Recall',
        filename='result/comp_%s_%s_N_from100to1000.pdf' % (dataset, 'recall'))
    
    baselines_figure(
        np.linspace(100, 1000, 51).astype(np.int32),
        svd_p, rbm_p, sttpp_p, ae_p, lda_p, rand_p,
        title='Precision of retrieval for %s cases' % dataset, xlabel=r'$N$', ylabel='Precision',
        filename='result/comp_%s_%s_N_from100to1000.pdf' % (dataset, 'precision'))

    # # results of convergence analysis
    # all_precision = np.loadtxt("result/other_precision_convergence.txt", delimiter=',')
    # all_recall    = np.loadtxt("result/other_recalls_convergence.txt", delimiter=',')
    # all_loglik    = np.loadtxt('result/other_loglik_convergence.txt', delimiter=',')
    # all_lowerb    = np.loadtxt('result/other_lowerb_convergence.txt', delimiter=',')
    # burglary_precision = np.loadtxt("result/burglary_precision_convergence.txt", delimiter=',')
    # burglary_recall    = np.loadtxt("result/burglary_recalls_convergence.txt", delimiter=',')
    # burglary_loglik    = np.loadtxt('result/burglary_loglik_convergence.txt', delimiter=',')
    # burglary_lowerb    = np.loadtxt('result/burglary_lowerb_convergence.txt', delimiter=',')
    # robbery_precision = np.loadtxt("result/robbery_precision_convergence.txt", delimiter=',')
    # robbery_recall    = np.loadtxt("result/robbery_recalls_convergence.txt", delimiter=',')
    # robbery_loglik    = np.loadtxt('result/robbery_loglik_convergence.txt', delimiter=',')
    # robbery_lowerb    = np.loadtxt('result/robbery_lowerb_convergence.txt', delimiter=',')
    #
    # # plot loglikelihood over iterations
    # brokenyaxes_figure(
    #     xspan=list(range(1, 26)), yrobbery=robbery_loglik, yburglary=burglary_loglik, yall=all_loglik,
    #     yscaler=1e+4, robbery_ylim=(16.136e+4, 16.143e+4), burglary_ylim=(35.099e+4, 35.110e+4), all_ylim=(53.974e+4, 53.980e+4),
    #     title='Data log-likelihood over iterations', xlabel='iteration', ylabel='log-likelihood',
    #     filename='result/comp_loglik_iter25.pdf', is_scientific=True)
    #
    # # plot lower bound of loglikelihood over iterations
    # brokenyaxes_figure(
    #     xspan=list(range(1, 26)), yrobbery=robbery_lowerb, yburglary=burglary_lowerb, yall=all_lowerb,
    #     yscaler=1e+4, robbery_ylim=(7.111e+4, 7.114e+4), burglary_ylim=(25.552e+4, 25.554e+4), all_ylim=(45.572e+4, 45.574e+4),
    #     title='Lower bound of data log-likelihood over iterations', xlabel='iteration', ylabel='lower bound of log-likelihood',
    #     filename='result/comp_lowerb_iter25.pdf', is_scientific=True)
    #
    # # plot precision over iterations
    # brokenyaxes_figure(
    #     xspan=list(range(1, 26)), yrobbery=robbery_precision, yburglary=burglary_precision, yall=all_precision,
    #     yscaler=1e-2, robbery_ylim=(1.5e-2, 6.5e-2), burglary_ylim=(2.5e-2, 7.5e-2), all_ylim=(9.1e-2, 14.1e-2),
    #     title='Precision over iterations', xlabel='iteration', ylabel='precision',
    #     filename='result/comp_precision_iter25.pdf', is_scientific=True)
    #
    # brokenyaxes_figure(
    #     xspan=list(range(1, 26)), yrobbery=robbery_recall, yburglary=burglary_recall, yall=all_recall,
    #     yscaler=1e-1, robbery_ylim=(2.0e-1, 2.8e-1), burglary_ylim=(0.7e-1, 1.5e-1), all_ylim=(1.5e-1, 2.3e-1),
    #     title='Recall over iterations', xlabel='iteration', ylabel='recall',
    #     filename='result/comp_recall_iter25.pdf', is_scientific=True)

    # # result of spatio factor analysis
    # all_precision      = np.loadtxt("result/other_precision_N_from100to1000.txt", delimiter=',')
    # all_recall         = np.loadtxt("result/other_recalls_N_from100to1000.txt", delimiter=',')
    # burglary_precision = np.loadtxt("result/burglary_precision_N_from100to1000.txt", delimiter=',')
    # burglary_recall    = np.loadtxt("result/burglary_recalls_N_from100to1000.txt", delimiter=',')
    # robbery_precision  = np.loadtxt("result/robbery_precision_N_from100to1000.txt", delimiter=',')
    # robbery_recall     = np.loadtxt("result/robbery_recalls_N_from100to1000.txt", delimiter=',')
    #
    # all_precision      = all_precision.mean(axis=1)
    # all_recall         = all_recall.mean(axis=1)
    # burglary_precision = burglary_precision.mean(axis=1)
    # burglary_recall    = burglary_recall.mean(axis=1)
    # robbery_precision  = robbery_precision.mean(axis=1)
    # robbery_recall     = robbery_recall.mean(axis=1)
    #
    # # precision
    # lines_figure(
    #     np.linspace(100, 1000, 51).astype(np.int32), robbery_precision, burglary_precision, all_precision,
    #     title='Precision of retrieval', xlabel=r'$N$', ylabel='precision',
    #     filename='result/comp_precision_N_from100to1000.pdf')
    #
    # # recall
    # lines_figure(
    #     np.linspace(100, 1000, 51).astype(np.int32), robbery_recall, burglary_recall, all_recall,
    #     title='Recall of retrieval', xlabel=r'$N$', ylabel='recall',
    #     filename='result/comp_recall_N_from100to1000.pdf')
    #
    # # f-score
    # lines_figure(
    #     np.linspace(100, 1000, 51).astype(np.int32),
    #     2 * (robbery_precision * robbery_recall) / (robbery_precision + robbery_recall),
    #     2 * (burglary_precision * burglary_recall) / (burglary_precision + burglary_recall),
    #     2 * (all_precision * all_recall) / (all_precision + all_recall),
    #     title=r'$F_1$-score of retrieval', xlabel=r'$N$', ylabel=r'$F_1$-score',
    #     filename='result/comp_fscore_N_from100to1000.pdf')

    # # result of spatio factor analysis
    # dataset = 'other'
    # metric  = 'f-score'
    # sttpp = np.loadtxt("result/%s_%s_N_from100to1000.txt" % (dataset, metric), delimiter=',')
    # svd   = np.loadtxt("result/svd_%s_%s_N_from100to1000.txt" % (dataset, metric), delimiter=',')
    # ae    = np.loadtxt("result/autoencoder_%s_%s_N_from100to1000.txt" % (dataset, metric), delimiter=',')
    #
    # sttpp = sttpp.mean(axis=1)
    # svd   = svd.mean(axis=1)
    # ae    = ae.mean(axis=1)
    #
    

    # ngram_dict = corpora.Dictionary.load("resource/dict/10k.bigram.dict")
    # print(ngram_dict)
    # corpus_tfidf = corpora.MmCorpus('resource/corpus/10k.bigram.tfidf.corpus')
    # print(corpus_tfidf)
    #
    # corpus_tfidf = corpus_tfidf[48:55]
    # corpus_histogram(corpus_tfidf, ngram_dict, sort_by="weighted_sum", show=True, N=10, file_name="result/test.pdf", title="Pedestrain Robbery Committed by Suspect T")
