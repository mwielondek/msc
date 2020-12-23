import numpy as np
import pandas as pd

from functools import partial, update_wrapper, cached_property
from operator import attrgetter
from contextlib import redirect_stdout
from sys import stdout

from BCPNN.recurrent_modular import rmBCPNN
from parent.msc.utils.clusters import collect_cluster_ids, binary_search
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import matplotlib.pyplot as plt

class Wrappers:

    k = 10

    class KM:

        @property
        def n_cls(self):
            return [Wrappers.k]

        def __repr__(self):
            return "kMeans"

        def get_clusters(self, x):
            clusterings = {}
            for n in self.n_cls:
                km = KMeans(n_clusters=n)
                km.fit(x)
                clusterings[n] = km.labels_
            return clusterings


    class AC:
        linkage = ['ward', 'average', 'complete', 'single']

        def __repr__(self):
            return "Agglomerative"

        def get_clusters(self, x):
            clusterings = {}
            for l in self.linkage:
                clf = AgglomerativeClustering(n_clusters=Wrappers.k, linkage=l)
                clf.fit(x)
                clusterings[l] = clf.labels_
            return clusterings


    class RB:
        gvals = np.linspace(0.3,0.6,20)

        def __init__(self, limit=True):
            self.limit = limit

        def __repr__(self):
            return "rmBCPNN"

        def get_clusters(self, x):
            clf = rmBCPNN(verbose=False)
            clf.fit(x, module_sizes=self.module_sizes)
            if self.limit:
                g = binary_search(clf, x, verbose=0, k=Wrappers.k)
                print("G-value for k={}: {:.4f}".format(Wrappers.k, g))
                cls = collect_cluster_ids(clf, x, np.array([g]), verbose=0)
            else:
                cls = collect_cluster_ids(clf, x, self.gvals, verbose=0)
            d = dict(zip(self.gvals, cls))
            assert len(d) > 0
            return d


    class LDA:
        #Needs to be manually fitted prior to adding it to DEFAULT_CLFS

        def __repr__(self):
            return "LDA"

        def __init__(self, x, y):
            self.clf = LDA().fit(x, y)

        def get_clusters(self, x):
            pred = self.clf.predict(x)
            d = dict(lda=pred)
            return d

class Scorer:

    DEFAULT_METRICS = [metrics.fowlkes_mallows_score, metrics.completeness_score, metrics.homogeneity_score,
                    metrics.normalized_mutual_info_score,
                    update_wrapper(partial(metrics.f1_score, average='micro'), metrics.f1_score)]
    DEFAULT_CLFS = [Wrappers.AC(), Wrappers.KM(), Wrappers.RB(limit=True)]

    def __init__(self, metrics=None, clfs=None, module_sizes=None, verbose=1):
        self.metrics = metrics or self.DEFAULT_METRICS
        self.clfs = clfs or self.DEFAULT_CLFS
        self.verbose = verbose

        if clfs is None:
            assert module_sizes is not None
            self.DEFAULT_CLFS[-1].module_sizes = module_sizes

    def run(self, X, y, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.y = y
        with redirect_stdout(None if not self.verbose else stdout):
            self.clusterings = self.Clusterings(self._collect_clusters(X), labels_true=y, X=X)
            results = dict(zip(['scores', 'params'], self._collect_metrics(self.clusterings, y)))
            self.results = self.ResultsDict(results, self.metrics)

    def _collect_clusters(self, X):
        return [dict(clf=clf, clusterings=clf.get_clusters(X)) for clf in self.clfs]

    def _collect_metrics(self, clusterings, true_y):
        results_scores = []
        results_params = []

        for c in clusterings:
            clf, cls_dict = list(c.values())

            scores = np.empty((len(cls_dict), len(self.metrics)))
            for i, clustering in enumerate(cls_dict.values()):
                if self.verbose > 0:
                    print(f"{clf!s:>14}:\t {np.unique(clustering).size} clusters")
                scores[i] = [m(true_y, clustering) for m in self.metrics]

            results_scores.append([str(clf), *np.max(scores, axis=0)])

            idx = np.argsort(scores, axis=0)[-1]
            sorted_params = np.array(list(cls_dict.keys()))[idx]
            results_params.append([str(clf), *sorted_params])

        return results_scores, results_params


    class ResultsDict(dict):

        def __init__(self, d, metrics):
            super().__init__(d)

            def to_pandas(self):
                return pd.DataFrame(self, columns=['clf'] + [m.__name__ for m in metrics])

            values_type = type(next(iter(d.values())))
            PandasTransformableList = type("PandasTransformableList", (values_type,), {'to_pandas': to_pandas})

            for k in self.keys():
                self[k] = PandasTransformableList(self[k])


    class Clusterings(list):

        def __init__(self, arg, *, labels_true, X):
            for d in arg:
                d['clusterings'] = self.SubClusterings(d['clusterings'], labels_true=labels_true, X=X)

            super().__init__(arg)

        def __getitem__(self, key):
            value = super().__getitem__(key)
            return value['clusterings']

        def plot_similarity_matrices(self):
            return self.plot_matrices('similarity_matrix')

        def plot_confusion_matrices(self):
            return self.plot_matrices('confusion_matrix.normalized')

        def plot_matrices(self, matrix):
            n_cols = len(self)
            f, axs = plt.subplots(1, n_cols)

            for i,ax in enumerate(axs):
                attrgetter(matrix)(self[i]).plot(ax, colorbar=False)
                ax.set_title(super().__getitem__(i)['clf'])

            im = ax.get_images()[0]
            f.colorbar(im, cax = f.add_axes([0.95, 0.33, 0.02, 0.35]))

            w = len(self) * 8
            h = w // 2
            f.set_size_inches(w,h)
            plt.close()
            return f


        class SubClusterings(dict):

            def __init__(self, arg, *, labels_true, X):
                super().__init__(arg)
                self.labels_true = labels_true
                self.X = X

            def first(self):
                return next(iter(self.values()))

            @cached_property
            def confusion_matrix(self):
                return Scorer.ConfusionMatrix(self.labels_true, self.first())

            @cached_property
            def similarity_matrix(self):
                order = (-self.first()).argsort()
                cluster_sizes = np.unique(self.first(), return_counts=True)[1][::-1]
                return Scorer.SimilarityMatrix(self.X[order], cluster_sizes)


    class ConfusionMatrix(np.ndarray):

        def __new__(cls, labels_true, labels_pred):
            labels_pred_matched = cls._match_labels(labels_true, labels_pred)
            return metrics.confusion_matrix(labels_true, labels_pred_matched).view(cls)

        def normalize(self, mode='recall'):
            if mode == 'recall':
                axis = 1
            elif mode == 'precision':
                axis = 0
            return self.astype('float') / self.sum(axis=axis)

        @property
        def normalized(self):
            return self.normalize('recall')

        def plot(self, ax=None, colorbar=True):
            cmd = metrics.ConfusionMatrixDisplay(self).plot(cmap=plt.cm.Blues, ax=ax)
            if not colorbar:
                cmd.im_.colorbar.remove()
            return cmd

        @staticmethod
        def _match_labels(true_labels, pred_labels, cm=None, output_transl_table=False):
            """Match partitions with true clusters using max likelihood.

            For each partitioning, assign a cluster based on which the most number of intra-partition samples
            belong to, giving priority to partitions with highest number of samples belonging to a single cluster.
            """
            if cm is None:
                cm = metrics.confusion_matrix(true_labels, pred_labels)
                n_classes = np.unique(true_labels).size
                max_idx = np.argsort(cm, axis=None)[::-1]
                visited_true = []
                visited_pred = []
                trans_pred_true = np.full(n_classes,-1)
                for m in max_idx:
                    i, j = np.unravel_index(m, cm.shape)
                    if not j in visited_true and not i in visited_pred:
                        trans_pred_true[j] = i
                        visited_true.append(j)
                        visited_pred.append(i)
                        if len(visited_true) == cm.shape[0]:
                            assert set(trans_pred_true) == set(range(n_classes)), "incomplete matching"
                            if output_transl_table:
                                print(trans_pred_true)
                            return trans_pred_true[pred_labels]

    class SimilarityMatrix(np.ndarray):

        def __new__(cls, X, cluster_sizes):
            obj = metrics.pairwise.cosine_similarity(X).view(cls)
            obj.cluster_sizes = cluster_sizes
            return obj


        def plot(self, a=None, colorbar=True):
            if a is None:
                _,a = plt.subplots()
            im = a.imshow(self, cmap='jet')
            if colorbar:
                plt.colorbar(im)
            for axis in [a.xaxis, a.yaxis]:
                axis.set_ticks(np.cumsum(self.cluster_sizes))
                axis.set_ticklabels(range(self.cluster_sizes.size))
            return a
