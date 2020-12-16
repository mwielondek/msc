import numpy as np
import pandas as pd

from functools import partial, update_wrapper, cached_property

from BCPNN.recurrent_modular import rmBCPNN
from parent.msc.utils.clusters import collect_cluster_ids
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics
import matplotlib.pyplot as plt

class Wrappers:

    class KM:
        n_cls = [10] #range(2,15)

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
                clf = AgglomerativeClustering(n_clusters=10, linkage=l)
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
            cls = collect_cluster_ids(clf, x, self.gvals)
            d = dict(zip(self.gvals, cls))
            if self.limit:
                # filter for only those that have 10 clusters (or closest to 10)
                cls_len = np.array([np.unique(x).size for x in cls])
                closest = cls_len[np.array(abs(cls_len - 10)).argmin()]
                d = {k:v for k,v in d.items() if np.unique(v).size == closest}
            assert len(d) > 0
            return d

class Scorer:

    DEFAULT_METRICS = [metrics.fowlkes_mallows_score, metrics.completeness_score, metrics.homogeneity_score,
                    metrics.normalized_mutual_info_score,
                    update_wrapper(partial(metrics.f1_score, average='micro'), metrics.f1_score)]
    DEFAULT_CLFS = [Wrappers.AC(), Wrappers.KM(), Wrappers.RB(limit=True)]

    def __init__(self, metrics=None, clfs=None, module_sizes=None):
        self.metrics = metrics or self.DEFAULT_METRICS
        self.clfs = clfs or self.DEFAULT_CLFS

        if clfs is None:
            assert module_sizes is not None
            self.DEFAULT_CLFS[-1].module_sizes = module_sizes

    def run(self, X, y):
        self.y = y
        self.clusterings = self.Clusterings(self._collect_clusters(X), labels_true=y)
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

        def __init__(self, arg, *, labels_true):
            for d in arg:
                d['clusterings'] = self.SubClusterings(d['clusterings'], labels_true=labels_true)

            super().__init__(arg)

        class SubClusterings(dict):

            def __init__(self, arg, *, labels_true):
                super().__init__(arg)
                self.labels_true = labels_true

            def first(self):
                return next(iter(self.values()))

            @cached_property
            def confusion_matrix(self):
                return Scorer.ConfusionMatrix(self.labels_true, self.first())

        def __getitem__(self, key):
            value = super().__getitem__(key)
            return value['clusterings']

    class ConfusionMatrix(np.ndarray):

        def __new__(cls, labels_true, labels_pred):
            labels_pred_matched = cls._match_labels(labels_true, labels_pred)
            return metrics.confusion_matrix(labels_true, labels_pred_matched).view(cls)


        def plot(self):
            metrics.ConfusionMatrixDisplay(self).plot(cmap=plt.cm.Blues)

        @staticmethod
        def _match_labels(true_labels, pred_labels, cm=None):
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
                            print(trans_pred_true)
                            return trans_pred_true[pred_labels]
