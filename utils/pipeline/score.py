import numpy as np
import pandas as pd

from functools import partial, update_wrapper

from BCPNN.recurrent_modular import rmBCPNN
from parent.msc.utils.clusters import collect_cluster_ids
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

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
        gvals = np.linspace(0.5,0.64,12)

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


    class ResultsDict(dict):

        def __init__(self, d, metrics):
            super().__init__(d)

            def to_pandas(self):
                return pd.DataFrame(self, columns=['clf'] + [m.__name__ for m in metrics])

            values_type = type(next(iter(d.values())))
            PandasTransformableList = type("PandasTransformableList", (values_type,), {'to_pandas': to_pandas})

            for k in self.keys():
                self[k] = PandasTransformableList(self[k])


    def run(self, X, y):
        self.clusterings = self._collect_clusters(X)
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
