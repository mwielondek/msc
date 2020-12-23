from BCPNN.recurrent_modular import rmBCPNN
from BCPNN.encoder import OneHotEncoder

import numpy as np
import pandas as pd

from sklearn.metrics import fowlkes_mallows_score as fms

from . import Transformer
from ..clusters import collect_cluster_ids, binary_search
from ..scoring import GridSearch as GS

class WrapperClf(Transformer):
    """We use the old gridsearch class and using a wrapper around the transformer and clf we
    make it work for this purpose."""

    def fit(self, X):
        pass

    def predict(self, X):
        x, y = self.transform()

        enc = OneHotEncoder()
        xt = enc.fit_transform(x, recurrent=True)

        clf = rmBCPNN(verbose=False)
        clf.fit(xt, module_sizes=enc.module_sizes_)

        gval = binary_search(clf, xt, verbose=0, k=self.k)

        cls = self.get_clusters(clf, xt, np.array([gval]))

        return cls

    def scoring_fn(self, pred, y, clf):
        y = clf.ys_
        scores = [fms(y, u) for u in pred.values()]
        return max(scores)

    def get_clusters(self, clf, x, gvals):
        cls = collect_cluster_ids(clf, x, gvals, verbose=0)
        d = dict(zip(gvals, cls))
        assert len(d) > 0
        return d


class GridSearch(GS):

    def __init__(self):
        return super().__init__(WrapperClf.scoring_fn)

    def get_opt(self, X, y, params, verbose=0, **kwargs):
        clf = WrapperClf(VERBOSE_LEVEL=0, data=(X, y), **kwargs)
        clf.k = np.unique(y).size
        self.fit(clf, None, None, params, verbose=verbose)
        return self.get_res().iloc[0].params

    @staticmethod
    def pivot(res):
        arr=[]
        for r in res.iterrows():
            d=r[1][0].copy()
            d.update(fm=r[1][1])
            arr.append(d)

        return pd.DataFrame(arr)\
                 .pivot(columns='N_HYPERCOLS', values='fm', index='N_MINICOLS')\
                 .style.background_gradient(cmap='Blues', axis=None).format('{:2.0%}')

    def plot(self):
        return self.pivot(self.get_res())
