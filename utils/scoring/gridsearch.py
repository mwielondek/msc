import pandas as pd

from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.model_selection import ParameterGrid

from ..clusters import get_cluster_ids

class GridSearch:

    decimals_key = '__decimals'
    mode_key = '__mode'

    def __init__(self, scoring_fn):
        self.scoring_fn = scoring_fn.__get__(self)


    def fit(self, clf, X, y, params, fit_params={}, verbose=0, decimals=None):
        if verbose > 0:
            print("Fitting...", end='')
        clf.fit(X, **fit_params)
        if verbose > 0:
            print(" ✔︎")

        if verbose > 0:
            nl = '\n'
            if verbose == 1:
                nl = ''
            print("Predicting...", end=nl)
        res = dict(params=[], score=[])
        params = ParameterGrid(params)
        for i, param_set in enumerate(params):
            if verbose > 0:
                if verbose > 1:
                    print("[{}/{}] Predicting with params {}".format(i+1, len(params), param_set))
                else:
                    print('.' * max(1, 100 // len(params)), end='')
            for k,v in param_set.items():
                if k in [self.decimals_key, self.mode_key]:
                    continue
                setattr(clf, k, v)
            pred = clf.predict(X)

            self.decimals_param = {}
            if self.decimals_key in param_set.keys():
                decimals = param_set[self.decimals_key]
            if decimals is not None:
                self.decimals_param = dict(decimals=decimals)

            self.mode_param = {}
            if self.mode_key in param_set.keys():
                self.mode_param = dict(mode=param_set[self.mode_key])

            score = self.scoring_fn(pred, y, clf)
            res['params'].append(param_set)
            res['score'].append(score)

        if verbose > 0:
            print(" ✔︎")

        self.res_ = res

    def get_res(self):
        return pd.DataFrame(self.res_).sort_values('score', ascending=False)

    def disp_res(self):
        with pd.option_context('display.max_colwidth', None, 'display.float_format', '{:.2%}'.format):
            display(self.get_res())


class GridSearchCluster(GridSearch):

    def fn(self, pred, y, clf):
        clsid = get_cluster_ids(pred, **self.decimals_param, **self.mode_param)
        score = ami(y, clsid)
        return score

    def __init__(self):
        super().__init__(self.fn)
