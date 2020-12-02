from dataclasses import dataclass, field
import numpy as np
import warnings

from sklearn.manifold import MDS
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.vq import vq

from ..datasets import load_digits_784, stratified_split

@dataclass
class Transformer:

    N_HYPERCOLS: int = 12
    N_MINICOLS: int = 10
    N_BINS: int = 8
    N_LIM: int = 400
    N_MDS_COMPONENTS: int = 60
    VERBOSE_LEVEL: int = 1

    data: np.ndarray = field(init=False, compare=False, repr=False, default_factory=load_digits_784)

    def transform(self, X=None, y=None):
        if X is None:
            X, y = self.data

        return self._transform(X, y)

    def __getattribute__(self, name):
        ret = super().__getattribute__(name)
        if name == "VERBOSE_LEVEL":
            return ret
        if self.VERBOSE_LEVEL > 0 and callable(ret):
            print(">>", name)
        return ret

    def _transform(self, X, y):
        Xb = self._discretize(X)
        Xe = self._embed(Xb)
        hypercols = self._cluster_hypercolumns(Xe)
        minicols = self._cluster_minicolumns(hypercols, Xb)

        self.codebook_encoder = Encoder(hypercols, minicols)
        Xlim, y_lim = stratified_split(Xb, y, self.N_LIM//np.unique(y).size)
        Xlim_enc = self.codebook_encoder.transform(Xlim)

        # add all vars to self with underscore for easy access
        for var, val in locals().items():
            setattr(self, var + '_', val)

        return Xlim_enc, y_lim


    def _discretize(self, X):
        kbins = KBinsDiscretizer(encode='ordinal', strategy='uniform', n_bins=self.N_BINS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xb = kbins.fit_transform(X)
        return Xb

    def _embed(self, Xb):
        similarity_matrix = cosine_similarity(Xb.T)

        embedding = MDS(dissimilarity='precomputed', n_components=self.N_MDS_COMPONENTS, n_jobs=-1, n_init=2)
        Xe = embedding.fit_transform(1 - similarity_matrix)

        return Xe

    def _cluster_hypercolumns(self, Xe):
        km = KMeans(n_clusters=self.N_HYPERCOLS)
        km.fit(Xe)

        labels = np.unique(km.labels_)
        hypercols = np.empty(labels.size, dtype='object')
        for i, label in enumerate(labels):
            hypercols[i] = np.flatnonzero(km.labels_ == label)
        return hypercols

    def _cluster_minicolumns(self, hypercols, Xb):
        km = KMeans(n_clusters=self.N_MINICOLS, n_init=2)
        minicols = []
        for col in hypercols:
            km.fit(Xb[:, col])
            minicols.append(km.cluster_centers_)
        return minicols


@dataclass
class Encoder:

    hypercolumns: np.ndarray
    minicolumns: np.ndarray

    def transform(self, X):
        n_samples, _ = X.shape
        X_encoded = np.empty((n_samples, self.hypercolumns.shape[0]), dtype=int)
        for i, (hypercol, codebook) in enumerate(zip(self.hypercolumns, self.minicolumns)):
            obs = X[:, hypercol]
            X_encoded[:, i] = vq(obs, codebook)[0]
        return X_encoded

    def inverse_transform(self, sample_encoded):
        sample_restored = np.empty(28*28)
        for codeword, idx, codebook in zip(sample_encoded.astype(int), self.hypercolumns, self.minicolumns):
            sample_restored[idx] = codebook[codeword]
        return sample_restored
