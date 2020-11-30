from dataclasses import dataclass, field
import numpy as np

from BCPNN.recurrent_modular import rmBCPNN
from BCPNN.encoder import OneHotEncoder, ComplementEncoder

from parent.msc.utils.clusters import collect_cluster_ids
from parent.msc.utils.datasets import stratified_split
from parent.msc.utils.datasets import load_digits_784

import warnings
from sklearn import metrics
from sklearn.manifold import MDS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import KBinsDiscretizer

from scipy.cluster.vq import vq

@dataclass
class Pipeline:

    N_RFIELDS: int = 12
    N_CODEWORDS: int = 10
    N_BINS: int = 8
    N_LIM: int = 400
    N_MDS_COMPONENTS: int = 60
    VERBOSE_LEVEL: int = 1

    data: np.ndarray = field(init=False, compare=False, default_factory=load_digits_784)

    def transform(self, x=None, y=None):
        if x is None:
            x, y = self.data

        return self._transform(x, y)

    def __getattribute__(self, name):
        ret = super().__getattribute__(name)
        if callable(ret):
            print(">>", name)
        return ret

    def _transform(self, x, y):
        xb = self._discretize(x)
        xe = self._embed(xb)
        hypercols = self._cluster_hypercolumns(xe)
        minicols = self._cluster_minicolumns(hypercols, xb)

        self.codebook_encoder = Encoder(hypercols, minicols)
        xlim, y_lim = xb[:self.N_LIM], y[:self.N_LIM]
        xlim_enc = self._encode(self.codebook_encoder, xlim)
        self.clf_encoder = OneHotEncoder()
        xlim_tr = self.clf_encoder.fit_transform(xlim_enc, recurrent=True)

        return xlim_tr, y_lim


    def _discretize(self, x):
        kbins = KBinsDiscretizer(encode='ordinal', strategy='uniform', n_bins=self.N_BINS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xb = kbins.fit_transform(x)
        return xb

    def _embed(self, xb):
        similarity_matrix = np.corrcoef(xb.T)
        # mask nan (where all values either 0 or 1 most likely)
        similarity_matrix[np.isnan(similarity_matrix)] = 1

        embedding = MDS(dissimilarity='precomputed', n_components=self.N_MDS_COMPONENTS)
        xe = embedding.fit_transform(1 - similarity_matrix)

        return xe

    def _cluster_hypercolumns(self, xe):
        km = KMeans(n_clusters=self.N_RFIELDS)
        km.fit(xe)

        labels = np.unique(km.labels_)
        hypercols = np.empty(labels.size, dtype='object')
        for i, label in enumerate(labels):
            hypercols[i] = np.flatnonzero(km.labels_ == label)
        return hypercols

    def _cluster_minicolumns(self, hypercols, xb):
        km = KMeans(n_clusters=self.N_CODEWORDS)
        minicols = []
        for col in hypercols:
            km.fit(xb[:, col])
            minicols.append(km.cluster_centers_)
        return minicols

    def _encode(self, encoder, xlim):
        xlim_encoded = np.zeros((xlim.shape[0], self.N_RFIELDS), dtype=int)
        for i, sample in enumerate(xlim):
            xlim_encoded[i] = encoder.transform(sample)
        return xlim_encoded



@dataclass
class Encoder:

    hypercolumns: np.ndarray
    minicolumns: np.ndarray

    def transform(self, sample):
        sample_encoded = np.empty(self.hypercolumns.shape[0], dtype=int)
        for i, (hypercol, codebook) in enumerate(zip(self.hypercolumns, self.minicolumns)):
            obs = sample[hypercol]
            sample_encoded[i] = vq([obs], codebook)[0]
        return sample_encoded

    def inverse_transform(self, sample_encoded):
        sample_restored = np.empty(28*28)
        for codeword, idx, codebook in zip(sample_encoded.astype(int), self.hypercolumns, self.minicolumns):
            sample_restored[idx] = codebook[codeword]
        return sample_restored
