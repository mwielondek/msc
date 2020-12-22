import numpy as np
import time

def get_unique_patterns(X):
    """Returns distinct pattern count"""
    return np.unique(X, axis=0)

def round_patterns(X, decimals=2, mode='truncate'):
    """Truncates (or rounds) patterns to `decimals`"""
    if mode == 'truncate':
        return np.trunc(X*10**decimals)/(10**decimals)
    elif mode == 'round':
        return X.round(decimals=decimals)

def get_num_clusters(X, **kwargs):
    return len(get_unique_patterns(round_patterns(X, **kwargs)))

def binary_search(clf, X, k=10, low=0, high=100, delta=1e-3, verbose=2):
    iters = 0
    max_iter = np.log2((high - low) * 1/delta)
    if verbose > 0:
        print("Running binary search, max iters:", max_iter)

    while iters < max_iter:
        iters += 1
        g = low + (high - low) / 2
        clf.g = g
        pred = clf.predict(X)
        n = get_num_clusters(pred)
        if verbose > 1:
            print("iter {:<2}: g={:.5f} >> n={}".format(iters, g, n))
        if n == k:
            if verbose > 0:
                print("Found value after {} iterations".format(np.floor(iters)))
            return g
        elif n < k:
            low = g
        elif n > k:
            high = g

    raise UserWarning('couldnt find proper value for g')

def get_cluster_arrays(X, oneindexed=False, **kwargs):
    """Returns an array of clusters, where each value corresponds to sample ID"""
    clusters = {}
    rounded = round_patterns(X, **kwargs)
    for pat in get_unique_patterns(rounded):
        clusters[pat.tobytes()] = []
    for idx,pat in enumerate(rounded):
        clusters[pat.tobytes()].append(idx + oneindexed)
    return list(clusters.values())

def get_cluster_ids(X, **kwargs):
    """Returns an array of cluster IDs, where the index corresponds to sample ID"""
    n_samples, _ = X.shape
    clusters = get_cluster_arrays(X, False, **kwargs)
    arr = np.zeros(n_samples).astype(int)
    for cidx, c in enumerate(clusters):
        for sample in c:
            arr[sample] = cidx
    return arr

def collect_cluster_ids(clf, X, gvals, decimals=2, fit_params={}, predict_params={}, **kwargs):
    """Get cluster IDs as a function of g values"""
    n_samples, _ = X.shape
    n_gvals = len(gvals)
    clusters = np.empty((n_gvals, n_samples))

    # check if already fitted, otherwise fit
    if not hasattr(clf, 'X_'):
        clf.fit(X, **fit_params)

    decimals_param = {}
    if decimals is not None:

        if decimals <= 0:
            # default to a precision of `decimals` less SF than clf.TOL
            decimals =  int(np.log10(clf.TOL) * -1) + decimals

        decimals_param = dict(decimals=decimals)

    print_progress(0, gvals.size)
    for idx, g in enumerate(gvals):
        clf.g = g
        pred = clf.predict(X, **predict_params)
        clusters[idx] = get_cluster_ids(pred, **decimals_param, **kwargs)
        print_progress(idx + 1, gvals.size)

    return clusters.astype(int)

previous_lap = time.time()
def print_progress(i, n):
    global previous_lap
    now = time.time()
    lap_duration = now - previous_lap
    est_time_left = (n-i) * lap_duration if i > 0 else 0
    previous_lap = now
    timerstr = "est time left: {:5.0f}s".format(est_time_left)
    print("Progress: {}{} | {}".format("#"*i, "-"*(n-i), timerstr), end='\r')

def get_distance(X):
    """Calculate hamming distance between patterns"""
    dists = {}
    upper = X.shape[1]
    for idx, pat in enumerate(X):
        dists[idx+1] = {}
        for idx2, pat2 in enumerate(X):
            dists[idx+1][idx2+1] = upper - (pat == pat2).sum()


    table = {}
    for n in range(upper+1):
        table[n] = {}
    for idx, v in dists.items():
        for n in range(upper+1):
            keys = { key for key,value in v.items() if value == n }
            table[n][idx] = keys
    return scrub_dict(table)

# removes empty values
def scrub_dict(d):
    if type(d) is dict:
        return dict((k, scrub_dict(v)) for k, v in d.items() if v and scrub_dict(v))
    else:
        return d
