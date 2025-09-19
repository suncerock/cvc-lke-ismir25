import numpy as np
    
########## global evaluation measures on mpe outputs ##########

def recall(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.count_nonzero(y_true[y_true != -1] == y_pred[y_true != -1]) / np.count_nonzero(y_true != -1)

def mirex(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mode_pred = y_pred % 2
    mode_true = y_true % 2
    ks_pred = y_pred // 2
    ks_true = y_true // 2

    correct = np.count_nonzero((ks_true == ks_pred) * (mode_true == mode_pred))
    fifth = np.count_nonzero(((ks_true - ks_pred) % 12 == 7) * (mode_true == mode_pred)) \
            + np.count_nonzero(((ks_true - ks_pred) % 12 == 5) * (mode_true == mode_pred))
    relative = np.count_nonzero(((ks_true - ks_pred) % 12 == 3) * (mode_pred == 1) * (mode_true == 0)) \
            + np.count_nonzero(((ks_true - ks_pred) % 12 == 9) * (mode_pred == 0) * (mode_true == 1))
    parallel = np.count_nonzero((ks_true == ks_pred) * (mode_true != mode_pred))
    return (correct + fifth * 0.5 + relative * 0.3 + parallel * 0.2) / len(y_true)

def tvd(x, y):
    return 1 - 0.5 * np.sum(np.abs(x - y))

def cyclic_cdf(arr, step=1):
    """Compute the cyclic cumulative distribution function (CDF)."""
    N = len(arr)
    cdfs = np.zeros((N, N))  # Store shifted CDFs
    
    for k in range(N):
        # Compute cumulative sum starting at k (to handle circularity)
        cdfs[k, :] = np.cumsum(np.roll(arr, -k))
    
    return cdfs

def cyclic_wasserstein_cdf(a, a_prime):
    """Compute cyclic Wasserstein distance using CDFs."""
    N = len(a)
    
    # Compute all cyclic CDFs
    cdf_a = cyclic_cdf(a)
    cdf_a_prime = cyclic_cdf(a_prime)
    
    # Compute Wasserstein distance for all shifts and take the minimum
    wasserstein_distances = np.sum(np.abs(cdf_a - cdf_a_prime), axis=1) / N
    return 2 * np.min(wasserstein_distances)  # Scale factor from your paper

def emd(x, y):
    assert x.shape == (24, ) and y.shape == (24, )
    
    x = x.reshape(12, 2)
    y = y.reshape(12, 2)

    # C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    # ->
    # C, G, D, A, E, B, F#, C#, G#, D#, A#, F
    ids = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5])

    x[:, 0] = x[ids][:, 0]
    # x[:, 1] = x[ids][:, 1]
    x[:, 1] = x[(ids - 3) % 12][:, 1]
    y[:, 0] = y[ids][:, 0]
    # y[:, 1] = y[ids][:, 1]
    y[:, 1] = y[(ids - 3) % 12][:, 1]
    x = x.reshape(24, )
    y = y.reshape(24, )
    return cyclic_wasserstein_cdf(x, y)
