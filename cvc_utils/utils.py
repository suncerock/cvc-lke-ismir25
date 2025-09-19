import pandas as pd


def get_transposition(id_1, id_2, metadata):
    """
    Get the transposition between two versions of the same work in semitones.

    Parameters:
    ----------
    id_1 (str): The id of the first version of the work. E.g 'Schubert_D911-01_SC06'
    id_2 (str): The id of the second version of the work. E.g 'Schubert_D911-01_HU33'
    metadata (pd.DataFrame): The metadata DataFrame with columns 'filename' and 'key_midi'.

    Returns:
    transposition: int
    The transposition between the two versions of the work in semitones.
    """
    def key(fn): return metadata.loc[metadata['filename'] == fn, f'key_midi'].values[0]
    return key(id_1) - key(id_2)

def get_warping_path(id_1, id_2, wp_dir):
    """
    Get the warping path between two versions of the same work.
    This function handles the case where the warping path is stored in reverse order.

    Parameters:
    ----------
    id_1(str): The id of the first version of the work. E.g 'Schubert_D911-01_SC06'
    id_2(str): The id of the second version of the work. E.g 'Schubert_D911-01_HU33'
    wp_dir(Path): The directory where warping paths are stored(as exported from CH-data-synchronizer).

    Returns:
   wp: np.ndarray of shape(2, n_frames)
   The warping path between the two versions of the work.

    """
    v1 = id_1.split('_', maxsplit=2)[-1]
    v2 = id_2.split('_', maxsplit=2)[-1]
    work = '_'.join(id_1.split('_', maxsplit=2)[:-1])
    wp_fn = wp_dir / f'{work}_{v1}_{v2}.csv'
    wp_rev_fn = wp_dir / f'{work}_{v2}_{v1}.csv'
    if wp_fn.exists():
        wp = pd.read_csv(wp_fn, delimiter=';').to_numpy()
        # wp = (wp * 50).astype(int) # convert time to indices (TODO: this should be somewhere else) FIXED: this is now done in the ConsistencyEvaluator
        return wp
    elif wp_rev_fn.exists():
        wp_rev = pd.read_csv(wp_rev_fn, delimiter=';').to_numpy()
        wp = wp_rev[:, [1, 0]]  # switch channels
        # wp = (wp * 50).astype(int) # convert time to indices (TODO: this should be somewhere else) FIXED: this is now done in the ConsistencyEvaluator
        return wp
    else:
        raise FileNotFoundError(f'Neither {wp_fn} nor {wp_rev_fn} exists.')
    
def cut_warping_path(wp, array_1, array_2):
    """
    Due to inacuracies in resampling, the warping path can be longer than the original frames.
    Here we make sure to cut the warping path in a way, that the max index is the length of the original frames for both versions.

    Parameters:
    ----------
    wp(np.ndarray): The warping path between two versions of the same work.
    array_1(np.ndarray): Array for he first version of the work.
    array_2(np.ndarray): Array for the second version of the work.

    Returns:
    wp_cut: np.ndarray of shape(2, n_frames)
    The warping path cut to the length of the original frames.
    """
    wp_cut = wp[(wp[:,0] < array_1.shape[0]) & (wp[:,1] < array_2.shape[0]),:]
    return wp_cut
