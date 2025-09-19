import json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from .utils import get_warping_path, cut_warping_path, get_transposition


class ConsistencyEvaluator:
    def __init__(self,
                 predictions_dir: Path,
                 annotations_dir: Path,
                 wp_dir: Path,
                 metadata: Path,

                 transpose_function,

                 local_similarity_measure=None,
                 global_evaluation_measure=None,

                 sr=5,
                 model_name='', 
                 base_results_dir=Path('./results')):
        """
        Initialize the ConsistencyEvaluator.

        Parameters
        ----------
        predictions_dir: Path
            The directory where the predictions are stored as npy files (named as the id of the track).
        annotations_dir: Path
            The directory where the annotations are stored as npy files.
        wp_dir: Path
            The directory where the warping paths are stored as csv (output of ch-data-synchronizer).
        metadata: Path
            The metadata file with columns 'work' and 'version' as csv.

        transpose_function: function
            The function to use for transposing the predictions.

        local_similarity_measure: function
            A frame-wise similarity measure to use for the local prediction consistency.
        global_evaluation_measure: function
            A global evaluation measure to use for the global evaluation consistency.

        sr: int
            The sample rate of predictions, annotations and warping paths.
        model_name: str
            The name of the model to use in the results.
        base_results_dir: Path
            The base directory to store the results in.

        """
        # set paths and metadata
        self.predictions_dir = predictions_dir
        self.annotations_dir = annotations_dir
        self.wp_dir = wp_dir
        self.metadata = pd.read_csv(metadata)
        # set measures, transpose function and sample rate
        self.local_similarity_measure = local_similarity_measure
        self.global_evaluation_measure = global_evaluation_measure

        self.transpose = transpose_function
        self.sr = sr
        # set model name and visualizer
        self.model_name = model_name

        # get and set all works and versions
        self.all_works = list(self.metadata.work.unique())
        self.all_works.sort()
        self.all_versions = list(self.metadata.version.unique())
        self.all_versions.sort()
    
        # create results dir and save config in results_dir
        self.results_dir = base_results_dir / model_name.lower().replace(' ', '_')
        self.results_dir.mkdir(exist_ok=True, parents=True)
        config = {
            'predictions_dir': predictions_dir.as_posix(),
            'annotations_dir': annotations_dir.as_posix(),
            'metadata': metadata.as_posix(),
            'wp_dir': wp_dir.as_posix(),        
        }
        with open(self.results_dir / 'config.json', 'w') as f:
            json.dump(config, f)
    
    # local prediction consistency
    def prediction_similarity_framewise(self, id_1, id_2, enforce_recompute=True):
        """
        Compute the framewise similarity between the predictions of two versions of a work.

        Parameters
        ----------
        id_1: str
            The id of the first version of the work. E.g 'Schubert_D911-01_SC06'
        id_2: str
            The id of the second version of the work. E.g 'Schubert_D911-01_SC07'

        Returns
        -------
        similarity: np.array of shape(n_frames,)
            The similarities between the predictions of the two versions along the warping path axis.

        """
        if self.local_similarity_measure is None:
            raise ValueError('No similarity measure defined')
        
        prediction_v1 = np.load(self.predictions_dir / f'{id_1}.npy')
        prediction_v2 = np.load(self.predictions_dir / f'{id_2}.npy')
        transposition = get_transposition(id_1, id_2, self.metadata)
        prediction_v2 = self.transpose(prediction_v2, transposition)
        wp = get_warping_path(id_1, id_2, self.wp_dir)
        wp = np.round(wp * self.sr).astype(int)  # convert the wp (wall time) to indices - this might lead to resampling if sr does not correpond well to wp fps
        wp = cut_warping_path(wp, prediction_v1, prediction_v2)

        result_fn = f'{self.local_similarity_measure.__name__}_{id_1}_{id_2}.npy'
        result_subdir = self.results_dir / 'prediction_similarity'
        if (result_subdir/ result_fn).exists() and not enforce_recompute:
            similarity = np.load(result_subdir / result_fn)
        else: 
            result_subdir.mkdir(exist_ok=True)
            similarity = [
                self.local_similarity_measure(prediction_v1[int(index_1)], prediction_v2[int(index_2)])
                for index_1, index_2 in wp
            ]
            similarity = np.array(similarity)
            np.save(result_subdir/ result_fn, similarity)

        return similarity

    def local_prediction_consistency_one_work(self, work, temp_agg=lambda x: np.mean(x)):
        """
       Compute the pairwise prediction similarity between all version pairs of a work.

        Parameters
        ----------
        work: str
            The work to compute the similarity matrix for . E.g 'Schubert_D911-01'
        temp_agg: function
            The temporal aggregation function to use. Default is np.mean

        Returns
        -------
        similarity_matrix: np.array of shape(n_versions, n_versions)
            The similarity matrix between the predictions of all versions of the work.
        """

        versions = list(self.metadata[self.metadata['work'] == work]['version'])
        similarity_matrix = np.full([len(versions), len(versions)], np.nan)
        
        for (index_v1, v1), (index_v2, v2) in combinations(enumerate(versions), 2):
                id_1 = f'{work}_{v1}'
                id_2 = f'{work}_{v2}'
                framewise_similarity = self.prediction_similarity_framewise(id_1, id_2)
                similarity_matrix[index_v1, index_v2] = temp_agg(framewise_similarity)

        return similarity_matrix

    def local_prediction_consistency(self, temp_agg=lambda x: np.mean(x)):
        """
        Compute the local prediction consistency across versions over all works.

        Parameters
        ----------
        temp_agg: function
            The temporal aggregation function to use for the local consistency. Default is np.mean

        Returns
        -------
        lpc: float
            The local prediction consistency across all works.
        """
        print(f'Computing local prediction consistency across versions for {len(self.all_works)} works')
        # first dimension is the work, second dimension is the versions * versions
        lpc_works = np.zeros([len(self.all_works), len(self.all_versions), len(self.all_versions)])
        for index, work in tqdm(enumerate(self.all_works), desc='Computing LPC', total=len(self.all_works)):
            lpc_one_work = self.local_prediction_consistency_one_work(work, temp_agg)
            lpc_works[index] = lpc_one_work
        lpc = np.mean(lpc_works, where=~np.isnan(lpc_works))
        return lpc
