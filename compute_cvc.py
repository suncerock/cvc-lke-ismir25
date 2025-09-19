import os
from itertools import combinations

import numpy as np

from pathlib import Path
from cvc_utils.consistency_evaluator import ConsistencyEvaluator
from cvc_utils.evaluators import recall, tvd


wp_dir = Path('../ch-data-synchronizer/swd/wp_a2a_5fps')
metadata = Path('../ch-data-synchronizer/swd/metadata.csv')
predictions_dir = Path('../inference_output/swd_octave/1/predictions')
annotations_dir = Path('../inference_output/swd_octave/1/annotations')

def transpose(arr, shift):
    return np.roll(arr, shift=shift*2, axis=1)

evaluator = ConsistencyEvaluator(predictions_dir, annotations_dir, wp_dir, metadata, model_name="test", sr=5,
                                 transpose_function=transpose,
                                 local_similarity_measure=tvd)

def get_lpc_acc_one_output(output_dir, model_name):
    # Time Complexity: O(W(V^2 + V))
    predictions_dir = output_dir / 'predictions'
    annotations_dir = output_dir / 'annotations'

    filenames = [filename.replace(".npy", "") for filename in os.listdir(predictions_dir) if filename.endswith(".npy")]
    versions = list(set([filename.split("_", maxsplit=2)[-1] for filename in filenames]))
    works = list(set(["_".join(filename.split("_", maxsplit=2)[:-1]) for filename in filenames]))

    evaluator = ConsistencyEvaluator(predictions_dir, annotations_dir, wp_dir, metadata, model_name=model_name, sr=5,
                                     transpose_function=transpose,
                                     local_similarity_measure=tvd, global_evaluation_measure=recall)

    all_lpc = []
    all_acc = []

    for work in works:
        lpc = []
        acc = []

        for v1, v2 in combinations(versions, 2):
            id_1 = f'{work}_{v1}'
            id_2 = f'{work}_{v2}'

            lpc.append(evaluator.prediction_similarity_framewise(id_1, id_2).mean())

        lpc = sum(lpc) / len(lpc)

        for version in versions:
            id = f'{work}_{version}'
            if not id in filenames:
                continue
            annotations = np.load(annotations_dir / f'{id}.npy')
            predictions = np.load(predictions_dir / f'{id}.npy')
            acc.append(evaluator.global_evaluation_measure(annotations, predictions))

        acc = sum(acc) / len(acc)

        all_lpc.append(lpc)
        all_acc.append(acc)

    return all_lpc, all_acc

all_lpc, all_acc = get_lpc_acc_one_output(Path('inference_output/swd_octave/1/best-epoch=15-step=160-loss=2.2736-recall=0.1254'), 'octave')
print(f'Local Prediction Consistency: {np.mean(all_lpc):.4f}')
print(f'Recall: {np.mean(all_acc):.4f}')