import torch
from torchmetrics import Metric

class LKEMetric(Metric):
    KEYS = [
        "A:maj", "A:min", "A#:maj", "A#:min", "B:maj", "B:min", "C:maj", "C:min",
        "C#:maj", "C#:min", "D:maj", "D:min", "D#:maj", "D#:min", "E:maj", "E:min",
        "F:maj", "F:min", "F#:maj", "F#:min", "G:maj", "G:min", "G#:maj", "G#:min"
    ]
    MAJOR_MODE = 0
    MINOR_MODE = 1
    NUM_MODES = 2
    def __init__(self) -> None:
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fifth", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("parallel", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("relative", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fifth_parallel", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fifth_relative", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("others", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y: torch.Tensor):
        assert len(y_pred) == len(y) == 1  # Only support song-level aggregation

        y_pred = y_pred.argmax(dim=1)

        # Filter out nan labels
        mask = y != -1
        y = y[mask]  # (seq_len, )
        y_pred = y_pred[mask]  # (seq_len, )

        pitch_pred = y_pred // self.NUM_MODES
        mode_pred = y_pred % self.NUM_MODES
        pitch_gt = y // self.NUM_MODES
        mode_gt = y % self.NUM_MODES

        correct, fifth, parallel, relative, fifth_parallel, fifth_relative = 0, 0, 0, 0, 0, 0
        correct += torch.count_nonzero((pitch_pred == pitch_gt) * (mode_pred == mode_gt))

        fifth += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 5) * (mode_pred == mode_gt))
        fifth += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 7) * (mode_pred == mode_gt))

        parallel += torch.count_nonzero((pitch_pred == pitch_gt) * (mode_pred != mode_gt))

        relative += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 3) * (mode_pred == self.MAJOR_MODE) * (mode_gt == self.MINOR_MODE))
        relative += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 9) * (mode_pred == self.MINOR_MODE) * (mode_gt == self.MAJOR_MODE))

        fifth_parallel += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 5) * (mode_pred != mode_gt))
        fifth_parallel += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 7) * (mode_pred != mode_gt))

        fifth_relative += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 8) * (mode_pred == self.MAJOR_MODE) * (mode_gt == self.MINOR_MODE))
        fifth_relative += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 10) * (mode_pred == self.MAJOR_MODE) * (mode_gt == self.MINOR_MODE))
        fifth_relative += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 2) * (mode_pred == self.MINOR_MODE) * (mode_gt == self.MAJOR_MODE))
        fifth_relative += torch.count_nonzero(((pitch_pred - pitch_gt) % 12 == 4) * (mode_pred == self.MINOR_MODE) * (mode_gt == self.MAJOR_MODE))

        others = len(y_pred) - correct - fifth - parallel - relative - fifth_parallel - fifth_relative

        self.correct += correct / len(y_pred)
        self.fifth += fifth / len(y_pred)
        self.parallel += parallel / len(y_pred)
        self.relative += relative / len(y_pred)
        self.fifth_parallel += fifth_parallel / len(y_pred)
        self.fifth_relative += fifth_relative / len(y_pred)
        self.others += others / len(y_pred)
        self.count += 1
    
    def compute(self):
        return {
            "correct": self.correct / self.count * 100,
            "fifth": self.fifth / self.count * 100,
            "parallel": self.parallel / self.count * 100,
            "relative": self.relative / self.count * 100,
            "fifth_parallel": self.fifth_parallel / self.count * 100,
            "fifth_relative": self.fifth_relative / self.count * 100,
            "others": self.others / self.count * 100,
        }