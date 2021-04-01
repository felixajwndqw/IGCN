import math
import torch
from quicktorch.metrics import MetricTracker
from sklearn.metrics import jaccard_score


class SegRegMetric(MetricTracker):
    """Tracks metrics for simultaneous segmentation and regression.
    """
    def __init__(self):
        super().__init__()
        self.master_metric = "IoU"
        self.metrics = {
            "PSNR": torch.tensor(0.),
            "IoU": torch.tensor(0.),
            "RMSE": torch.tensor(0.)
        }
        self.best_metrics = self.metrics.copy()
        self.mse_fn = torch.nn.MSELoss()
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        seg_out, seg_tar = output[0], target[0]
        reg_out, reg_tar = output[1], target[1]
        mse = self.mse_fn(seg_out, seg_tar)
        pred = seg_out.clone().detach().cpu().round().flatten().numpy()
        pred = pred.clip(0, 1)
        lbl = seg_tar.clone().detach().cpu().flatten().numpy()
        # print(f'pred.max()={pred.max()}, pred.min()={pred.min()}')
        rmse = torch.sqrt(self.mse_fn(reg_out, reg_tar))
        metrics = {
            'PSNR': (
                (self.batch_count * self.metrics['PSNR'] + 10 * math.log10(1 / mse.item())) /
                (self.batch_count + 1)
            ),
            'IoU': (
                (self.batch_count * self.metrics['IoU'] + jaccard_score(lbl, pred)) /
                (self.batch_count + 1)
            ),
            'RMSE': (
                (self.batch_count * self.metrics['RMSE'] + rmse) /
                (self.batch_count + 1)
            )
        }
        return metrics