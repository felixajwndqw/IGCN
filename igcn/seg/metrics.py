import math
import torch
from quicktorch.metrics import MetricTracker, iou, dice


class RCFMetric(MetricTracker):
    """Tracks metrics for dual attention features networks.
    """
    def __init__(self):
        super().__init__()
        self.master_metric = "IoU"
        self.metrics = {
            "PSNR": torch.tensor(0.),
            "IoU": torch.tensor(0.),
            "Dice": torch.tensor(0.)
        }
        self.best_metrics = self.metrics.copy()
        self.mse_fn = torch.nn.MSELoss()
        self.reset()

    def calculate(self, output, target):
        """Calculates metrics for given batch
        """
        if type(output) == tuple or type(output) == list:
            segmentations = output
            seg_pred = sum(segmentations) / len(segmentations)
        else:
            seg_pred = output

        seg_pred = seg_pred.detach()
        seg_pred = torch.sigmoid(seg_pred)
        mse = self.mse_fn(seg_pred, target)
        seg_pred = seg_pred.round().cpu().numpy().clip(0, 1).astype(int)
        target = target.round().cpu().numpy().clip(0, 1).astype(int)
        self.metrics['PSNR'] = self.batch_average(10 * math.log10(1 / mse.item()), 'PSNR')
        self.metrics['IoU'] = self.batch_average(iou(seg_pred, target), 'IoU')
        self.metrics['Dice'] = self.batch_average(dice(seg_pred, target), 'Dice')

        return self.metrics
