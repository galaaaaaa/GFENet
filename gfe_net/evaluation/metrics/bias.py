# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metrics based on pixels."""

from mmagic.registry import METRICS
from mmagic.evaluation.metrics.base_sample_wise_metric import BaseSampleWiseMetric
import numpy as np

@METRICS.register_module()
class BIAS(BaseSampleWiseMetric):
    """Statistical indicator: Bias for LST image.

    mean(a-b)

    Args:

        gt_key (str): Key of ground-truth. Default: 'gt_img'
        pred_key (str): Key of prediction. Default: 'pred_img'
        mask_key (str, optional): Key of mask, if mask_key is None, calculate
            all regions. Default: None
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None

    Metrics:
        - BIAS (float): mean(a-b)
    """

    metric = "BIAS"

    def process_image(self, gt, pred, mask=None):
        """Process an image.

        Args:
            gt (Torch | np.ndarray): GT image.
            pred (Torch | np.ndarray): Pred image.
            mask (Torch | np.ndarray): Mask of evaluation.
        Returns:
            result (np.ndarray): BIAS result.
        """

        # mean(a-b)
        # diff = gt - pred
        diff = pred - gt

        if self.mask_key is not None:
            diff *= mask
            result = diff.sum() / (mask.sum() + 1e-12)
        else:
            result = diff.mean()

        return result

@METRICS.register_module()
class LSTPSNR(BaseSampleWiseMetric):
    metric = "PSNR"

    def process_image(self, gt, pred, mask=None):
        diff = gt - pred
        if self.mask_key is not None:
            mse = ((diff ** 2) * mask).sum() / (mask.sum() + 1e-12)
            data_range = gt.max() - gt.min()
        else:
            mse = (diff ** 2).mean()
            data_range = gt.max() - gt.min()
        data_range = float(data_range)
        if data_range <= 1e-12:
            data_range = 1.0
        return 20 * np.log10(data_range + 1e-12) - 10 * np.log10(mse + 1e-12)

@METRICS.register_module()
class LSTSSIM(BaseSampleWiseMetric):
    metric = "SSIM"

    def process_image(self, gt, pred, mask=None):
        L = float(gt.max() - gt.min())
        if L <= 1e-12:
            L = 1.0
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
        if self.mask_key is not None:
            n = mask.sum()
            g = gt * mask
            p = pred * mask
            mu_g = g.sum() / (n + 1e-12)
            mu_p = p.sum() / (n + 1e-12)
            g[mask == 0] = mu_g
            p[mask == 0] = mu_p
            var_g = ((g - mu_g) ** 2).sum() / (n + 1e-12)
            var_p = ((p - mu_p) ** 2).sum() / (n + 1e-12)
            cov = (((g - mu_g) * (p - mu_p)).sum()) / (n + 1e-12)
        else:
            mu_g = gt.mean()
            mu_p = pred.mean()
            var_g = ((gt - mu_g) ** 2).mean()
            var_p = ((pred - mu_p) ** 2).mean()
            cov = ((gt - mu_g) * (pred - mu_p)).mean()
        num = (2 * mu_g * mu_p + C1) * (2 * cov + C2)
        den = (mu_g ** 2 + mu_p ** 2 + C1) * (var_g + var_p + C2)
        return num / (den + 1e-12)

@METRICS.register_module()
class SAM(BaseSampleWiseMetric):
    metric = "SAM"

    def process_image(self, gt, pred, mask=None):
        if self.mask_key is not None:
            g = gt * mask
            p = pred * mask
        else:
            g = gt
            p = pred
        g_vec = g.reshape(-1)
        p_vec = p.reshape(-1)
        dot = float((g_vec * p_vec).sum())
        norm_g = np.sqrt((g_vec ** 2).sum())
        norm_p = np.sqrt((p_vec ** 2).sum())
        cos = dot / (norm_g * norm_p + 1e-12)
        cos = np.clip(cos, -1.0, 1.0)
        return np.arccos(cos)

@METRICS.register_module()
class ERGAS(BaseSampleWiseMetric):
    metric = "ERGAS"

    def process_image(self, gt, pred, mask=None):
        eps = 1e-12
        if gt.ndim == 3:
            C = gt.shape[0]
        else:
            C = 1
        def ch_view(x, c):
            if C == 1:
                return x
            return x[c]
        vals = []
        for c in range(C):
            g = ch_view(gt, c)
            p = ch_view(pred, c)
            if self.mask_key is not None:
                m = mask if C == 1 else mask
                diff = (g - p) * m
                mse = (diff ** 2).sum() / (m.sum() + eps)
                mu = g.sum() / (m.sum() + eps)
            else:
                diff = g - p
                mse = (diff ** 2).mean()
                mu = g.mean()
            rmse = np.sqrt(mse)
            vals.append((rmse / (abs(mu) + eps)) ** 2)
        return 100.0 * np.sqrt(np.mean(vals))

@METRICS.register_module()
class SCC(BaseSampleWiseMetric):
    metric = "SCC"

    def process_image(self, gt, pred, mask=None):
        def grad_mag(x):
            gx = np.zeros_like(x)
            gy = np.zeros_like(x)
            gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
            gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
            return np.sqrt(gx ** 2 + gy ** 2)
        g = grad_mag(gt)
        p = grad_mag(pred)
        if self.mask_key is not None:
            m = mask
            g = g * m
            p = p * m
            n = m.sum()
            mu_g = g.sum() / (n + 1e-12)
            mu_p = p.sum() / (n + 1e-12)
            g[m == 0] = mu_g
            p[m == 0] = mu_p
            var_g = ((g - mu_g) ** 2).sum() / (n + 1e-12)
            var_p = ((p - mu_p) ** 2).sum() / (n + 1e-12)
            cov = (((g - mu_g) * (p - mu_p)).sum()) / (n + 1e-12)
        else:
            mu_g = g.mean()
            mu_p = p.mean()
            var_g = ((g - mu_g) ** 2).mean()
            var_p = ((p - mu_p) ** 2).mean()
            cov = ((g - mu_g) * (p - mu_p)).mean()
        return cov / (np.sqrt(var_g * var_p) + 1e-12)

@METRICS.register_module()
class Q(BaseSampleWiseMetric):
    metric = "Q"

    def process_image(self, gt, pred, mask=None):
        if self.mask_key is not None:
            m = mask
            g = gt * m
            p = pred * m
            n = m.sum()
            mu_g = g.sum() / (n + 1e-12)
            mu_p = p.sum() / (n + 1e-12)
            g[m == 0] = mu_g
            p[m == 0] = mu_p
            var_g = ((g - mu_g) ** 2).sum() / (n + 1e-12)
            var_p = ((p - mu_p) ** 2).sum() / (n + 1e-12)
            cov = (((g - mu_g) * (p - mu_p)).sum()) / (n + 1e-12)
        else:
            mu_g = gt.mean()
            mu_p = pred.mean()
            var_g = ((gt - mu_g) ** 2).mean()
            var_p = ((pred - mu_p) ** 2).mean()
            cov = ((gt - mu_g) * (pred - mu_p)).mean()
        num = 4 * cov * mu_g * mu_p
        den = (var_g + var_p) * (mu_g ** 2 + mu_p ** 2)
        return num / (den + 1e-12)

@METRICS.register_module()
class DLambda(BaseSampleWiseMetric):
    metric = "D_LAMBDA"

    def process_image(self, gt, pred, mask=None):
        eps = 1e-12
        if gt.ndim != 3:
            return 0.0
        C = gt.shape[0]
        if C < 2:
            return 0.0
        def corr(a, b, m):
            if m is not None:
                n = m.sum()
                aa = a * m
                bb = b * m
                mu_a = aa.sum() / (n + eps)
                mu_b = bb.sum() / (n + eps)
                aa[m == 0] = mu_a
                bb[m == 0] = mu_b
                var_a = ((aa - mu_a) ** 2).sum() / (n + eps)
                var_b = ((bb - mu_b) ** 2).sum() / (n + eps)
                cov = (((aa - mu_a) * (bb - mu_b)).sum()) / (n + eps)
            else:
                mu_a = a.mean()
                mu_b = b.mean()
                var_a = ((a - mu_a) ** 2).mean()
                var_b = ((b - mu_b) ** 2).mean()
                cov = ((a - mu_a) * (b - mu_b)).mean()
            return cov / (np.sqrt(var_a * var_b) + eps)
        s = 0.0
        count = 0
        m = mask if self.mask_key is not None else None
        for i in range(C):
            for j in range(i + 1, C):
                rho_gt = corr(gt[i], gt[j], m)
                rho_pr = corr(pred[i], pred[j], m)
                s += (rho_pr - rho_gt) ** 2
                count += 1
        return np.sqrt((2.0 / (C * (C - 1))) * s)

@METRICS.register_module()
class DS(BaseSampleWiseMetric):
    metric = "DS"

    def process_image(self, gt, pred, mask=None):
        eps = 1e-12
        if gt.ndim == 3:
            C = gt.shape[0]
        else:
            C = 1
        def grad_mag(x):
            gx = np.zeros_like(x)
            gy = np.zeros_like(x)
            gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
            gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
            return np.sqrt(gx ** 2 + gy ** 2)
        vals = []
        for c in range(C):
            g = gt if C == 1 else gt[c]
            p = pred if C == 1 else pred[c]
            ggm = grad_mag(g)
            pgm = grad_mag(p)
            if self.mask_key is not None:
                m = mask
                ggm = ggm * m
                pgm = pgm * m
                n = m.sum()
                mu_g = ggm.sum() / (n + eps)
                mu_p = pgm.sum() / (n + eps)
                ggm[m == 0] = mu_g
                pgm[m == 0] = mu_p
                var_g = ((ggm - mu_g) ** 2).sum() / (n + eps)
                var_p = ((pgm - mu_p) ** 2).sum() / (n + eps)
                cov = (((ggm - mu_g) * (pgm - mu_p)).sum()) / (n + eps)
            else:
                mu_g = ggm.mean()
                mu_p = pgm.mean()
                var_g = ((ggm - mu_g) ** 2).mean()
                var_p = ((pgm - mu_p) ** 2).mean()
                cov = ((ggm - mu_g) * (pgm - mu_p)).mean()
            rho = cov / (np.sqrt(var_g * var_p) + eps)
            vals.append((rho - 1.0) ** 2)
        return np.sqrt(np.mean(vals))

@METRICS.register_module()
class QNR(BaseSampleWiseMetric):
    metric = "QNR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dl_metric = DLambda(prefix='qnr_dl')
        self.ds_metric = DS(prefix='qnr_ds')

    def process_image(self, gt, pred, mask=None):
        dl = self.dl_metric.process_image(gt, pred, mask)
        ds = self.ds_metric.process_image(gt, pred, mask)
        qnr = (1.0 - dl) * (1.0 - ds)
        return np.clip(qnr, 0.0, 1.0)
