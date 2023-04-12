import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, logits, gts, mask):
        if mask is not None:
            return torch.mean(self.criterion(logits, gts) * (mask.unsqueeze(-1) + 1)) * 100
        else:
            return torch.mean(self.criterion(logits, gts)) * 100


#
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.
    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.,
                 ):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target, mask):
        """Criterion of wingloss.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            mask (torch.Tensor[NxK]): mask some channel
        """
        delta = (target - pred).abs()
        _, _, h, w = target.shape
        A = self.omega * (
                1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, h, w)
            mask_bool = torch.where(mask == 1., True, False)
            losses = losses[mask_bool]
        else:
            losses = losses
        return torch.mean(losses)

    def forward(self, output, target, target_weight=None, mask=None):
        """Forward function.
        Note:
            batch_size: N
            num_keypoints: K
        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            loss = self.criterion(output * target_weight.unsqueeze(-1),
                                  target * target_weight.unsqueeze(-1), mask)
        else:
            loss = self.criterion(output, target, mask)

        return loss * self.loss_weight


class FocalHeatmapLoss(nn.Module):

    def __init__(self, alpha=2, beta=4):
        super(FocalHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask=None):
        """Modified focal loss.
        Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
          pred (batch x c x h x w)
          gt_regr (batch x c x h x w)
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        if mask is not None:
            pos_inds = pos_inds * mask
            neg_inds = neg_inds * mask

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(
            pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


#
# class PeakLoss(nn.Module):
#     def __init__(self, H, W):
#         super(PeakLoss, self).__init__()
#         self.criterion = nn.CrossEntropyLoss(reduction='none')
#         self.find_max = nn.MaxPool1d(kernel_size=H * W, stride=H * W, return_indices=True)
#
#     def forward(self, pred, heatmap, mask=None):
#         """
#         :argument pred: heatmap: bsize x K x H x W
#         """
#         b, k, h, w = heatmap.shape
#         heatmap = heatmap.view(b, k, -1)  # bsize x K x (HW)
#         _, index = self.find_max(heatmap)
#         index = index[:, :-1, :].squeeze(-1)  # bsize x (K-1)
#         pred_flat = pred[:, :-1, :].view(b, k - 1, -1).permute(0, 2, 1).contiguous()  # bsize x (HW) x (K-1)
#         # pred_flat = torch.softmax(pred_flat, dim=2)
#         loss = self.criterion(pred_flat, index)  # bsize x(k-1)
#         # print(loss.shape, mask[:, :-1, :].shape)
#         if mask is not None:
#             mask_bool = torch.where(mask[:, :-1] == 1., True, False)
#             loss = loss[mask_bool]
#         return torch.mean(loss)


class PeakLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, gt, mask=None):
        """

        :param pred: bs, c, h, w
        :param gt: bs, c, h, w
        :return:
        """
        pred = pred[:, :-1, :, :]
        gt = gt[:, :-1, :, :]
        bs, c, h, w = pred.shape
        gt_flatten = gt.view(bs, c, -1)
        gt_max = torch.argmax(gt_flatten, dim=-1)
        pmf = torch.softmax(pred.view(bs, c, -1), dim=-1)  # bs, c, h*w
        c10 = torch.gather(pmf, dim=-1, index=gt_max.view(bs, c, 1)).squeeze(-1)
        cross_entropy = -torch.log(c10)  # bs, c
        if mask is not None:
            mask_bool = torch.where(mask[:, :-1] == 1., True, False)
            cross_entropy = cross_entropy[mask_bool]
        return cross_entropy.mean()


if __name__ == "__main__":
    l = AdaptiveWingLoss()
    a = torch.randn((8, 20, 128, 128))
    b = torch.randn((8, 20, 128, 128))
    b = torch.softmax(b, dim=1)
    print(l(a, b))
