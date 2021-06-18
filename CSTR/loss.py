import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps: float = 0.1, reduction='mean', ignore_index=2):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps, self.reduction = eps, reduction
        self.ignore_index = ignore_index

    def forward(self, output, target, *args):
        pred = output[:, :-1, :].contiguous().view(-1, output.shape[-1])
        target = target.to(pred.device).contiguous().view(-1)

        c = pred.size()[-1]
        
        log_preds = F.log_softmax(pred, dim=-1)

        # ignore index for smooth label        
        ignore_target = target != self.ignore_index

        log_preds = log_preds * ignore_target[:, None]

        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * \
               F.nll_loss(log_preds, target, reduction=self.reduction,
                          ignore_index=self.ignore_index)
