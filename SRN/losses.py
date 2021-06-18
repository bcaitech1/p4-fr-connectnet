import torch
import torch.nn as nn

class SRNLoss(nn.Module):
    def __init__(self, option, **kwargs):
        super(SRNLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index = 244)
        self.batch_size = option.batch_size
        self.max_length = option.model.max_length
        self.char_num = option.model.char_num
        
    def forward(self, predicts, label):
        predict = predicts['predict'].permute(0,2,1)
        word_predict = predicts['word_out'].permute(0,2,1)
        gsrm_predict = predicts['gsrm_out'].permute(0,2,1)

        cost_word = self.loss_func(word_predict, label)
        cost_gsrm = self.loss_func(gsrm_predict, label)
        cost_vsfd = self.loss_func(predict, label)

        cost_word = torch.reshape(torch.sum(cost_word), shape=[1])
        cost_gsrm = torch.reshape(torch.sum(cost_gsrm), shape=[1])
        cost_vsfd = torch.reshape(torch.sum(cost_vsfd), shape=[1])

        sum_cost = cost_word * 3.0 + cost_vsfd + cost_gsrm * 0.15

        return {'loss': sum_cost, 'word_loss': cost_word, 'img_loss': cost_vsfd}