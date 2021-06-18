import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import torchvision
import timm
from dataset import START, PAD
from .dcn import DeformableConv2d
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AsterBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_deform=False):
        super(AsterBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if is_deform:
            self.conv2 = DeformableConv2d(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet_ASTER(nn.Module):

    def __init__(self, in_channel, with_lstm=True, n_group=1):
        super(ResNet_ASTER, self).__init__()
        self.with_lstm = with_lstm
        self.n_group = n_group

        in_channels = in_channel
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.inplanes = 32
        self.layer1 = self._make_layer(32,  3, [2, 2]) # [16, 50]
        self.layer2 = self._make_layer(64,  4, [2, 2]) # [8, 25]
        self.layer3 = self._make_layer(128, 6, [2, 1]) # [4, 25]
        self.layer4 = self._make_layer(256, 6, [2, 1]) # [2, 25]
        self.layer5 = self._make_layer(512, 3, [2, 1], is_deform = True) # [1, 25]

        if with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
            self.out_planes = 2 * 256
        else:
            self.out_planes = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, is_deform = False):
        downsample = None
        if stride != [1, 1] or self.inplanes != planes:
            downsample = nn.Sequential(
              conv1x1(self.inplanes, planes, stride),
              nn.BatchNorm2d(planes))

        layers = []
        layers.append(AsterBlock(self.inplanes, planes, stride, downsample, is_deform))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(AsterBlock(self.inplanes, planes, is_deform=is_deform))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x) #(B, 32, H, W)
        x1 = self.layer1(x0) #(B, 32, H/2, W/2)
        x2 = self.layer2(x1) #(B, 64, H/4, W/4)
        x3 = self.layer3(x2) #(B, 128, H/8, W/4)
        x4 = self.layer4(x3) #(B, 256, H/16, W/4)
        x5 = self.layer5(x4) #(B, 512, H/32, W/4)

        #return x5
        cnn_feat = x5.squeeze(2) # [N, c, w]
        cnn_feat = cnn_feat.transpose(2, 1)
        if self.with_lstm:
            b,c,h,w = x5.size()
            x5 = x5.view(b,c,h*w).transpose(1,2)
            rnn_feat, _ = self.rnn(x5)
            return rnn_feat #(B, H/32 * W/4, 512)
        else:
            return cnn_feat

class EffNet(nn.Module):
    def __init__(self, nc, with_lstm=True):
        super(EffNet, self).__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
        self.backbone.conv_stem = nn.Conv2d(nc, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone.conv_head = nn.Conv2d(320, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        self.backbone.blocks[5][0].conv_dw = DeformableConv2d(672, 672, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.blocks[5][1].conv_dw = DeformableConv2d(1152, 1152, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.blocks[5][2].conv_dw = DeformableConv2d(1152, 1152, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.blocks[5][3].conv_dw = DeformableConv2d(1152, 1152, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.backbone.bn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.classifier = nn.Linear(in_features=512, out_features=1000, bias=True)
        self.with_lstm = with_lstm
        if with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        
    def forward(self, x):
        x = self.backbone.forward_features(x)
        if self.with_lstm:
            b,c,h,w = x.size()
            x = x.view(b,c,h*w).transpose(1,2)
            rnn_feat, _ = self.rnn(x)
            return rnn_feat 
        else:
            return x
        #return x


class CNN(nn.Module):
    def __init__(self, nc, leakyRelu=False, with_lstm=True):
        super(CNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]



        def convRelu(i, batchNormalization=False):
            cnn = nn.Sequential()
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]

            if i< 4:
                cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            else:
                cnn.add_module('dcv{0}'.format(i),
                            DeformableConv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            # cnn.add_module('conv{0}'.format(i),
            #                nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))


            return cnn

        self.with_lstm = with_lstm
        self.conv0 = convRelu(0)
        self.pooling0 = nn.MaxPool2d(2, 2)
        self.conv1 = convRelu(1)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.conv2 = convRelu(2, True)
        self.conv3 = convRelu(3)
        self.pooling3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4 = convRelu(4, True)
        self.conv5 = convRelu(5)
        self.pooling5 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv6 = convRelu(6, True)
        if with_lstm:
            self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)

    def forward(self, input):
        out = self.conv0(input)     # [batch size, 64, 128, 128]
        out = self.pooling0(out)    # [batch size, 64, 64, 64]
        out = self.conv1(out)       # [batch size, 128, 64, 64]
        out = self.pooling1(out)    # [batch size, 128, 32, 32]
        out = self.conv2(out)       # [batch size, 256, 32, 32]
        out = self.conv3(out)       # [batch size, 256, 32, 32]
        out = self.pooling3(out)    # [batch size, 256, 16, 33]
        out = self.conv4(out)       # [batch size, 512, 16, 33]
        out = self.conv5(out)       # [batch size, 512, 16, 33]
        out = self.pooling5(out)    # [batch size, 512, 8, 34]
        out = self.conv6(out)       # [batch size, 512, 7, 33]

        if self.with_lstm:
            b,c,h,w = out.size()
            out = out.view(b,c,h*w).transpose(1,2)
            rnn_feat, _ = self.rnn(out)
            return rnn_feat 
        else:
            return out

class AttentionCell(nn.Module):
    def __init__(self, src_dim, hidden_dim, embedding_dim, num_layers=1, cell_type='LSTM'):
        super(AttentionCell, self).__init__()
        self.num_layers = num_layers

        self.i2h = nn.Linear(src_dim, hidden_dim, bias=False)
        self.h2h = nn.Linear(
            hidden_dim, hidden_dim
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        if num_layers == 1:
            if cell_type == 'LSTM':
                self.rnn = nn.LSTMCell(src_dim + embedding_dim, hidden_dim)
            elif cell_type == 'GRU':
                self.rnn = nn.GRUCell(src_dim + embedding_dim, hidden_dim)
            else:
                raise NotImplementedError
        else:
            if cell_type == 'LSTM':
                self.rnn = nn.ModuleList(
                    [nn.LSTMCell(src_dim + embedding_dim, hidden_dim)]
                    + [
                        nn.LSTMCell(hidden_dim, hidden_dim)
                        for _ in range(num_layers - 1)
                    ]
                )
            elif cell_type == 'GRU':
                self.rnn = nn.ModuleList(
                    [nn.GRUCell(src_dim + embedding_dim, hidden_dim)]
                    + [
                        nn.GRUCell(hidden_dim, hidden_dim)
                        for _ in range(num_layers - 1)
                    ]
                )
            else:
                raise NotImplementedError

        self.hidden_dim = hidden_dim

    def forward(self, prev_hidden, src, tgt):   # src: [b, L, c]
        src_features = self.i2h(src)  # [b, L, h]
        if self.num_layers == 1:
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)    # [b, 1, h]
        else:
            prev_hidden_proj = self.h2h(prev_hidden[-1][0]).unsqueeze(1)    # [b, 1, h]
        attention_logit = self.score(
            torch.tanh(src_features + prev_hidden_proj) # [b, L, h]
        )  # [b, L, 1]
        alpha = F.softmax(attention_logit, dim=1)  # [b, L, 1]
        context = torch.bmm(alpha.permute(0, 2, 1), src).squeeze(1)  # [b, c]

        concat_context = torch.cat([context, tgt], 1)  # [b, c+e]

        if self.num_layers == 1:
            cur_hidden = self.rnn(concat_context, prev_hidden)
        else:
            cur_hidden = []
            for i, layer in enumerate(self.rnn):
                if i == 0:
                    concat_context = layer(concat_context, prev_hidden[i])
                else:
                    concat_context = layer(concat_context[0], prev_hidden[i])
                cur_hidden.append(concat_context)

        return cur_hidden, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        src_dim,
        embedding_dim,
        hidden_dim,
        pad_id,
        st_id,
        num_layers=1,
        cell_type='LSTM',
        checkpoint=None,
    ):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.attention_cell = AttentionCell(
            src_dim, hidden_dim, embedding_dim, num_layers, cell_type
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.generator = nn.Linear(hidden_dim, num_classes)
        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(
        self, src, text, is_train=True, teacher_forcing_ratio=1.0, batch_max_length=50
    ):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [START] token. text[:, 0] = [START].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = src.size(0)
        num_steps = batch_max_length - 1  # +1 for [s] at end of sentence.

        output_hiddens = (
            torch.FloatTensor(batch_size, num_steps, self.hidden_dim)
            .fill_(0)
            .to(device)
        )
        if self.num_layers == 1:
            hidden = (
                torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
            )
        else:
            hidden = [
                (
                    torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                    torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                )
                for _ in range(self.num_layers)
            ]

        if is_train and random.random() < teacher_forcing_ratio:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                embedd = self.embedding(text[:, i])
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, src, embedd)
                if self.num_layers == 1:
                    output_hiddens[:, i, :] = hidden[
                        0
                    ]  # LSTM hidden index (0: hidden, 1: Cell)
                else:
                    output_hiddens[:, i, :] = hidden[-1][0]
            probs = self.generator(output_hiddens)

        else:
            targets = (
                torch.LongTensor(batch_size).fill_(self.st_id).to(device)
            )  # [START] token
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.num_classes)
                .fill_(0)
                .to(device)
            )

            for i in range(num_steps):
                embedd = self.embedding(targets)
                hidden, alpha = self.attention_cell(hidden, src, embedd)
                if self.num_layers == 1:
                    probs_step = self.generator(hidden[0])
                else:
                    probs_step = self.generator(hidden[-1][0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class Attention(nn.Module):
    def __init__(
        self,
        FLAGS,
        train_dataset,
        checkpoint=None,
    ):
        super(Attention, self).__init__()
        
        self.encoder = CNN(FLAGS.data.rgb)
        #self.encoder = EffNet(FLAGS.data.rgb)
        #self.encoder = CPNet()
        #self.encoder = ResNet_ASTER(FLAGS.data.rgb)

        self.decoder = AttentionDecoder(
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.Attention.src_dim,
            embedding_dim=FLAGS.Attention.embedding_dim,
            hidden_dim=FLAGS.Attention.hidden_dim,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            num_layers=FLAGS.Attention.layer_num,
            cell_type=FLAGS.Attention.cell_type)

        self.criterion = (
            nn.CrossEntropyLoss(ignore_index=train_dataset.token_to_id[PAD])
        )

        if checkpoint:
            self.load_state_dict(checkpoint)
    
    def forward(self, input, expected, is_train, teacher_forcing_ratio):
        enc_result = self.encoder(input)
        #print(out.shape)
        #b, c, h, w = enc_result.size()
        #enc_result = enc_result.view(b, c, h * w).transpose(1, 2)  # [b, h x w, c]
        dec_result = self.decoder(enc_result, expected, is_train, teacher_forcing_ratio, batch_max_length=expected.size(1))    # [b, sequence length, class size]

        result = {"dec_result": dec_result, "enc_result": enc_result}
        return result