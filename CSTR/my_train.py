import torchvision.transforms.transforms as tf
import my_dataset
from my_loss import LabelSmoothingCrossEntropy
from torch.optim import Adadelta
from my_scheduler import StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
from my_util import *
import random
import numpy as np
from my_converter import Averager, FCConverter

def seed_fix(random_seed=21):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def go(options, file_name):
    t = tf.Compose([tf.Resize((options.input_size.height, options.input_size.width)), tf.ToTensor()])
    train_loader, valid_loader, train_dataset, valid_dataset = my_dataset.dataset_loader(options, t)

    converter = FCConverter(train_dataset.token_to_id.keys())
    loss_avg = Averager()

    model = torch.load('stn_cstr.pth').to('cuda')

    seed_fix(options.seed)

    max_epoch = options.max_iter // len(train_loader)

    criterion = LabelSmoothingCrossEntropy()
    optimizer = Adadelta(model.parameters(), lr = options.optimizer.lr, rho= options.optimizer.rho, eps=options.optimizer.eps)
    scheduler = StepLR(optimizer = optimizer, niter_per_epoch = len(train_loader), max_epochs=max_epoch,
                       iter_based=True, milestones=[150000, 250000], warmup_epochs=0.2)
    now_acc = -1

    for epoch in range(max_epoch):

        wer = 0
        num_wer = 0
        sent_acc = 0
        num_sent_acc = 0
        loss_avg.reset()
        model.train()
        gt = set()
        pr = set()

        for batch in train_loader:
            image = batch['image'].to('cuda')
            label_input, label_len, label_target = converter.encode(batch['truth']['encoded'])

            preds = model((image,))

            cost = criterion(preds, label_target)
            
            _, seq = torch.topk(preds, 1, dim=2)

            seq = seq.squeeze(2)
            real_seq = label_input

            optimizer.zero_grad()
            cost.backward()
            clip_grad_norm_(model.parameters(), options.max_grad_norm)  # gradient clipping with 5 (Default)
            optimizer.step()

            loss_avg.add(cost)

            expected_str = id_to_string(real_seq, train_loader, do_eval=1)
            sequence_str = id_to_string(seq, train_loader, do_eval=1)

            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(seq, real_seq)
            num_sent_acc += 1
            
            if num_wer < 3:
                gt.add(expected_str[0])
                pr.add(sequence_str[0])

        print(f'Epoch : {epoch + 1}/{max_epoch} | Train_Loss : {loss_avg.val()} | Train_Sentence_Acc : {sent_acc / num_sent_acc} | Train_WER : {wer / num_wer}')
        for i, j in zip(gt, pr):
            print(f'Ground Truth : {i}')
            print(f'Predict : {j}')
            print('=' * 20)
        scheduler.iter_nums()
        scheduler.step()

        wer = 0
        num_wer = 0
        sent_acc = 0
        num_sent_acc = 0
        loss_avg.reset()
        model.eval()
        gt = set()
        pr = set()

        with torch.no_grad():
            for batch in valid_loader:
                image = batch['image'].to('cuda')
                label_input, label_len, label_target = converter.encode(batch['truth']['encoded'])

                preds = model((image, ))

                cost = criterion(preds, label_input)

                _, seq = torch.topk(preds, 1, dim=2)

                seq = seq.squeeze(2)
                real_seq = label_input

                expected_str = id_to_string(real_seq, valid_loader, do_eval=1)
                sequence_str = id_to_string(seq, valid_loader, do_eval=1)

                wer += word_error_rate(sequence_str, expected_str)
                num_wer += 1
                sent_acc += sentence_acc(seq, real_seq)
                num_sent_acc += 1
                
                if num_wer < 3:
                    gt.add(expected_str[0])
                    pr.add(sequence_str[0])

                loss_avg.add(cost)

        print(f'Validation_Loss : {loss_avg.val()} | Validation_Sentence_Acc : {sent_acc / num_sent_acc} | Validation_WER : {wer / num_wer}')
        for i, j in zip(gt, pr):
            print(f'Ground Truth : {i}')
            print(f'Predict : {j}')
            print('=' * 20)
        if sent_acc / num_sent_acc > now_acc:
            now_acc = sent_acc / num_sent_acc
            print(f'New Acc Record! Saving state dicts')
            torch.save({'model' : model.state_dict(),
                        'optim' : optimizer.state_dict()}, file_name)