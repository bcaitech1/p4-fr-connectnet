import net
import losses
import torch.optim
from my_utils import *
import time
from tqdm import tqdm
from psutil import virtual_memory

def go(opt, train_loader, val_loader, model, file_name, device):
    epochs = opt.num_epochs
    learning_rate = opt.optimizer.lr
    weight_decay = opt.optimizer.weight_decay

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = losses.SRNLoss(opt)
    max_acc = 0

    for epoch in range(epochs):
        start_time = time.time()
        epoch_text = "[{current:>{pad}}/{end}]".format(
            current=epoch + 1,
            end=opt.num_epochs,
            pad=len(str(opt.num_epochs))
        )
        train_result = train_one_epoch(train_loader, model, device, optimizer, criterion, epoch_text)
        print(f'Epoch : {epoch + 1}/{epochs} | Train Loss : {train_result["train_loss"]}'
#               f' | Train Symbol Acc : {train_result["train_symbol_acc"]}'
              f' | Train Sentence Acc : {train_result["train_sentence_acc"]}'
              f' | Train WER : {train_result["train_wer"]}')
        valid_result = val_one_epoch(val_loader, model, device, criterion, epoch_text)
        print(f'Valid Loss : {valid_result["valid_loss"]}'
#               f' | Valid Symbol Acc : {valid_result["valid_symbol_acc"]}'
              f' | Valid Sentence Acc : {valid_result["valid_sentence_acc"]}'
              f' | Valid WER : {valid_result["valid_wer"]}')
        if max_acc < valid_result['valid_sentence_acc']:
            print(f'New Record Acc : {valid_result["valid_sentence_acc"]}! Now Saving')
            max_acc = valid_result['valid_sentence_acc']
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, file_name)
            
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f'Time Elapsed : {elapsed_time}')

def train_one_epoch(data_loader, model, device, optimizer, criterion, epoch_text):
    model.train()
    total_loss = 0
    cnt = 0
    correct_symbols = 0
    total_symbols = 0
    wer = 0
    num_wer = 0
    sent_acc = 0
    num_sent_acc = 0
    pad = data_loader.dataset.token_to_id['<PAD>']
    truth = []
    expect = []

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for batch in data_loader:
            image = batch['image'].float().to(device)
            curr_batch_size = len(image)
            others = batch['others']  # 4개짜리 리스트
            label = batch['truth']['encoded'].to(device)
            label[label == -1] = pad

            for i in range(4):
                others[i] = others[i].to(device)

            res = model(image, others)
            loss = criterion(res, label)['loss']

            total_loss += loss.item()

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            sequence = torch.squeeze(res['decoded_out'], dim = -1)
            sequence[sequence == pad] = -1
            cnt += sequence.size()[0]
            label[label == pad] = -1
            expected_str = id_to_string(label, data_loader, do_eval=1)
            sequence_str = id_to_string(sequence, data_loader, do_eval=1)
            wer += word_error_rate(sequence_str, expected_str)
            num_wer += 1
            sent_acc += sentence_acc(sequence_str, expected_str)
            num_sent_acc += 1
            sequence[sequence == -1] = pad
    #         correct_symbols += torch.sum(sequence == label[:, 1:], dim=(0, 1)).item()
    #         total_symbols += torch.sum(label[:, 1:] != -1, dim=(0, 1)).item()

            if num_wer <= 3:
                truth.append(expected_str)
                expect.append(sequence_str)
                
            pbar.update(curr_batch_size)

    for i,j in zip(truth, expect):
        print('Train GT')
        print(i)
        print('Train Expect')
        print(j)

    ans = {'train_loss' : total_loss / cnt,
#            'train_symbol_acc' : correct_symbols / total_symbols,
           'train_sentence_acc' : sent_acc / num_sent_acc,
           'train_wer' : wer / num_wer}

    return ans

def val_one_epoch(data_loader, model, device, criterion, epoch_text):
    model.eval()
    pad = data_loader.dataset.token_to_id['<PAD>']
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        correct_symbols = 0
        total_symbols = 0
        wer = 0
        num_wer = 0
        sent_acc = 0
        num_sent_acc = 0
        truth = []
        expect = []
        with tqdm(
            desc="{} ({})".format(epoch_text, "Validation"),
            total=len(data_loader.dataset),
            dynamic_ncols=True,
            leave=False,
        ) as pbar:

            for batch in data_loader:
                image = batch['image'].float().to(device)
                curr_batch_size = len(image)
                others = batch['others']  # 4개짜리 리스트
                label = batch['truth']['encoded'].to(device)
                label[label == -1] = pad

                for i in range(4):
                    others[i] = others[i].to(device)

                res = model(image, others)
                loss = criterion(res, label)['loss']
                total_loss += loss.item()

                sequence = torch.squeeze(res['decoded_out'], dim=-1)
                cnt += sequence.size()[0]
                sequence[sequence == pad] = -1
                label[label == pad] = -1
                expected_str = id_to_string(label, data_loader, do_eval=1)
                sequence_str = id_to_string(sequence, data_loader, do_eval=1)
                wer += word_error_rate(sequence_str, expected_str)
                num_wer += 1
                sent_acc += sentence_acc(sequence_str, expected_str)
                num_sent_acc += 1
                sequence[sequence == -1] = pad
    #             correct_symbols += torch.sum(sequence == label[:, 1:], dim=(0, 1)).item()
    #             total_symbols += torch.sum(label[:, 1:] != -1, dim=(0, 1)).item()

                if num_wer <= 3:
                    truth.append(expected_str)
                    expect.append(sequence_str)
                
                pbar.update(curr_batch_size)
            
    for i,j in zip(truth, expect):
        print('Valid GT')
        print(i)
        print('Valid Expect')
        print(j)

    ans = {'valid_loss': total_loss / cnt,
#            'valid_symbol_acc': correct_symbols / total_symbols,
           'valid_sentence_acc': sent_acc / num_sent_acc,
           'valid_wer': wer / num_wer}

    return ans
