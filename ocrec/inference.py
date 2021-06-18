import torch
import os
#from train import id_to_string
#from metrics import word_error_rate, sentence_acc
from checkpoint import load_checkpoint
from torchvision import transforms
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from flags import Flags
from utils import get_network, get_optimizer
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm
from PIL import Image, ImageOps

from networks.Attention import Attention
# import cv2

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# def inference(data):
#     print("Before:",data)
#     data = gen_data(data)
#     print("After:",data)
    
    
#     #TODO
#     #이곳에서 위에서 생성한 데이터를 기반으로 inference한 값들을 평균을 내서 입력해주시면 되겠습니다.
#     #probability의 평균
#     result = 100

#     return result 


def id_to_string(tokens, data_loader,do_eval=0):
    result = []
    if do_eval:
        special_ids = [data_loader.dataset.token_to_id["<PAD>"], data_loader.dataset.token_to_id["<SOS>"],
                       data_loader.dataset.token_to_id["<EOS>"]]

    for example in tokens:
        string = ""
        if do_eval:
            for token in example:
                token = token.item()
                if token not in special_ids:
                    if token != -1:
                        string += data_loader.dataset.id_to_token[token] + " "
        else:
            for token in example:
                token = token.item()
                if token != -1:
                    string += data_loader.dataset.id_to_token[token] + " "

        result.append(string)
    return result

def get_target(model, test_data_loader, device):
    target = ""
    for d in test_data_loader:
        input = d["image"].to(device)
        expected = d["truth"]["encoded"].to(device)
        output = model(input, expected, False, 0.0)   #bath , seq_len, class
        
        decoded_values = output.transpose(1, 2) 
        _, sequence = torch.topk(decoded_values, 1, dim=1)

        sequence = sequence.squeeze(1)
        sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
        target = sequence_str[0]
        break
    return target

def main(parser):
    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )
    #print(options.input_size.height)

    transformed = transforms.Compose(
        [
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )


    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
        rgb=options.data.rgb
    )


    
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_eval_batch,
    )

  

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )

    model.eval()

    target = get_target(model, test_data_loader, device)

    print(target)
    
    target = get_target(model, test_data_loader, device)

    print(target)

class LInference():
    def __init__(self):
        checkpoint ="./att.pth"
        file_path = os.path.join("./static/", 'input.txt')

        is_cuda = torch.cuda.is_available()
        checkpoint = load_checkpoint(checkpoint, cuda=is_cuda)
        options = Flags(checkpoint["configs"]).get()
        torch.manual_seed(options.seed)
        random.seed(options.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        hardware = "cuda" if is_cuda else "cpu"
        self.device = torch.device(hardware)
        print("--------------------------------")
        print("Running {} on device {}\n".format(options.network, self.device))

        model_checkpoint = checkpoint["model"]

        print("load model")

        self.transformed = transforms.Compose(
            [
                transforms.Resize((options.input_size.height, options.input_size.width)),
                transforms.ToTensor(),
            ]
        )


        dummy_gt = "\sin " * 230  # set maximum inference sequence

        root = os.path.join(os.path.dirname(file_path), "images")
        with open(file_path, "r") as fd:
            reader = csv.reader(fd, delimiter="\t")
            data = list(reader)
        test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]

        #print(test_data)

        test_dataset = LoadEvalDataset(
            test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=self.transformed,
            rgb=options.data.rgb
        )

        
      
        self.test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_eval_batch,
        )

        #print(test_dataset[0]["truth"]["encoded"])
        self.test_dataset = test_dataset

        self.model = get_network(
            options.network,
            options,
            model_checkpoint,
            self.device,
            test_dataset,
        )

        self.model.eval()
        print("load finish")

    def getLatext(self, image):
        target = ""
        image = image.convert("L")
        image = self.transformed(image)

        input = image.unsqueeze(0).to(self.device)

        val = torch.full((1, 232), 158)
        val[0][0]=0
        val[0][-1]=1
        
        
        with torch.no_grad():
            expected = val.to(self.device)
            output = self.model(input, expected, False, 0.0)   #bath , seq_len, class
            
            decoded_values = output.transpose(1, 2) 
            _, sequence = torch.topk(decoded_values, 1, dim=1)

            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence, self.test_data_loader, do_eval=1)
            target = sequence_str
            
        return target

    def get_target(self):
        target = ""
        for d in self.test_data_loader:
            input = d["image"].to(self.device)
            expected = d["truth"]["encoded"].to(self.device)

            
            #print(expected)
            output = self.model(input, expected, False, 0.0)   #bath , seq_len, class
            
            decoded_values = output.transpose(1, 2) 
            _, sequence = torch.topk(decoded_values, 1, dim=1)

            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence, self.test_data_loader, do_eval=1)
            target = sequence_str
            
        return target


def load_model():

    checkpoint ="./aster.pth"
    file_path = os.path.join("./static/", 'input.txt')

    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    model_checkpoint = checkpoint["model"]

    print("load model")

    transformed = transforms.Compose(
        [
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )


    dummy_gt = "\sin " * 230  # set maximum inference sequence

    root = os.path.join(os.path.dirname(file_path), "images")
    with open(file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    
    print(test_data)

    test_dataset = LoadEvalDataset(
        test_data, checkpoint["token_to_id"], checkpoint["id_to_token"], crop=False, transform=transformed,
        rgb=options.data.rgb
    )


    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )

    model.eval()
    print("load finish")




if __name__ == "__main__":

    #load_model()
    ll = LInference()
    #for test
    print(ll.get_target())


