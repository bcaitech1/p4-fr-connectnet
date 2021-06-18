


import torch
import os
from train import id_to_string
from metrics import word_error_rate, sentence_acc
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
import cv2

import cv2
from PIL import Image
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def main(parser):
    is_cuda = torch.cuda.is_available()
    checkpoint = load_checkpoint(parser.checkpoint, cuda=is_cuda)
    options = Flags(checkpoint["configs"]).get()
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hardware = "cuda" if is_cuda else "cpu"
    #hardware = "cuda"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    csv.field_size_limit(100000000)

    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )
    print(options.input_size.height, options.input_size.width)
    print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Symbol Accuracy : {:.5f}\n".format(checkpoint["train_symbol_accuracy"][-1]),
            "Train Sentence Accuracy : {:.5f}\n".format(checkpoint["train_sentence_accuracy"][-1]),
            "Train WER : {:.5f}\n".format(checkpoint["train_wer"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Symbol Accuracy : {:.5f}\n".format(
                checkpoint["validation_symbol_accuracy"][-1]
            ),
            "Validation Sentence Accuracy : {:.5f}\n".format(
                checkpoint["validation_sentence_accuracy"][-1]
            ),
            "Validation WER : {:.5f}\n".format(
                checkpoint["validation_wer"][-1]
            ),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

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
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
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
    results = []
    cnt=0
    with torch.no_grad():
        for d in tqdm(test_data_loader):
            input = d["image"].to(device)
            expected = d["truth"]["encoded"].to(device)

            output = model(input, expected, False, 0.0)

            decoded_values = output["dec_result"]
            decoded_values = decoded_values.transpose(1, 2)

            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)
            sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)

            for path, predicted in zip(d["file_path"], sequence_str):
                results.append((path, predicted))

            encoded_values = output["enc_result"].detach()
            encoded_values = encoded_values.cpu().detach().numpy()

            ori_im = [Image.open(p) for p in d["path"]]
            im_sizes = [(im.size[1], im.size[0]) for im in ori_im ]

            output_path = "/opt/ml/code/activation_map2/"
            for bs in range(0, parser.batch_size):
                alpha = skimage.transform.resize(encoded_values[bs], (im_sizes[bs][0], im_sizes[bs][1], 1))
                alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())

                ori = np.transpose(ori_im[bs], (0, 1, 2))
                ori = (ori - ori.min()) / (ori.max() - ori.min())

                vis = show_cam_on_image(ori, alpha)
                vis = np.uint8(255 * vis)
                vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

                plt.imsave(os.path.join(output_path + str(cnt) + "_" + str(bs) + "_atm" + d["path"][bs].split("/")[-1]), alpha.squeeze())
                plt.imsave(os.path.join(output_path + str(cnt) + "_" + str(bs) + "_ori" + d["path"][bs].split("/")[-1]), ori)
                plt.imsave(os.path.join(output_path + str(cnt) + "_" + str(bs) + "_vis" + d["path"][bs].split("/")[-1]), vis)
                cnt += 1

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default="./log/0050.pth",
        type=str,
        help="Path of checkpoint file",
    )
    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=8,
        type=int,
        help="batch size when doing inference",
    )

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/')
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )

    parser = parser.parse_args()
    main(parser)
