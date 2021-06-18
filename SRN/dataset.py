import csv
import os
import random
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):
    truth_tokens = truth.split()
    for token in truth_tokens:
        if token not in token_to_id:
            raise Exception("Truth contains unknown token")
    truth_tokens = [token_to_id[x] for x in truth_tokens]
    if '' in truth_tokens: truth_tokens.remove('')
    return truth_tokens


def load_vocab(tokens_paths):
    tokens = []
    tokens.extend([START, END])
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    tokens.append(PAD)
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i : tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def split_gt(groundtruth, proportion=1.0, test_percent=None):
    root = os.path.join(os.path.dirname(groundtruth), "images")
    with open(groundtruth, "r") as fd:
        data = []
        for line in fd:
            data.append(line.strip().split("\t"))
        random.shuffle(data)
        dataset_len = round(len(data) * proportion)
        data = data[:dataset_len]
        data = [[os.path.join(root, x[0]), x[1]] for x in data]

    if test_percent:
        test_len = round(len(data) * test_percent)
        return data[test_len:], data[:test_len]
    else:
        return data


def collate_batch(data):
    max_len = 254
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    encoder_word_pos = [d['others'][0] for d in data]
    encoder_word_pos = torch.stack(encoder_word_pos, dim=0)

    gsrm_word_pos = [d['others'][1] for d in data]
    gsrm_word_pos = torch.stack(gsrm_word_pos, dim=0)

    gsrm_slf_attn_bias1 = [d['others'][2] for d in data]
    gsrm_slf_attn_bias1 = torch.stack(gsrm_slf_attn_bias1, dim=0)

    gsrm_slf_attn_bias2 = [d['others'][3] for d in data]
    gsrm_slf_attn_bias2 = torch.stack(gsrm_slf_attn_bias2, dim=0)

    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
        'others' : [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2],
    }


def collate_eval_batch(data):
    max_len = 254
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]

    encoder_word_pos = [d['others'][0] for d in data]
    encoder_word_pos = torch.stack(encoder_word_pos, dim=0)

    gsrm_word_pos = [d['others'][1] for d in data]
    gsrm_word_pos = torch.stack(gsrm_word_pos, dim=0)

    gsrm_slf_attn_bias1 = [d['others'][2] for d in data]
    gsrm_slf_attn_bias1 = torch.stack(gsrm_slf_attn_bias1, dim=0)

    gsrm_slf_attn_bias2 = [d['others'][3] for d in data]
    gsrm_slf_attn_bias2 = torch.stack(gsrm_slf_attn_bias2, dim=0)

    return {
        "path": [d["path"] for d in data],
        "file_path": [d["file_path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded)
        },
        'others': [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2],
    }

def srn_other_inputs(image_shape, num_heads, max_text_length):
    imgC, imgH, imgW = image_shape
    feature_dim = int((imgH / 8) * (imgW / 8))

    encoder_word_pos = np.array(range(0, feature_dim)).reshape(
        (feature_dim, 1)).astype('int64')
    gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
        (max_text_length, 1)).astype('int64')

    gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
    gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1,
                                  [num_heads, 1, 1]) * [-1e9]

    gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
        [1, max_text_length, max_text_length])
    gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2,
                                  [num_heads, 1, 1]) * [-1e9]

    return [
        torch.tensor(encoder_word_pos), torch.tensor(gsrm_word_pos), torch.tensor(gsrm_slf_attn_bias1),
        torch.tensor(gsrm_slf_attn_bias2)
    ]

class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(self, opt, groundtruth, tokens_file, transform=None):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadDataset, self).__init__()
        self.transform = transform
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.width = opt.input_size.width
        self.height = opt.input_size.height
        self.max_length = opt.model.max_length
        self.num_heads = opt.model.num_heads

        self.data = [
            {
                "path": p,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        
        width, height = image.size
        
        if (width/ height) < 0.75:
            angle = 90
            image = image.rotate(angle, expand=True)

        image = image.convert("L")

        others = srn_other_inputs((1, self.height, self.width), self.num_heads, self.max_length)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "truth": item["truth"], "image": image, 'others' : others}


class LoadEvalDataset(Dataset):
    """Load Dataset"""

    def __init__(
            self,
            groundtruth,
            token_to_id,
            id_to_token,
            crop=False,
            transform=None,
            rgb=3,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadEvalDataset, self).__init__()
        self.crop = crop
        self.rgb = rgb
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.transform = transform
        self.data = [
            {
                "path": p,
                "file_path": p1,
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, p1, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)



    def __getitem__(self, i):
        item = self.data[i]
        image = cv2.imread(item["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        others = srn_other_inputs((1, 64, 64), 8, 254)

        if self.transform:
            image = self.transform(image=image)['image']

        return {"path": item["path"], "file_path": item["file_path"], "truth": item["truth"], "image": image, 'others' : others}


def dataset_loader(options, transformed):
    # Read data
    train_data, valid_data = [], []
    if options.data.random_split:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train, valid = split_gt(path, prop, options.data.test_proportions)
            train_data += train
            valid_data += valid
    else:
        for i, path in enumerate(options.data.train):
            prop = 1.0
            if len(options.data.dataset_proportions) > i:
                prop = options.data.dataset_proportions[i]
            train_data += split_gt(path, prop)
        for i, path in enumerate(options.data.test):
            valid = split_gt(path)
            valid_data += valid

    # Load data
    train_dataset = LoadDataset(
        options, train_data, options.data.token_paths, transform=transformed
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    valid_dataset = LoadDataset(
        options, valid_data, options.data.token_paths, transform=transformed
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )

    return train_data_loader, valid_data_loader, train_dataset, valid_dataset
