import warnings
warnings.filterwarnings('ignore')
import argparse
import dataset
from flags import Flags
from torchvision import transforms
import torch
from my_utils import seed_fix
from my_train import go
from net import SRN

print(torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_file",
    dest="config_file",
    default="SRN.yaml",
    type=str,
    help="Path of configuration file",
)
parser = parser.parse_args()
options = Flags(parser.config_file).get()
trans = transforms.Compose(
    [
        # Resize so all images have the same size
        transforms.Resize((options.input_size.height, options.input_size.width)),
        transforms.ToTensor(),
    ]
)
train_loader, val_loader, train_dataset, val_dataset = dataset.dataset_loader(options, trans)
SEED = options.seed
seed_fix(SEED)
model = SRN(options).to(device)
go(options, train_loader, val_loader, model,'test_file.pt', device)