import my_train
import argparse
from flags import *
import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config_file",
    dest="config_file",
    default="cstr.yaml",
    type=str,
    help="Path of configuration file",
)
parser = parser.parse_args()
option = Flags(parser.config_file).get()
my_train.go(option, 'test_file.pth')
