import argparse
import os
import torch
from pynvml import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
       
class Config():
    def build_parser(self):
        parser = argparse.ArgumentParser()
        requiredNamed = parser.add_argument_group('required named arguments')

        ### CHOOSE NAME OF RUN ###
        requiredNamed.add_argument('--name', help='name your exp anything', required=True)

        ### EXPIREMENT SETTINGS ###
        parser.add_argument('--train', type=str2bool, default=False)
        parser.add_argument('--val', type=str2bool, default=False)
        parser.add_argument('--test', type=str2bool, default=False)

        ## RUN SETTINGS ###
        parser.add_argument('--device', type=str, default="cpu")
        parser.add_argument('--pin', type=str2bool, default=False)
        parser.add_argument('--seed', type=int, default=1111)
        parser.add_argument('--benchmark', type=str2bool, default=True)
        parser.add_argument('--data_root', type=str, default=
                "/pine/scr/a/v/avmendoz/VQA-research/data")
        parser.add_argument('--output', type=str, default="logs")
        parser.add_argument('--ckp_path', type=str, default="logs/ckp")
        parser.add_argument('--tb_path', type=str, default="logs/tb")
        parser.add_argument('--log_tb', type=str2bool, default=True)

        ### MODEL SETTINGS ###
        parser.add_argument('--epoch', type=int, default=0)
        parser.add_argument('--batch_iter', type=int, default=0)
        parser.add_argument('--epochs', type=int, default=18)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--ckp', type=int, default=None)
        parser.add_argument('--e_dim', type=int, default=300)
        parser.add_argument('--q_dim', type=int, default=1024)
        parser.add_argument('--v_dim', type=int, default=2048)
        parser.add_argument('--a_dim', type=int, default=3133)
        parser.add_argument('--proj_dim', type=int, default=1024)
        parser.add_argument('--bn', type=str2bool, default=False)
        parser.add_argument('--wn', type=str2bool, default=False)
        parser.add_argument('--ln', type=str2bool, default=False)
        parser.add_argument('--SGD', type=str2bool, default=True)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--momentum', type=float, default=0.85)
        parser.add_argument('--grad_clip', type=float, default=0.25)
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        args = vars(args)
        for key in args:
            setattr(self, key, args[key])
        
if __name__ == "__main__":
    config = Config()
    for k, v in config.__dict__.items():
        print("\t", k, ":", v)   
    print()



