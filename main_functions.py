### base libraries ###
import os
import torch
import json
### from torch ###
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
### personal file imports ###
from dataloader import VQA_dataset
from utils import compute_multi_loss, compute_multi_acc
from config import Config
### model Imports
from model import Model
from datetime import datetime

def load_data(config):
    ### train loaders that can be returned ###
    train_loader = None
    val_loader = None
    test_loader = None

    ### run options ### 
    test = config.test
    train = config.train
    val = config.val
    root = config.data_root
    batch_size = config.batch_size

    if test:
        data_test = VQA_dataset(
            root_dir = root , 
            test = True) 
        test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, 
            shuffle=True, pin_memory = config.pin)
    if val or train:
        data_val = VQA_dataset(
            root_dir = root , 
            val = True)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, 
            shuffle=True, pin_memory = config.pin)
    if train:
       data_train = VQA_dataset(
            root_dir = root , 
            train = True)
       train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, 
            shuffle=True, pin_memory = config.pin)

    return train_loader, val_loader, test_loader

def tb_writer(config):
    ### settings ###
    date = str(datetime.now().replace(microsecond=0))[:-3].replace(" ", "-")
    out_path = config.output
    run_name = config.name
    ### define default paths ###
    ckp = "ckps_{}_{}".format(date, run_name)
    tb =  "tb_events_{}_{}".format(date, run_name)
    ckp_path = os.path.join(os.getcwd(), out_path, ckp)
    tb_path = os.path.join(os.getcwd(), out_path, tb)

    try:
        os.mkdir(ckp_path)
    except:
        pass
    try:
        os.mkdir(tb_path)
    except:
        pass
    writer = SummaryWriter(log_dir = tb_path)
    config.ckp_path = ckp_path
    config.tb_path = tb_path

    return writer

def load_model(config):

    emb_dim = config.e_dim
    q_dim = config.q_dim
    v_dim = config.v_dim
    a_dim = config.a_dim
    bn = config.bn
    wn = config.wn
    ln = config.ln
    proj_dim = config.proj_dim
    device = config.device
    ckp_path = config.ckp_path
    data_path = config.data_root
    name = config.name
    
    model = Model(
        weight_dir = data_path,
        v_dim = v_dim, 
        q_dim = q_dim, 
        a_dim = a_dim, 
        common_dim = proj_dim, 
        glove_dim = emb_dim, 
        bn = bn, 
        wn = wn, 
        ln = ln).to(device)
    
    if config.SGD:
        optim = torch.optim.SGD(model.grad_params(), lr = config.lr, momentum = config.momentum) 
    else:
        optim = torch.optim.Adam(model.grad_params()) 

    ### loading from checkpoint if possible ###
    if config.ckp is None:
        batch_iter = 0
        epoch = 0
    else:
        checkpoint = torch.load(config.ckp, map_location=torch.device(device))
        epoch = checkpoint["epoch"]
        pretrained_dict = checkpoint["weights"]
        batch_iter = checkpoint["batch_iter"]
        w_optim_dict = checkpoint["w_optim_dict"]
        ### add batch_number here later ###
        try:
            model.load_state_dict(pretrained_dict)
            w_optim.load_state_dict(w_optim_dict)
            a_optim.load_state_dict(a_optim_dict)
            lr_scheduler.load_state_dict(lr_dict)         
        except:
            raise Exception("checkpoint is not compatible")
        print("\nLOADED: checkpoint")

    return model, optim, epoch, batch_iter
