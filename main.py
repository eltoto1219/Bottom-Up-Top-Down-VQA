### base libraries ###
import torch
import numpy as np
from tqdm import tqdm
### from torch ###
import torch.nn as nn
import torch.backends.cudnn as cudnn
### personal file imports ###
from utils import write_events, save, compute_multi_loss, compute_multi_acc
from config import Config
### model Imports
from main_functions import tb_writer, load_data, load_model

#def eval_epoch(config, model, val_loader):
#
#    device = config.device
#    tq_loader = tqdm(iterable = val_loader, desc = "eval:{}".format(config.name), ncols = 0)
#
#    for i, (v, bb, spat, obs, a, q, q_len, item) in enumerate(tq_loader):
#        model.train()
#        begin_iter = datetime.now()
#
#        v, bb, spat, a, q, objs, q_len = v.to(device), bb.to(device),\
#            spat.to(device), a.to(device), q.to(device), obs.to(device), q_len.to(device)
#
#        with torch.no_grad():
#            loss, out, atts = model(q, q_len, v, obs, a, layout = True, 
#            loss = loss.mean().item()
#            acc = compute_multi_acc(out, a) 
#            accs.append(acc)
#            avg_acc = sum(accs)/len(accs)
#
#        tq_loader.set_postfix(acc = avg_acc, loss = loss)
#        tq_loader.update(i)
# 

def batch(config, model, optim, loader, writer, train):

    ### prelims ###
    split = "train" if train else "val"
    device = config.device
    avg_losses = []
    avg_accs = []
    val_iter = 0
    tq_loader = tqdm(iterable = loader, desc = "{}:EP{}".format(split, config.epoch), ncols = 0)
    model.train() if train else model.eval()

    for (v, bb, spat, objs, a,q, q_len, item) in tq_loader:

        ### step 0. move inputs onto device ### 
        v, bb, spat, objs, a,q, q_len, item = v.to(device), bb.to(device), spat.to(device),\
        objs.to(device), a.to(device),q.to(device), q_len.to(device), item.to(device)

        ### step 1. update ###
        if train:
            optim.zero_grad()
            pred = model(q, q_len, v, objs)
            loss = compute_multi_loss(pred, a)
            loss_item = round(loss.item(), 3)
            loss.backward()
            nn.utils.clip_grad_norm_(model.grad_params(), config.grad_clip)
            optim.step()

        else:
            with torch.no_grad():
                pred = model(q, q_len, v, objs)
                loss = compute_multi_loss(pred, a)
                loss_item = round(loss.item(), 3)
        acc = round(compute_multi_acc(pred, a), 3)
        
        ### step 2. compute stats, update batch_iter ###
        avg_losses.append(loss_item)
        avg_accs.append(acc)
        avg_acc = round(sum(avg_accs)/len(avg_accs), 3)
        avg_loss = round(sum(avg_losses)/len(avg_losses), 3) 

        if train:
            config.batch_iter += 1
            save_iter = config.batch_iter
        else:
            val_iter += 1
            save_iter = val_iter
        
        ### step 3: write events, update tqdm ### 
        if config.log_tb:
            write_events(writer, save_iter, config.epoch, acc, avg_acc, loss, avg_loss, 
                        train = train)

        ### step 4: save ###

        tq_loader.set_postfix(acc = acc, loss = loss_item, m_loss = avg_loss, m_acc = avg_acc)
        tq_loader.update(save_iter)

def epochs(config, model, optim, writer, train_loader, val_loader, test_loader, start):

    for _ in range(start, config.epochs):
        if config.train:
            batch(config, model, optim, train_loader, writer, train = True)
            save(config, model, optim, _, config.batch_iter)
        if config.train or config.val:
            batch(config, model, optim, val_loader, writer, train = False)
        if config.test:
            raise NotImplemented 
        config.epoch += 1
        
def main(config):
    torch.cuda.empty_cache()

    ### set seed ###
    if config.seed:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    if config.benchmark:
        torch.backends.cudnn.benchmark = True 

    ### load data ### 
    train_loader, val_loader, test_loader = load_data(config)

    ### make tb writer ###
    writer = tb_writer(config) if config.log_tb else None

    ### load model ### 
    print("LOADING: model ... \n")
    model, optim, start, batch_iter = load_model(config)
    config.epoch = start
    config.batch_iter = batch_iter
    
    ### run ### 
    print("Begin Run... \n")
    epochs(config, model, optim, writer, train_loader, val_loader, test_loader, start)
    ### close writer ###
    writer.close()
    print("Succesful Run")


if __name__ == '__main__':
    print("INITIALIZING config AND run...\n")
    config = Config()
    for k, v in config.__dict__.items():
        print("\t", k, ":", v)   
    print()
    main(config)
