import torch
import os
import torch.nn as nn 
import pickle
from tensorboardX import SummaryWriter

def mask_tensor(src, lens, mask_val = float("-inf"), reverse = False, dtype = torch.FloatTensor):

    packed = torch.nn.utils.rnn.pack_sequence(
        [x[int(l)] if reverse else x[:int(l)] for x,l in zip(src,lens)],
        enforce_sorted = False)
    padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
        packed, batch_first = True, total_length = 100, 
        padding_value = mask_val)

    return padded.type(dtype).to(src.device)
    
def load_weights(home_dir):
    PATH = os.path.join(home_dir, "token_weights.pkl")
    with open(PATH, "rb") as f:
        data = pickle.load(f)
    return data

def compute_multi_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).to(labels.device)
    loss *= labels.size(1)
    return loss

def compute_multi_acc(logits, labels):
    logit_max = logits.max(dim = 1)[1] # argmax
    logit_max.unsqueeze_(1).to(labels.device)
    one_hots = torch.zeros(*labels.size()).to(labels.device)
    one_hots.scatter_(1, logit_max, 1)
    acc = (one_hots * labels).sum() / labels.size(0) * 100
    return acc.item()

def write_events(writer, batch_iter, epoch, acc, avg_acc, loss, avg_loss, train = True):
    split = "train" if train else "val"
    writer.add_scalar('{}_epoch/batch_iter'.format(split), epoch, batch_iter)
    writer.add_scalar('{}/acc'.format(split), acc, batch_iter)
    writer.add_scalar('{}/avg_acc'.format(split), avg_acc, batch_iter)
    writer.add_scalar('{}/loss'.format(split), loss, batch_iter)
    writer.add_scalar('{}/avg_loss'.format(split), avg_loss, batch_iter)

def save(config, model, optim, epoch, batch_iter):

            ckp_path = config.ckp_path
            name = config.name
            target = name + "_epoch_{}.pt".format(epoch) 
            target = os.path.join(ckp_path, target)

            try:
                os.mkdir(ckp_path)
            except:
                pass
            
            results = {
                'name': config.name,
                'epoch_num': epoch,
                'weights': model.state_dict(),
                'batch_iter': batch_iter,
                'w_optim_dict': optim.state_dict(),
            }

            try:
                os.mknod(target)
            except:
                pass

            torch.save(results, target)  
