import torch
import torch.utils.data
import torch.nn as nn
import h5py
import json
import _pickle as cPickle
import tqdm
import sys
import os
import datetime as time
import torch.utils.data
import numpy as np

#PARAMS
ROOT_DIR = "/pine/scr/a/v/avmendoz/VQA_data/"

class VQA_dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, train = False, val = False, test = False):
        super(VQA_dataset, self).__init__()
        self.test = test
        self.root_dir = root_dir

        ### here i make sure VQA_dataset is intialized with correct combination of inputs ###
        if train: assert not val and not test, "fail"
        elif val: assert not train and not test, "fail"
        elif test: assert not val and not train, "fail"
        else: raise Exception("must choose split")
 
        ###setting up neccesary vars ###
        year = "2014" if not test else "2015"
        prefix = "train" if train else "val" if val else "test"

        ### setting up paths ### 
        print("LOADING: {}".format(prefix), "\n") 
        self.ind2qid = self.op(os.path.join(self.root_dir,"dicts/ind2qid_{}.pkl".format(prefix)))
        self.word2ind = self.op(os.path.join(self.root_dir, "dicts/word2ind.pkl".format(prefix)))
        self.ind2word = self.op(os.path.join(self.root_dir, "dicts/ind2word.pkl"))
        self.qid2que = self.op(os.path.join(self.root_dir, "dicts/qid2que_{}.pkl".format(prefix)))
        self.qid2vid = self.op(os.path.join(self.root_dir, "dicts/qid2vid_{}.pkl".format(prefix)))
        self.qid2a_inds = self.op(os.path.join(self.root_dir, "dicts/qid2_a_inds.pkl"))
        self.ans2ind = self.op(os.path.join(self.root_dir, "dicts/ans2ind.pkl"))
        self.ind2ans = self.op(os.path.join(self.root_dir, "dicts/ind2ans.pkl"))
        self.num_classes = len(self.ind2ans.values())
        self.f_path = os.path.join(self.root_dir, "adaptive_features/{}.h5".format(prefix))
        self.img2f = self.op(os.path.join(self.root_dir, 
                "adaptive_features/{}_img2idx.pkl".format(prefix)), pickle = True)

        ### print stats ###
        print("number of answers to predict:", self.num_classes)
        print("number of words:", self.num_tokens)
        print("number of questions:", self.__len__())
        print("question length:", self.qmax, "\n")


    def __getitem__(self, item):
        qid = self.ind2qid[item]
        assert self.qid2que[qid] is not None, "bad qid"
        q, q_len =  self.qid2que[qid]
        q = torch.from_numpy(np.array(q)).long()
        image_id = self.qid2vid[qid]
        assert image_id is not None, "fail"
        f, bb, spat, obs = self.load_image(image_id)
        f, bb, spat, obs = self.load_image(image_id)

        if self.test:
            return f, bb, spat, obs, torch.Tensor(), q, q_len, item
        else:
            a = self.make_multi_hot(str(qid))
            return f, bb, spat, obs, a, q,  q_len, torch.tensor(int(qid))

    def __len__(self):
        return len(self.ind2qid)

    def load_image(self, image_id):
        ### load data types
        hf = h5py.File(self.f_path, 'r')
        features = hf.get('image_features')
        spatials = hf.get('spatial_features')
        bb = hf.get('image_bb')
        index = self.img2f[int(image_id)]
        pos_boxes = hf.get('pos_boxes')
        last_obj = pos_boxes[index][1]
        first_obj = pos_boxes[index][0]
        bb = bb[first_obj:last_obj,:]
        bb = torch.from_numpy(bb)
        features = features[first_obj:last_obj,:]
        features = torch.from_numpy(features)
        spatials = spatials[first_obj:last_obj,:]
        spatials = torch.from_numpy(spatials)
        max_obs = 100
        obs = bb.size(0)
        pad_amount = max_obs - obs    
        f = torch.cat((features , torch.zeros(pad_amount, features.size(1))))
        bb = torch.cat((bb, torch.zeros(pad_amount, bb.size(1))))
        obs = torch.Tensor([obs])
        spat = torch.cat((spatials , torch.zeros(pad_amount, spatials.size(1)))) 
        return f, bb, spat, obs
   
    def make_multi_hot(self, qid):
        inds_scores = self.qid2a_inds.get(str(qid))
        inds = inds_scores[0]
        scores = inds_scores[1]
        vec = torch.zeros(self.num_classes)
        for ind in inds:
            vec[ind] = scores[ind]
        return vec

    ### Helper functions ###

    def getAnswer(self, topk = 1):
        ### not implemented yet ###
        pass

    def op(self, path, pickle = True):
        if not pickle:
            with open(path, 'r') as fd:
                data = json.load(fd)
        else:
            with open(path, 'rb') as fd:
                data = cPickle.load(fd)
        return data

    ### PROPERTIES ###

    @property
    def qmax(self):
        for q, qlen in self.qid2que.values():
            q_max = len(q)
            break
        return q_max

    @property
    def num_tokens(self):
        return len(self.word2ind) + 1  # add 1 for <pad> token at index 0
