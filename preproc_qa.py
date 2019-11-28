import json
import re
import tqdm
from collections import Counter
import os 
import sys
import argparse
from copy import deepcopy
from spacy.lang.en import English
import pickle
import numpy as np

### creatining tokenizer 

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

### init the root dir ### 

ROOT = "/pine/scr/a/v/avmendoz/VQA_data/"
OUT = os.path.join(ROOT, "dicts")
PRELOAD = False

#regulaize the vocab with some regular expresssions
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_special_chars = re.compile('[^a-z0-9 ]*')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

#process punctioan

def process_punctuation(s):
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '') 
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()
 
def load_glove(dim):
    ind2emb = [[float(0)] * dim ] 
    word2ind = {}

    with open(os.path.join(ROOT,"glove.42B.{}d.txt".format(dim)), "r") as f:
        for i, line in enumerate(tqdm.tqdm(f)):
            line = line.split()
            emb = [float(x) for x in line[1:]]
            ind2emb.append(emb)
            word2ind[line[0]] = i + 1    
    return word2ind, ind2emb
   
def load_answers():
    with open(os.path.join(ROOT, "v2_mscoco_train2014_annotations.json"), "r") as t:
        vocab_train = json.load(t)
    with open(os.path.join(ROOT,"v2_mscoco_val2014_annotations.json"), "r") as v:
        vocab_val = json.load(v)
    return vocab_train["annotations"] + vocab_val["annotations"]

def load_questions(split = "Train"):
    assert split == "Train" or split == "Val" or split == "Test"
    split = split.lower()
    if split == "test":
        year = "2015"
    else: 
        year = "2014"
    with open(os.path.join(ROOT,"v2_OpenEnded_mscoco_{}{}_questions.json".format(split, year)), "r") as t:
        q = json.load(t)
        return q["questions"]

def load_all_questions():
        with open(os.path.join(ROOT,"v2_OpenEnded_mscoco_train2014_questions.json"), "r") as t:
            q_train= json.load(t)
        with open(os.path.join(ROOT,"v2_OpenEnded_mscoco_val2014_questions.json"), "r") as v:
            q_val = json.load(v)
        with open(os.path.join(ROOT,"v2_OpenEnded_mscoco_test2015_questions.json"), "r") as v:
            q_test = json.load(v)
        with open(os.path.join(ROOT,"v2_OpenEnded_mscoco_test-dev2015_questions.json"), "r") as v:
            q_dev = json.load(v)
        return q_train["questions"] + q_val["questions"] + q_test["questions"] + q_dev["questions"]

def make_vqa_glove_vocab(dim = 300):
    questions_all = load_all_questions()
    print("loading glove {}d".format(dim))
    word2ind, ind2emb = load_glove(dim = dim)
    toke_weights = [ind2emb[0]] # this is just padding index here
    temp_vocab = [0]

    ignored_tokes = 0
    print("filtering tokens not in glove")
    for q in tqdm.tqdm(questions_all):
    
        question = q["question"].lower() 
        question = _special_chars.sub('', question)
        question = tokenizer(question)

        ### tokenize the question ### 
        for token in question:
            token = str(token)
            if token not in word2ind:
                ignored_tokes += 1
            else:
                if token in temp_vocab:
                    pass
                else:
                    temp_vocab.append(token)
                    emb = ind2emb[word2ind.get(token)]
                    toke_weights.append(emb)
                    assert len(emb) == 300, "something wrong with embeddding"

    print("total ignored tokens:", ignored_tokes)
    assert len(toke_weights) == len(temp_vocab),"w or v doesnt match"                
    ### saving weights
    with open(os.path.join(ROOT, "token_weights.pkl"), "wb") as p:
        pickle.dump(toke_weights, p)

    ### saving ind to word and word 2 ind
    ind2word = {i: v for i, v in enumerate(temp_vocab[1:], start = 1 )}
    word2ind = {v: i for i, v in enumerate(temp_vocab[1:], start = 1 )}

    return ind2word, word2ind

#only pick most common answer, if no most common pick answers that occurs first
def make_answers(annotations, min_freq = 9, count_only = False):

    print("making answers")
    ind2ans = {} # all of the unique answers are stored in here
    ans2ind = {} # given answer, find its unique index
    temp = {}
    qid2type = {}

    for i, d in enumerate(tqdm.tqdm(annotations)): 
        best_answer = d["multiple_choice_answer"]
        qid = d["question_id"]
        qtype = d["question_type"]
        qid2type[qid] = qtype
        temp[i] = best_answer
    count = 0
    for ans, freq in Counter(temp.values()).most_common():
        if freq < min_freq:
            pass
        else:            
            ind2ans[count] = ans
            ans2ind[ans] = count
            count += 1
    print("num classes :", len(ans2ind))
    qid2scores = compute_softscore(ans2ind, annotations, count_only = count_only)
    print("num of unfiltered answers:", len(annotations))
    print("num of filtered answers:", len(qid2scores))
    return qid2scores, ind2ans, ans2ind, qid2type

def compute_softscore(ans2ind, annotations, count_only = False):
    print("making answers scores")
    qid2scores = {}
    for i, d in enumerate(tqdm.tqdm(annotations)): 
        ### code used to compute the aswer soft score ### 
        answers_n = []
        answer_inds = []
        answer_scores = {}
        qid = d["question_id"]

        for answer in d["answers"]:
            a = process_punctuation(answer["answer"])
            if ans2ind.get(a) is None:
                pass
            else:
                answers_n.append(a)
                a_count = answers_n.count(a)
                answer_scores[ans2ind.get(a)] = get_score(a_count)
                answer_inds.append(ans2ind.get(a))
        if len(answer_inds) > 0:
            #if count_only == True and d["answer_type"] == "number":
            answer_inds = [i for i in set(answer_inds)]
            qid2scores[str(qid)] = (answer_inds, answer_scores)
        else:
            pass
    print("# qid to score mapping:", len(qid2scores))
    return qid2scores

def get_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1

def make_questions_and_vocab(questions_json, qid2a_inds, q_max = 15, preload_vocab = None, use_glove = None, test = False):
    print("this is how many qids:", len(qid2a_inds))
    ### preload vocab would be 
    ind2qid = {}
    qid2vid = {}
    qid2que = {}

    if use_glove is not None and preload_vocab is not None:
        ind2word = use_glove
        word2ind = preload_vocab
        n_w = len(word2ind)
    else:
        print("if here you did glove wrong")
        temp_vocab = [0]
    j = 0
    ignored = 0
    for q in tqdm.tqdm(questions_json):
        ### making sure that the answer is in the dictionary ###
        q_id = q["question_id"]
        assert q_id is not None
        ind = qid2a_inds.get(str(q_id))
        ### EVERYTHING BELOW IS DEPENDENT ON WHETEHR OR NOT THE QUESTION ID MAPS TO A FILTERED ANSWER ###
        if test == True or ind is not None:
            #qid2ind[q_id] = ind
            v_id = q["image_id"] 
            ### formatting question ###        
            question = q["question"].lower() 
            question = _special_chars.sub('', question)
            question = tokenizer(question)
            ## only going up to max quesiton length
            ### tokenize the question ### 
            question_toke = []
            for i, token in enumerate(question):
                if i == q_max:
                    break
                else:
                    token = str(token)
                    if preload_vocab is None and use_glove is None: 
                        if token not in temp_vocab:
                            temp_vocab.append(token)
                        question_toke.append(temp_vocab.index(token))
                    elif preload_vocab is None and use_glove is None: 
                        toke_ind = preload_vocab.get(token)
                        assert toke_ind is not None, "not in vocab"
                        question_toke.append(toke_ind)
                    elif use_glove is not None and preload_vocab is not None:
                        toke_ind = word2ind.get(token)
                        if toke_ind == None:
                            pass
                        else:
                            question_toke.append(toke_ind)
            ### pad question ### 
            q_len = len(question_toke)          
            amount_pad = q_max - q_len
            ## calc amount pad now
            if amount_pad == 0:
                pass
            else: 
                question_toke = question_toke + [0] * amount_pad
            ind2qid[j] =  q_id
            qid2vid[q_id] = v_id
            qid2que[q_id] = (question_toke, q_len)
            j += 1
        else:
            ignored +=1 
            pass

    print("num questions:", j)
    print("num ignored:", ignored)
    if test == False:
        assert len(ind2qid) > 10000, "done messed up"
    ### making vocab ### 
    if preload_vocab is None and use_glove is None:
        ind2word = {i: v for i, v in enumerate(temp_vocab[1:], start = 1)}
        word2ind = {v: i for i, v in enumerate(temp_vocab[1:], start = 1)}
    elif preload_vocab is not None and use_glove is None: 
        word2ind = preload_vocab
        ind2word = {i: w for w, i in preload_vocab.items()}
    elif use_glove is not None:
        pass
    ### returning results

    qid2inds = (qid2vid, qid2que, qid2a_inds)
    vocab = (word2ind, ind2word)

    return ind2qid, qid2inds, vocab

### save + load funcs ### 

def save(p, d):
    with open(p, "wb") as f:
        pickle.dump(d, f)
def op(p):
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":

    print("data loading")
    print("0%")
    anss = load_answers()
    print("33%")
    q_t = load_questions(split = "Train")
    q_test = load_questions(split = "Test")
    q_v = load_questions(split = "Val")
    print("67%")

    ind2ans  = os.path.join(OUT, "ind2ans.pkl")
    ans2ind = os.path.join(OUT, "ans2ind.pkl")
    q2a = os.path.join(OUT, "qid2_a_inds.pkl")
    
    
    if PRELOAD:
        i2w = op(os.path.join(OUT, "ind2word.pkl"))
        w2i = op(os.path.join(OUT, "word2ind.pkl"))
        ind2ans = op(ind2ans)   
        ans2ind = op(ans2ind)   
        q2a_inds = op(q2a)
        print("100%")
    else:
        i2w, w2i = make_vqa_glove_vocab()
        q2a_inds, i2a, a2i, q2type = make_answers(anss, min_freq = 9, count_only = False)
        save(os.path.join(OUT, "ind2word.pkl"), i2w)
        save(os.path.join(OUT, "word2ind.pkl"), w2i)
        save(os.path.join(OUT, "qid2type.pkl"), q2type)
        save(q2a, q2a_inds)
        save(ind2ans, i2a) 
        save(ans2ind, a2i)
    print("100%")

    ### now i make the questions vocab and everything else

    ### for test ### 
    ind2qid_test,\
    (qid2vid_test, qid2que_test, qid2a_inds_test),\
    (word2ind_test, ind2word_test) =\
    make_questions_and_vocab(q_test, q2a_inds, q_max =  14, use_glove = i2w, preload_vocab = w2i, test = True)

    ### for train ### 
    ind2qid_train,\
    (qid2vid_train, qid2que_train, qid2a_inds_train),\
    (word2ind_train, ind2word_train) =\
    make_questions_and_vocab(q_t, q2a_inds, q_max =  14, use_glove = i2w, preload_vocab = w2i)
    ### for val ### 
    ind2qid_val,\
    (qid2vid_val, qid2que_val, qid2a_inds_val),\
    (word2ind_val, ind2word_val) =\
    make_questions_and_vocab(q_v, q2a_inds, q_max =  14, use_glove = i2w, preload_vocab = w2i) 
    
    trains = [ind2qid_train, qid2vid_train, qid2que_train, qid2a_inds_train, word2ind_train, ind2word_train]
    vals = [ind2qid_val, qid2vid_val, qid2que_val, qid2a_inds_val, word2ind_val, ind2word_val]
    tests = [ind2qid_test, qid2vid_test, qid2que_test, qid2a_inds_test, word2ind_test, ind2word_test]

    print("saving data")
 
    for i in ["train", "val", "test"]:
        ind2qid = os.path.join(OUT, "ind2qid_{}.pkl".format(i))
        qid2vid = os.path.join(OUT, "qid2vid_{}.pkl".format(i))
        qid2que = os.path.join(OUT, "qid2que_{}.pkl".format(i))
        qid2ind = os.path.join(OUT, "qid2a_inds_{}.pkl".format(i))
        word2ind = os.path.join(OUT, "word2ind_{}.pkl".format(i))
        ind2word = os.path.join(OUT, "ind2word_{}.pkl".format(i))

        dicts = [ind2qid, qid2vid, qid2que, qid2ind, word2ind, ind2word]

        if i == "train":
            for d, p in zip(trains, dicts):
                print("saving:", p)
                save(p, d)
        elif i == "val":
             for d, p in zip(vals, dicts):
                print("saving:", p)
                save(p, d)
        elif i == "test":
             for d, p in zip(tests, dicts):
                print("saving:", p)
                save(p, d)

    print("DONE!")
