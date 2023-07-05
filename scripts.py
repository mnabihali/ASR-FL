import torch
import flwr as fl
from my_conf import *
from collections import OrderedDict
import math
import time
from torch import nn, optim
import os
import torchaudio
from torch.optim import Adam, AdamW
import sys
from models.model.early_exit import Early_encoder, Early_transformer, Early_conformer, my_conformer
from util.epoch_timer import epoch_time
from util.beam_infer  import ctc_predict, greedy_decoder
from dataset import *
from torchmetrics.text import WordErrorRate
from tqdm import tqdm
import numpy as np
from util.tokenizer import *
from util.beam_infer import *

import random


wer = WordErrorRate

def WER(metric, y_expected, dec_out):
    my_wer = metric(y_expected, dec_out)
    return my_wer

torch.set_num_threads(10)
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
wer = WordErrorRate()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

Loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

def train_asr(net, epochs:int, trainloader): #add_train=True
    net.train()
    epoch_loss = 0
    acc = []

    #if add_train:
    #    iterator = trainloader_gl
    #else:

    iterator = trainloader

    learning_rate = 0.01 #SGD: (google 0.008) 0.001
    optimizer=torch.optim.SGD(params=net.parameters(), lr=learning_rate)

    #optimizer = torch.optim.Adam(params=net.parameters(), lr=0, betas=(0.9, 0.98), eps=adam_eps, weight_decay=weight_decay)
    #optimizer = NoamOpt(d_model, warmup, AdamW(params=net.parameters(), lr=0, betas=(0.9, 0.98), eps=adam_eps, weight_decay=weight_decay))

    for i in range (epochs):
        for i, batch in enumerate(tqdm(iterator)):
        
            print('Iterator len', len(iterator))
            
            num_examples = len(iterator.dataset)
            if not batch:
                continue
    
            src = batch[0].to(device)
            trg = batch[1][:, :-1].to(device)  # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
            trg_expect = batch[1][:, 1:].to(device)  # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
            valid_lengths = batch[3]
            encoder = net(src, valid_lengths)
            ctc_target_len = batch[2]
            loss_layer = 0
            if i % 300 == 0:
                if bpe_flag:
                    #print("EXPECTED:", sp.decode(trg_expect[0].tolist()).lower())
                    expected = sp.decode(trg_expect[0].tolist()).lower()
                else:
                    #print("EXPECTED:", text_transform.int_to_text(trg_expect[0]))
                    expected = text_transform.int_to_text(trg_expect[0])
                        
            last_probs = encoder[encoder.size(0) - 1].to(device)
            ctc_input_len = torch.full(size=(encoder.size(1),), fill_value=encoder.size(2), dtype=torch.long)
            for enc in encoder[0:encoder.size(0) - 1]:
                loss_layer += ctc_loss(enc.permute(1, 0, 2), batch[1], ctc_input_len, ctc_target_len).to(device)
                if i % 300 == 0:
                    if bpe_flag:
                        #print("CTC_OUT at [", i, "]:", sp.decode(ctc_predict(enc[0].unsqueeze(0))).lower())
                        dec = sp.decode(ctc_predict(enc[0].unsqueeze(0))).lower()
                    else:
                        #print("CTC_OUT at [", i, "]:", ctc_predict(enc[0].unsqueeze(0)))
                        dec = ctc_predict(enc[0].unsqueeze(0))
            del encoder
            loss_layer += ctc_loss(last_probs.permute(1, 0, 2), batch[1], ctc_input_len, ctc_target_len).to(device)
            if i % 300 == 0:
                if bpe_flag:
                    #print("CTC_OUT at [", i, "]:", sp.decode(ctc_predict(last_probs[0].unsqueeze(0))).lower())
                    predicted = sp.decode(ctc_predict(last_probs[0].unsqueeze(0))).lower()
                else:
                    #print("CTC_OUT at [", i, "]:", ctc_predict(last_probs[0].unsqueeze(0)))
                    predicted = ctc_predict(last_probs[0].unsqueeze(0))
            metric = wer(predicted , expected)
            acc.append(metric)
            loss = loss_layer
            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.detach().item()
            #print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
            norm_loss = epoch_loss / len(iterator)
        avg_wer = sum (acc) / len(acc)
        print("LOSS:",norm_loss,"N_SAMPLES:",num_examples,"WER:",avg_wer)
        return norm_loss, num_examples, avg_wer
        
        


def eval_asr(net, devloader):  # I need to return the loss also and WER
    file_dict = 'librispeech.lex'
    words = load_dict(file_dict)
    net.eval()
    # w_ctc = float(sys.argv[1])
    epoch_loss = 0
    beam_size = 10
    #acc = []
    error_values = []
    random_index = random.randrange(len(valloaders))
    data_loader = valloaders[int(random_index)]
    
    
    set_ = "test_clean"
    for batch in tqdm(data_loader):
    
        print('len data_loader', len(data_loader))    
        num_examples = len(data_loader.dataset)
        trg_expect = batch[1][:, 1:].to(device)  # shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]
        trg = batch[1][:, :-1].to(device)  # cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28]
        
        loss_layer = 0
        expected_output = None
        dec_output = None
        
        for trg_expect_ in trg_expect:
        
            if bpe_flag:
                expected_output = sp.decode(trg_expect_.squeeze(0).tolist()).lower()
                #print(set_, "EXPECTED:", expected_output)
                
            else:
                expected_output = (re.sub(r"[#^$]+", "", text_transform.int_to_text(trg_expect_.squeeze(0))))
                #print(set_, "EXPECTED:", expected_output)
                
        valid_lengths = batch[2]

        encoder = net(batch[0].to(device), valid_lengths)
        #last_probs = encoder[encoder.size(0) -1].to(device)
        #ctc_input_len = torch.full(size=(encoder.size(1),), fill_value = encoder.size(2), dtype=torch.long)
        i = 0

        for enc in encoder:
            i = i + 1
            best_combined = ctc_predict(enc, i - 1)
            #loss_layer += ctc_loss(enc.permute(1,0,2), batch[1], ctc_input_len, valid_lengths).to(device)
            for best_ in best_combined:
            
                if bpe_flag:
                    dec_output = apply_lex(sp.decode(best_).lower(), words)
                    #print(set_, " BEAM_OUT_", i, ":", dec_output)
                    
                else:
                    dec_output = apply_lex(re.sub(r"[#^$]+", "", best_.lower()), words)
                    #print(set_, " BEAM_OUT_", i, ":", dec_output)
                    
    error_value = WER(wer, expected_output, dec_output)
    error_values.append(error_value)
    print("Word Error Rate for Current Sentence is: ", error_value.item())
    
    avg_wer = (sum(error_values)/len(error_values)).item()
    print("Cumulative Word Error Rate is: ", avg_wer)           
        
    norm_loss = 10 #epoch_loss / len(data_loader)
     #sum(acc) / len(acc)
    return norm_loss, num_examples, avg_wer



#metric = wer(predicted, expected)
        #acc.append(metric)
        #loss_layer += ctc_loss(last_probs.permute(1,0,2), batch[1], ctc_input_len, valid_lengths).to(device)
        #loss = loss_layer
        #epoch_loss += loss.item()

