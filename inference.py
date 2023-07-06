"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
import os
import torchaudio

from torchaudio.models.decoder import ctc_decoder
import sys
import re

from models.model.early_exit import Early_encoder, Early_transformer, Early_conformer, my_conformer

from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from util.data_loader import text_transform, spec_transform, melspec_transform, pad_sequence
from util.tokenizer import *
from util.beam_infer import *

from my_conf import *

from util.data_loader import collate_infer_fn
#from voxpopuli import dataloader_voxpopuli  
import argparse

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

'''
model = Early_encoder(src_pad_idx=src_pad_idx,
            n_enc_replay=n_enc_replay,
            d_model=d_model,
            enc_voc_size=enc_voc_size,
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            dim_feed_forward=dim_feed_forward,
            n_head=n_heads,
            n_encoder_layers=n_encoder_layers,
            features_length=n_mels,
            drop_prob=drop_prob,
            device=device).to(device)
'''

device="cuda:0"
#n_enc_replay=1
#n_encoder_layers=4
model = Early_conformer(src_pad_idx=src_pad_idx,
                        n_enc_replay=n_enc_replay,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        dim_feed_forward=dim_feed_forward,
                        n_head=n_heads,
                        n_encoder_layers=n_encoder_layers,
                        features_length=n_mels,
                        drop_prob=drop_prob,
                        depthwise_kernel_size=depthwise_kernel_size,
                        device=device).to(device)

'''
n_encoder_layers=12

model = my_conformer(src_pad_idx=src_pad_idx,
                     d_model=d_model,
                     enc_voc_size=enc_voc_size,
                     dec_voc_size=dec_voc_size,
                     max_len=max_len,
                     dim_feed_forward=dim_feed_forward,
                     n_head=n_heads,
                     n_encoder_layers=n_encoder_layers,
                     features_length=n_mels,
                     drop_prob=drop_prob,
                     depthwise_kernel_size=depthwise_kernel_size,
                     device=device).to(device)
'''
print(f'The model has {count_parameters(model):,} trainable parameters')
print("batch_size:",batch_size," num_heads:",n_heads," num_encoder_layers:", n_encoder_layers,"vocab_size:",dec_voc_size,"DEVICE:",device) 

loss_fn = nn.CrossEntropyLoss()
ctc_loss = nn.CTCLoss(blank=0,zero_infinity=True)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def collate_infer_fn(batch, SOS_token=trg_sos_idx, EOS_token=trg_eos_idx, PAD_token=trg_pad_idx):   

    tensors, targets, t_source = [], [], []

    # Gather in lists, and encode labels as indices
    #for waveform, _, label, *_, ut_id in batch:
    for waveform, smp_freq, label, spk_id, ut_id, *_ in batch:
        #label=re.sub(r"[#^$?:;'.!]+|<unk>","",label)
        label=re.sub(r"[#^$?:;.!]+|<unk>","",label)        

        if "ignore_time_segment_in_scoring" in label:
            continue
        spec=spec_transform(waveform)#.to(device)
        spec = melspec_transform(spec).to(device)
        t_source += [spec.size(2)]
        npads = 1000
        if spec.size(2)>1000:
            npads = 500
        #spec = F.pad(spec, (0,npads), mode='constant',value=0)
        
        tensors += spec
        del spec
        if bpe_flag == True:
            tg=torch.LongTensor([sp.bos_id()] + sp.encode_as_ids(label.upper()) + [sp.eos_id()])
        else:
            tg=torch.LongTensor(text_transform.text_to_int("^"+label.lower()+"$")) 
        
        targets += [tg.unsqueeze(0)]
        del waveform
        del label
    if tensors:
        tensors = pad_sequence(tensors,0)
        targets = pad_sequence(targets,PAD_token)
        return tensors.squeeze(1), targets.squeeze(1), torch.tensor(t_source)
    else:
        return None
    

        
def evaluate(model,model_name,round):

    file_dict='librispeech.lex'
    words=load_dict(file_dict)

    #path = os.getcwd()+'/trained_model/bpe_tedlium/'
    path = os.getcwd()+'/trained_models'

    model = avg_models(model, path, model_name, 1,round)
    model.eval()
    #w_ctc = float(sys.argv[1])


    beam_size=10
    batch_size=1
    #for set_ in "test-clean","test-other","dev-clean", "dev-other":
    for set_ in "dev", "test":
        print(set_)
        
        #test_dataset = torchaudio.datasets.LIBRISPEECH("/falavi/corpora", url=set_, download=False)
        #test_dataset=dataloader_voxpopuli.VOXPOPULI('/falavi/corpora/voxpopuli/',url=set_,lang='en')
        test_dataset = torchaudio.datasets.TEDLIUM("/falavi/corpora/", release="release3", subset=set_, download=False)
        
        data_loader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=batch_size, num_workers=0, shuffle=False, collate_fn=collate_infer_fn)
    
        for batch in data_loader:
            if not batch:
                continue
            trg_expect =batch[1][:,1:].to(device) #shift [0, 28, ..., 28, 29] -> [28, ..., 28, 29]   
            trg = batch[1][:,:-1].to(device) #cut [0, 28, ..., 28, 29] -> [0, 28, ..., 28] 
            for trg_expect_ in trg_expect:
                if bpe_flag == True:
                    print(set_,"EXPECTED:",sp.decode(trg_expect_.squeeze(0).tolist()).lower())
                else:                    
                    print(set_,"EXPECTED:",re.sub(r"[#^$]+","",text_transform.int_to_text(trg_expect_.squeeze(0))))
                   
            valid_lengths=batch[2]                    
            encoder=model(batch[0].to(device), valid_lengths)
            ###encoder=model(batch[0].to(device))
            i=0
            for enc in encoder:
                i=i+1
                best_combined = ctc_predict(enc, i-1)

                for best_ in best_combined:
                    if bpe_flag==True:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(sp.decode(best_).lower(),words))
                    else:
                        print(set_," BEAM_OUT_",i,":",  apply_lex(re.sub(r"[#^$]+","",best_.lower()),words))
                
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(prog='Inference')
    parser.add_argument('--model', type=str,help='model to be validate')
    parser.add_argument('--round', type=int, help='model round')
    args=parser.parse_args()
    evaluate(model,args.model,args.round)
