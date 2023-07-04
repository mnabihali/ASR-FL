from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_scripts import train_asr, eval_asr
from dataset import load_datasets
import gc
from models.model.early_exit import Early_encoder, Early_transformer, Early_conformer
from my_conf import *
import flwr as fl
from flwr.server.strategy.aggregate import aggregate

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Scalar,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

RAY_DEDUP_LOGS=0



n_encoder_layers = 2
n_enc_replay = 6
    
net = Early_conformer(src_pad_idx=src_pad_idx, n_enc_replay=n_enc_replay, d_model=d_model, enc_voc_size=enc_voc_size, dec_voc_size=dec_voc_size, max_len=max_len,
                      dim_feed_forward=dim_feed_forward, n_head=n_heads, n_encoder_layers=n_encoder_layers, features_length=n_mels, drop_prob=drop_prob,
                      depthwise_kernel_size=depthwise_kernel_size, device=device).to(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, _, trainloader_gl, test_loader = load_datasets() 



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
  



class custom_strategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, 
                      server_round:int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[BaseException],
                      ) -> Optional [fl.common.Parameters] :  #Tuple[Optional[Parameters], Dict[str, Scalar]]
                      
        if not results:
            return None, {}
            
        if self.accept_failures and failures:
            return None, {}
         
        key_name = "train_loss" if weight_strategy == "loss" else "wer"
        
        weights = None 
           
        if weight_strategy == 'num':
        
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]   
        
            weights = aggregate(weights_results)
        
        elif weight_strategy == "loss" or weight_strategy == "wer":
        
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics[key_name])
                for client, fit_res in results 
            ]
            
            weights = aggregate(weights_results)
            
        if weights is not None:
          print(f"One centralized training epoch will be performed")
          
          print('Done')
          print('Done')
          print('Done')
          print('Done')
          
          set_parameters(net, weights)
          
          
          
          train_asr(net, epochs=1, add_train=True)
          new_parameters = get_parameters(net)
           
          gc.collect()
          torch.cuda.empty_cache()
        
          return ndarrays_to_parameters(new_parameters), {}
            
            
           
           
class asr_client(fl.client.Client):

    def __init__(self, cid, net):
        self.cid = cid
        self.net = net

        
    def get_parameters(self, ins:GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        
        ndarrays: List[np.ndarray] = get_parameters(self.net)
        
        parameters = ndarrays_to_parameters(ndarrays)
        
        status = Status(code=Code.OK, message='Success')
        
        return GetParametersRes(status=status, parameters=parameters)
        
       
        
    def fit(self, ins: FitIns) -> FitRes:   #FitRes
    
        print(f"[Client {self.cid}] fit, config: {ins.config}")
        
        parameters_original = ins.parameters
        
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        
        set_parameters(self.net, ndarrays_original)
        
        norm_loss, num_examples, avg_wer = train_asr(self.net, epochs=5, add_train=False) 
        
        ndarrays_updated = get_parameters(self.net)
        
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        
        status = Status(code=Code.OK, message='Success')
        
        metrics = {"train_loss": norm_loss, "wer": avg_wer} 
        
        return  FitRes(status=status, parameters=parameters_updated, num_examples=num_examples, metrics=metrics)
        
        
    def evaluate(self, ins:EvaluateIns) -> EvaluateRes:
    
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")
        
        parameters_original = ins.parameters
        
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        
        set_parameters(self.net, ndarrays_original)
        
        norm_loss, num_examples, avg_wer = eval_asr(self.net)
        
        status = Status(code=Code.OK, message='Success')
        
        return EvaluateRes(status=status, loss = float(norm_loss), num_examples = num_examples, metrics = {"wer": float(avg_wer)})
        

def client_fn(cid) -> asr_client:
    
    n_encoder_layers = 2
    n_enc_replay = 6
    
    net = Early_conformer(src_pad_idx=src_pad_idx, n_enc_replay=n_enc_replay, d_model=d_model, enc_voc_size=enc_voc_size, dec_voc_size=dec_voc_size, max_len=max_len, dim_feed_forward=dim_feed_forward, n_head=n_heads, n_encoder_layers=n_encoder_layers, features_length=n_mels, drop_prob=drop_prob, depthwise_kernel_size=depthwise_kernel_size, device=device).to(device)
    return asr_client(cid, net)
    

#client_resources = None
#if device.type =="cuda":
    #client_resources = {"num_gpus":1}
    
client_resources = {
        "num_cpus": 8,
        "num_gpus": 1
    } 

def get_evaluate_fn() -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = Early_conformer(src_pad_idx=src_pad_idx,
                        N_enc_replay=n_enc_replay,
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
        set_parameters(net, parameters)
        net.to(device)

        norm_loss, num_examples, avg_wer = eval_asr(net)

        # return statistics
        return norm_loss, {"wer": avg_wer}

    return evaluate
    
def pre_trained_point(path):

    state_dict = torch.load(path)
    net.load_state_dict(state_dict)
    
    parameters = get_parameters(net)
    
    pre_trained = fl.common.ndarrays_to_parameters(parameters)
    
    del parameters
    gc.collect()
    torch.cuda.empty_cache()
    
    return pre_trained

pre_trained_path ="/falavi/slu256/trained_model/bpe_conformer_scratch/mod244-transformer"
pre_trained = pre_trained_point(pre_trained_path)
         
strategy = custom_strategy(
    initial_parameters = pre_trained,
    fraction_fit=1.0, 
    fraction_evaluate=0,#0.5, 
    min_fit_clients=1,#2,  
    min_evaluate_clients=2, 
    min_available_clients=1,#2,  
    #evaluate_fn=get_evaluate_fn(),
)
    
fl.simulation.start_simulation(client_fn=client_fn, 
                               num_clients=1, 
                               config=fl.server.ServerConfig(num_rounds=30),  
                               strategy=strategy, 
                               client_resources=client_resources)

 
 
 
 
 
  
 
 
 
 
 
 
 
 
 
 
        
        
