from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from scripts import train_asr, eval_asr
from dataset import *
import gc
import glob
import os
from models.model.early_exit import Early_encoder, Early_transformer, Early_conformer
from my_conf import *
import flwr as fl
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

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

parser = argparse.ArgumentParser(prog='Flower client for simulations')
parser.add_argument('--dataset', type=str, default="TEDLIUM", help='Dataset to be used for federated learning, at the moment TEDLIUM is available')
parser.add_argument('--modelname', type=str, default="TEDLIUM", help='Name of the model to be saved in trained_models... modelname-round-x.pth')
parser.add_argument('--centraltraining', action='store_true')
args=parser.parse_args()

###Global parameters of FL####
RAY_DEDUP_LOGS=0
_DATASET=args.dataset
_MODELNAME=args.modelname
centralizedTraining = args.centraltraining

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_encoder_layers = 2
n_enc_replay = 6
net = Early_conformer(src_pad_idx=src_pad_idx, n_enc_replay=n_enc_replay, d_model=d_model, enc_voc_size=enc_voc_size, dec_voc_size=dec_voc_size, max_len=max_len,
                      dim_feed_forward=dim_feed_forward, n_head=n_heads, n_encoder_layers=n_encoder_layers, features_length=n_mels, drop_prob=drop_prob,
                      depthwise_kernel_size=depthwise_kernel_size, device=device).to(device)

if _DATASET=="TEDLIUM":
    trainloaders, devloaders, centraloader, test_loader = load_datasets_TEDLIUM()
else:
    trainloaders, devloaders, centraloader, test_loader = load_datasets_LibriSpeech()

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)

class custom_strategy(fl.server.strategy.FedAvg):

    def __init__(
            self,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        state_dict = torch.load("./pretrained_models/librispeech1000.pth")

        #next lines if we want to use checkpoints... Must be parameterized.
        #list_of_files = [fname for fname in glob.glob("./trained_models/round-0*")]
        #latest_round_file = max(list_of_files, key=os.path.getctime)
        #print("Loading pre-trained model from: ", latest_round_file)
        #state_dict = torch.load(latest_round_file)
        net.load_state_dict(state_dict)
        torch.cuda.empty_cache()
        gc.collect()
        ndarrays = get_parameters(net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        sample_size=random.randint(min_num_clients, sample_size)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)

        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        #print("CONFIG.  sample:", sample_size,"min_num_clients:", min_num_clients, "n_clients:",n_clients)
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(self,
                      server_round:int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[BaseException],
                      ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]: #Optional [fl.common.Parameters] :  #Tuple[Optional[Parameters], Dict[str, Scalar]]

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

            params_dict = zip(net.state_dict().keys(), weights)
            state_dict = OrderedDict({k: torch.Tensor(np.array(v)) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            if centralizedTraining:
                print(f">>>>>>>>>>>>>>>>>>>>>>One centralized training epoch will be performed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                train_asr(net, epochs=1, trainloader=centraloader)
                # Save the model
            else:
                print(f"Centralized training not in place")

            torch.save(net.state_dict(), f"trained_models/{_MODELNAME}-round-{server_round}.pth")
            new_parameters = get_parameters(net)
            return ndarrays_to_parameters(new_parameters), {}
        else:
            print(f"returning None weights, something went wrongh during aggregation..... !!!!!!!!!!!!!!!")
            return ndarrays_to_parameters(weights), {}

class asr_client(fl.client.Client):

    def __init__(self, cid, net, trainloader, devloader):
        self.cid = cid
        self.net = net
        self.trainloader=trainloader
        self.devloader=devloader

    def get_parameters(self, ins:GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        ndarrays: List[np.ndarray] = get_parameters(self.net)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message='Success')
        #torch.cuda.empty_cache()
        #gc.collect()
        return GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:   #FitRes
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)

        #norm_loss, num_examples, avg_wer = train_asr(self.net, epochs=5, trainloader=self.trainloader)
        norm_loss, num_examples = train_asr(self.net, epochs=5, trainloader=self.trainloader)
        ndarrays_updated = get_parameters(self.net)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        status = Status(code=Code.OK, message='Success')
        metrics = {"train_loss": norm_loss}#, "wer": avg_wer}
        #torch.cuda.empty_cache()
        #gc.collect()
        return FitRes(status=status, parameters=parameters_updated, num_examples=num_examples, metrics=metrics)

    def evaluate(self, ins:EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_parameters(self.net, ndarrays_original)
        norm_loss, num_examples, avg_wer = eval_asr(self.net, devloader=self.devloader)
        status = Status(code=Code.OK, message='Success')
        #torch.cuda.empty_cache()
        #gc.collect()
        return EvaluateRes(status=status, loss = float(norm_loss), num_examples = num_examples, metrics = {"wer": float(avg_wer)})


def client_fn(cid) -> asr_client:
    n_encoder_layers = 2
    n_enc_replay = 6

    net = Early_conformer(src_pad_idx=src_pad_idx, n_enc_replay=n_enc_replay, d_model=d_model, enc_voc_size=enc_voc_size,\
                          dec_voc_size=dec_voc_size, max_len=max_len, dim_feed_forward=dim_feed_forward, n_head=n_heads,\
                          n_encoder_layers=n_encoder_layers, features_length=n_mels, drop_prob=drop_prob,\
                          depthwise_kernel_size=depthwise_kernel_size, device=device).to(device)

    trainloader = trainloaders[int(cid)]
    devloader= devloaders[int(cid)]

    return asr_client(cid, net, trainloader, devloader)


#client_resources = None
#if device.type =="cuda":
    #client_resources = {"num_gpus":1}

client_resources = {
    "num_cpus": 1,
    "num_gpus": 1,
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
        set_parameters(net, parameters)
        net.to(device)

        norm_loss, num_examples, avg_wer = eval_asr(net)

        # return statistics
        return norm_loss, {"wer": avg_wer}

    return evaluate

'''
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
'''

my_strategy = custom_strategy(
    #    initial_parameters = pre_trained,
    fraction_fit=0.1,
    fraction_evaluate=0,
    min_fit_clients=2,#15,
    min_evaluate_clients=2,
    min_available_clients=2,#15,
     #evaluate_fn=get_evaluate_fn(),
    )

fl.simulation.start_simulation(client_fn=client_fn,
                               num_clients=2351,
                               config=fl.server.ServerConfig(num_rounds=200),
                               strategy=my_strategy,
                               ray_init_args = {
                                   "include_dashboard": True, # we need this one for tracking
                                    "num_cpus": 1,
                                    "num_gpus": 1,
                                },
                                client_resources=client_resources)


