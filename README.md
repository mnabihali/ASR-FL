<div style="text-align: center;">
    <img src="https://github.com/mnabihali/ASR-FL/blob/main/assets/banner.webp" style="border-radius: 15px;" />
</div>

# Description
This repo contains our implementation for our research "Fed-EE: Federating Heterogeneous ASR Models using Early-Exit Architectures." The paper was accepted at the ENLSP (2023) workshop, hosted by NeurIPS, 2023. 
Also our under-review paper in Computer Speech & Language journal "Federating Dynamic Models using Early-Exit Architectures for Automatic Speech Recognition on Heterogeneous Clients"

# Early-Exit ASR-FL architecture
<div style="text-align: center;">
    <img src="https://github.com/mnabihali/ASR-FL/blob/main/assets/comp.png" style="border-radius: 15px;" />
</div>

# Scripts Description
`client.py` : Basic code of a client with fixed speakers
`script.py` : Training and Evaluation loop (based on client.py)
`inference.py` : Performs the inference on both dev and test sets (called by script.py)
<--------------------------------------------------------------------------------------->
`dynamic_client.py`: Code of a client with dynamic speakers
`dynamic_script.py`: Training and evaluation loop (based on dyamic_client.py)
<--------------------------------------------------------------------------------------->
`conf.py`: Model and training configuration parameters
`dataset.py`: Dataloaders and loading functions for Librispeech and TedLium-3 datasets
`tedlium_dataset.py`: Advanced dataloader for TedLium-3

# How to use
For training --> Run `client.py`  (it runs a single process that simulates multiple client.py)
For inference --> Run `inference.py -- model model_name --round round_number` (it evaluates the saved central model on the dev and test sets)

# Publication
```
@inproceedings{nawar2023fed,
  title={Fed-EE: Federating Heterogeneous ASR Models using Early-Exit Architectures},
  author={Nawar, Mohamed Nabih Ali Mohamed and Falavigna, Daniele and Brutti, Alessio},
  booktitle={Proceedings of 3rd Neurips Workshop on Efficient Natural Language and Speech Processing},
  year={2023}
}
```
```
@misc{ali2024federatingdynamicmodelsusing,
      title={Federating Dynamic Models using Early-Exit Architectures for Automatic Speech Recognition on Heterogeneous Clients}, 
      author={Mohamed Nabih Ali and Alessio Brutti and Daniele Falavigna},
      year={2024},
      eprint={2405.17376},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.17376}, 
}
```

# Acknowledgment
* We acknowledge the support of the PNRR project FAIR - Future AI Research (PE00000013), under the NRRP MUR program funded by the NextGenerationEU.
