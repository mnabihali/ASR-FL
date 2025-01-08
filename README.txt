<div style="text-align: center;">
    <img src="https://github.com/mnabihali/ASR-FL/blob/main/assets/comp.png" alt="ASR-FL" style="border-radius: 15px;" />
</div>

client.py: basic code of a client with fixed speakers
script.py: training and evaluation loop (based on client.py)
inference.py: perform inference on dev and test sets (called by script.py)


dynamic_client.py: code of client with dynamic speakers
dynamic_script.py: training and evaluation loop (based on dyamic_client.py)

conf.py: configuration parameters
dataset.py: dataloaders and loading functions for librispeech and TedLium3
tedlium_dataset.py: dataloader for TedLium

To use it:
run client.py (it runs a single process that simulates multiple client.py)
