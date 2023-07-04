import os
from pathlib import Path
from typing import Tuple, Union
import random
import torchaudio
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (extract_archive,)
import torch
from collections import OrderedDict
from typing import List, Tuple
from my_conf import *
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset, RandomSampler, Subset
import torchaudio.transforms as T
from tqdm import tqdm
import numpy as np
from util.data_loader import collate_fn,  collate_infer_fn  
import sys
from tedlium_dataset import TEDLIUM

URL = "train-clean-100"
URL_TS = "test-clean"
FOLDER_IN_ARCHIVE = "LibriSpeech"
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",  # noqa: E501
    "http://www.openslr.org/resources/12/dev-other.tar.gz": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",  # noqa: E501
    "http://www.openslr.org/resources/12/test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",  # noqa: E501
    "http://www.openslr.org/resources/12/test-other.tar.gz": "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",  # noqa: E501
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",  # noqa: E501
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2",  # noqa: E501
}


def load_librispeech_item(
    fileid: str, path: str, ext_audio: str, ext_txt: str
) -> Tuple[Tensor, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    #file_text = os.path.join(path, speaker_id, chapter_id, file_text)
    #print("PATH:",path)
    file_text = os.path.join(path, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id

    file_audio = fileid_audio + ext_audio
    #file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
    file_audio = os.path.join(path, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return (
        waveform,
        sample_rate,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


class LIBRISPEECH(Dataset):
    """Create a Dataset for LibriSpeech.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"dev-clean"``, ``"dev-other"``, ``"test-clean"``,
            ``"test-other"``, ``"train-clean-100"``, ``"train-clean-360"`` and
            ``"train-other-500"``. (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(
            self, root: Union[str, Path], url: str = URL, folder_in_archive: str = FOLDER_IN_ARCHIVE, download: bool = False, speaker_federated: str = ""
    ) -> None:

        if url in [
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        #basename = basename.split(".")[0]+"/"+speaker_federated
        basename = speaker_federated
        #print("BASE:",basename)
        
        folder_in_archive = os.path.join(folder_in_archive, basename)
        
        self._path = os.path.join(root, folder_in_archive)
        #print("PATH:",self._path)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                extract_archive(archive)

        #self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))
        #for p in Path(self._path).glob("*/*" + self._ext_audio):
        #    print("FILE:",str(p.stem), p)
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*" + self._ext_audio))

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)``
        """
        fileid = self._walker[n]
        #print("LOAD",fileid, self._path, self._ext_audio, self._ext_txt)
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)


    def __len__(self) -> int:
        return len(self._walker)


def load_datasets_():
    train_list = [100, 110, 119, 128, 151, 16, 17, 19, 200, 26, 32, 44, 52, 60, 75, 83, 92, 101, 111, 122, 133, 152, 161, 173, 192, 201, 27, 36, 45, 54, 62, 77, 85, 93, 102, 112, 123, 14, 153, 163, 175, 196, 202, 28, 37, 46, 55, 64, 78, 87, 94,      103, 114, 125, 147, 154, 166, 176, 198, 22, 29, 38, 47, 56, 65, 79, 89, 98, 104, 115, 126, 149, 157, 167, 177, 199, 23, 30, 39, 49, 57, 66, 81, 90, 9999, 107, 118, 127, 150, 159, 168, 188, 20, 25, 31, 40, 51, 58, 70, 82, 91]

    dev_list = [1272, 1462, 1673, 174, 1919, 1988, 1993, 2035, 2078, 2086, 2277, 2412, 2428, 251, 2803, 2902, 3000, 2081, 3170, 3536, 3576, 3752]

    x = random.choice(train_list) 
    y = random.choice(dev_list)

    tr_spk_pth = '/flower/data/LibriSpeech/train/{}/'.format(x)
    
    print('train path', tr_spk_pth)
  
    trgl_spk_pth = '/flower/data/LibriSpeech/train/9999/'
  
    print('trgl path', trgl_spk_pth)
    
    dev_spk_pth = '/flower/data/LibriSpeech/dev-clean/{}/'.format(y)
  
    print('dev path', dev_spk_pth)


    train_dataset = LIBRISPEECH("/flower/data/", url=URL, download=False, speaker_federated=tr_spk_pth)
    traingl_dataset = LIBRISPEECH("/flower/data/", url=URL, download=False, speaker_federated=trgl_spk_pth)
    dev_dataset = LIBRISPEECH("/flower/data/", url=URL_TS, download=False, speaker_federated=dev_spk_pth)
    test_dataset = torchaudio.datasets.LIBRISPEECH("/flower/data/", url="test-clean", download=False)

    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    devloader = DataLoader(dev_dataset, batch_size=1, shuffle = False, collate_fn=collate_infer_fn, num_workers=0)
    trainloader_gl = DataLoader(traingl_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=1, num_workers=0,
                                            shuffle=False, collate_fn=collate_infer_fn)
  
    return trainloader, devloader, trainloader_gl, testloader

def load_datasets():
    stm_path="/falavi/corpora/TEDLIUM_release-3/legacy/train/stm"
    

    train_list=sorted(os.listdir("/falavi/corpora/TEDLIUM_release-3/legacy/train/stm"))
    #random.shuffle(train_list)

    dev_list=sorted(os.listdir("/falavi/corpora/TEDLIUM_release-3/legacy/dev/stm"))
    #random.shuffle(dev_list)

    trgl_spk_pth = '/flower/data/LibriSpeech/train/9999_/'
    #print('trgl path', trgl_spk_pth)
    trainloaders = []
    devloaders = []

    for tr_spk_pth in train_list:
        #print('train path', tr_spk_pth)
        train_dataset= TEDLIUM("/falavi/corpora/", release="release3", subset="train", download=False, federated_speaker=tr_spk_pth)
        trainloaders.append(DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0))
    for dev_spk_pth in dev_list:
        #print('dev path', dev_spk_pth)
        dev_dataset= TEDLIUM("/falavi/corpora/", release="release3", subset="dev", download=False, federated_speaker=dev_spk_pth)
        devloaders.append(DataLoader(dev_dataset, batch_size=1, shuffle = False, collate_fn=collate_infer_fn, num_workers=0))
        
    traingl_dataset = LIBRISPEECH("/flower/data/", url=URL, download=False, speaker_federated=trgl_spk_pth)
    test_dataset = torchaudio.datasets.TEDLIUM("/falavi/corpora/", release="release3", subset="test", download=False)

    trainloader_gl = DataLoader(traingl_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, pin_memory=False, batch_size=1, num_workers=0, shuffle=False, collate_fn=collate_infer_fn)
    devloaders = trainloaders
    return trainloaders, devloaders, trainloader_gl, testloader





'''  
trainloader, testloader = load_datasets()
for i, batch in enumerate(tqdm(testloader)):
  print(i)
  pass
  testset = torchaudio.datasets.LIBRISPEECH("/home/mnabih/fedspeakers", url="test-clean", download=False)
  num_test_samples = np.random.randint(low=500, high=2000)
  print(num_test_samples)
  num_test_samples = 100
  sample_ds = Subset(testset, np.arange(num_test_samples))
  sample_sampler = RandomSampler(sample_ds)
  testloader = DataLoader(sample_ds, sampler=sample_sampler, batch_size=BATCH_SIZE_te, shuffle = False, collate_fn=collate_infer_fn, num_workers=0)
  testloader = DataLoader(testset, batch_size=BATCH_SIZE_te, shuffle = False, collate_fn=collate_infer_fn, num_workers=0)
  ID_list = [441, 4640, 8098]
  x = random.choice(ID_list)
  spk_pth = '/home/mnabih/federated_ASR/LibriSpeech/train-clean-100/{}/'.format(x)
  print(spk_pth)
  train_dataset = LIBRISPEECH("/home/mnabih/federated_ASR/", url="", download=False, speaker_federated='/home/mnabih/federated_ASR/LibriSpeech/train-clean-100/441/')
  test_dataset = LIBRISPEECH("/home/mnabih/federated_ASR/", url="", download=False, speaker_federated='/home/mnabih/federated_ASR/LibriSpeech/test-clean/121/')
  for x in test_dataset:
    print(x)
'''
