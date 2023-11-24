import torch
import numpy as np

from pathlib import Path
from transformers import AutoImageProcessor, VideoMAEModel
from utils.extract_frames import sample_frames
from tqdm import tqdm
from typing import Union
import os

np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)


def _extract_file_name(file_full_path):
    return Path(file_full_path).stem


def extract_latents(input_video_filepath: str, output_directory: Union[str, Path], mean_pool: bool = True,
                    skip_existing: bool = False):
    """
    Extracts latents from video and saves in specified directory
    Args:
        input_video_filepath (str): string path to video, does not accept pathlike
        output_directory (str|pathlib.Path): string or pathlike object where generated latents will be saved
        mean_pool (bool): perform mean pooling on latents before saving, or save the whole latent
        skip_existing (bool): skip already saved latents

    Returns:
        None
    """
    file_name = _extract_file_name(input_video_filepath)
    videos = sample_frames(input_video_filepath)
    for idx, video in tqdm(enumerate(videos)):
        save_filename = f"{output_directory}/{file_name}_{idx}.pt"
        if os.path.isfile(save_filename) and skip_existing:
            continue
        inputs = image_processor(list(video), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        if mean_pool:
            latent = last_hidden_states.mean(1)
        else:
            latent = last_hidden_states
        torch.save(latent, save_filename)
