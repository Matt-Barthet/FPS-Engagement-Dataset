import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from utils.extract_frames import _create_windows
from torchaudio import transforms
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIRECTORY = Path(r"")
SAVE_DIRECTORY = Path(r"")
CHECKPOINT_PATH = Path("")
SESSIONS = ['1', '2', '3', '7']
WINDOW = 1
OFFSET = 1

SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def extract_latents(in_file_path, out_file_dir, window, offset):
    out_file_path = Path(out_file_dir) / in_file_path.stem
    waveform, sample_rate = torchaudio.load(str(in_file_path), normalize=True)
    mono_waveform = torch.mean(waveform, dim=0, keepdim=True)

    interval_length = int(sample_rate * window)
    interval_offset = int(sample_rate * offset)

    intervals = _create_windows(np.arange(mono_waveform.shape[1]), interval_length, interval_offset, sample_rate)
    for idx, interval in enumerate(tqdm(intervals)):
        interval_start = interval[0]
        transform = transforms.MFCC(sample_rate=sample_rate, n_mfcc=13,
                                    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
                                    ).to(device)
        mfcc = transform(mono_waveform.squeeze()[interval_start:interval_start + interval_length].to(
                device).unsqueeze(dim=0))
        torch.save(mfcc, f"{out_file_path}_{idx}.pt")


if __name__ == '__main__':
    for in_subdirectory_path in ROOT_DIRECTORY.glob("*"):
        in_subdirectory_name = in_subdirectory_path.name
        if in_subdirectory_name not in SESSIONS:
            continue
        out_subdirectory_path = SAVE_DIRECTORY / in_subdirectory_name
        out_subdirectory_path.mkdir(parents=True, exist_ok=True)
        print(f"Generating latents for session {in_subdirectory_name}:")
        for in_file_path in in_subdirectory_path.glob("*.wav"):
            extract_latents(in_file_path, out_subdirectory_path, WINDOW, OFFSET)
