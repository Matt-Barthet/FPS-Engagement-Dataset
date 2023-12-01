import torch
from BEATs import BEATs, BEATsConfig
import torchaudio
from tqdm import tqdm
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIRECTORY = Path(r"")
SAVE_DIRECTORY = Path(r"")
CHECKPOINT_PATH = Path("")
SESSIONS = ['1', '2', '3', '7']
WINDOW = 1

SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    cfg = BEATsConfig(checkpoint['cfg'])
    weights = checkpoint['model']

    model = BEATs(cfg)
    model.load_state_dict(weights)
    model = model.to('cuda')
    model.eval()

    return model


def extract_latents(in_file_path, out_file_dir, model, window):
    out_file_path = Path(out_file_dir) / in_file_path.stem
    waveform, sample_rate = torchaudio.load(str(in_file_path))
    mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
    interval_length = int(sample_rate * window)

    for idx, interval_start in enumerate(tqdm(range(0, mono_waveform.shape[1], interval_length))):
        with torch.no_grad():
            audio_sample = mono_waveform.squeeze()[interval_start:interval_start + interval_length].to(
                device).unsqueeze(dim=0)
            # skip samples that don't have minimum length
            if audio_sample.shape[1] < interval_length:
                continue
            probs = model.extract_features(audio_sample, padding_mask=None)[0]
        torch.save(probs, f"{out_file_path}_{idx}.pt")


if __name__ == '__main__':
    beats_model = _load_model(CHECKPOINT_PATH)
    for in_subdirectory_path in ROOT_DIRECTORY.glob("*"):
        in_subdirectory_name = in_subdirectory_path.name
        out_subdirectory_path = SAVE_DIRECTORY / in_subdirectory_name
        out_subdirectory_path.mkdir(parents=True, exist_ok=True)
        print(f"Generating latents for session {in_subdirectory_name}:")
        for in_file_path in in_subdirectory_path.glob("*.wav"):
            extract_latents(in_file_path, out_subdirectory_path, beats_model, WINDOW)
