from pathlib import Path
from models.videomae.videomae_latente_extractor import extract_latents


FILE_PATH = r"C:\Users\rasajski\Downloads\apexlegends\0.mp4"
SAVE_DIRECTORY = Path(r"C:\Users\rasajski\Downloads\save")

SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Takes about 30 seconds per video on GPU, and 2.5 minutes per video on CPU
    extract_latents(FILE_PATH, SAVE_DIRECTORY, mean_pool=False, skip_existing=True)
