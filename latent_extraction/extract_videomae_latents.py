from pathlib import Path

from models.videomae.videomae_latent_extractor import extract_latents

# Download data from this link: https://drive.google.com/file/d/1nKzAZATJFhXr1OXmhmRRrmg0P2WQYbRL/view?usp=drive_link
# Set ROOT_DIRECTORY to the path where you downloaded and unpacked above link
ROOT_DIRECTORY = Path(r"ROOT_DIRECTORY")
SAVE_DIRECTORY = Path(r"SAVE_DIRECTORY")
# VideoMAE gives latents of shape (sequence_length x 768) to compact this to just 768 mean-pooling is performed
# If you want to skip mean-pooling set this flag to False. WARNING: One full sized latent is 4.8MB on disc (there are
# 60 / window_size latents per video (if windows_size = 1s, that 60 latents per video))
MEAN_POOL = True
# Skip already existing latents
SKIP = False
# Sampling window in seconds
WINDOW = 1

SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Takes about 30 seconds per video on GPU, and 2.5 minutes per video on CPU
    for in_subdirectory_path in ROOT_DIRECTORY.glob("*"):
        in_subdirectory_name = in_subdirectory_path.name
        out_subdirectory_path = SAVE_DIRECTORY / in_subdirectory_name
        out_subdirectory_path.mkdir(parents=True, exist_ok=True)
        print(f"Generating latents for session {in_subdirectory_name}:")
        for in_file_path in in_subdirectory_path.glob("*.mp4"):
            extract_latents(str(in_file_path), out_subdirectory_path, window=WINDOW, mean_pool=MEAN_POOL,
                            skip_existing=SKIP)
