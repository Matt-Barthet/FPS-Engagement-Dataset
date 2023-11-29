# FPS-Engagement-Dataset
 
## VideoMAE latent extraction
Prerequisites:
1. Clone this repository
2. Install the following libraries:
   - PyTorch: https://pytorch.org/get-started/locally/
   - Transformers: https://huggingface.co/docs/transformers/installation
   - av: https://pypi.org/project/av/
### Running
#### On local machine in PyCharm
1. Open latent extraction script: `latent_extraction.extract_videomae_latents.py`
2. Set the following parameters:
```
ROOT_DIRECTORY - Path to unizipped sessions.zip from drive: https://drive.google.com/file/d/1nKzAZATJFhXr1OXmhmRRrmg0P2WQYbRL/view?usp=drive_link
SAVE_DIRECTORY
MEAN_POOL - VideoMAE returns latents of shape (SEQUENCE_LENGTH x 768), where SEQUENCE_LENGHT is usually 1536. Mean pooling compresses these latents to shape (768). Full size latents take up ~5MB on disc when saved, mean pooled latents are ~5KB.
SKIP - Flag for skipping already created latents. Useful when the whole conversion process can't be performed at once.
WINDOW - Non-overlapping sampling window in seconds. 
```
In addition to these, the pre-trained model version can be changed in `models.videomae.videomae_latent_extractor.py`. Current checkpoint is `MCG-NJU/videomae-base`
#### On Banshee (recommended), Beast etc. in terminal
1. Set above mentioned parameters
2. `cd FPS-Engagement-Dataset`
3. Add this project to PYTHONPATH by running `export PYTHONPATH='~/FPS-Engagement-Dataset'`
4. Run with command `python -m latent_extraction.extract_videomae_latents`
   - If you get an error similar to: `PermissionError: [Errno 13] Permission denied: '/media/storage/model_checkpts/huggingface/.locks`, don't use `sudo`, just change transformers library cache location with `export TRANSFORMERS_CACHE='/home/YOUR_USERNAME/.cache'`