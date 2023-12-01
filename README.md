# FPS-Engagement-Dataset
## Embedding extraction
General prerequisites:
1. Clone this repository
2. Install the following libraries:
   - PyTorch: https://pytorch.org/get-started/locally/
   - Transformers: https://huggingface.co/docs/transformers/installation
   - av: https://pypi.org/project/av/
## VideoMAE embedding extraction
Data can be found at: https://drive.google.com/file/d/1nKzAZATJFhXr1OXmhmRRrmg0P2WQYbRL/view?usp=drive_link
### Running
#### On local machine in PyCharm
1. Open embedding extraction script: `latent_extraction.extract_videomae_latents.py`
2. Set the following parameters:
```
ROOT_DIRECTORY - Path to unizipped sessions.zip
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
## BEATs embedding extraction
Data will be added to drive shortly
### Running
#### On local machine in PyCharm
1. Clone unilm/beats repository: https://github.com/microsoft/unilm/tree/master/beats
2. Copy embedding extraction script into beats repository `latent_extraction.extract_videomae_latents.py`
3. Download a pre-trained checkpoint: https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D
4. Set the following parameters in the script
```
ROOT_DIRECTORY - Path to folder containing audio files (.wav) organized by session
SAVE_DIRECTORY
CHECKPOINT_PATH = Path to pre-trained checkpoint
SESSIONS = Sessions that emebeddings will be extracted for (['1', '2', '3', '7'] is default)
WINDOW - Non-overlapping sampling window in seconds. 
```
#### On Banshee (recommended), Beast etc. in terminal
1. Follow above mentioned steps
2. `cd beats`
3. Add this project to PYTHONPATH by running `export PYTHONPATH='~/beats'`
4. Run with command `python -m extract_beats_latents`