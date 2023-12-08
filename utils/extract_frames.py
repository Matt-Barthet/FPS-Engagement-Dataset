import av
import numpy as np


def _create_windows(array, window=30, stride=30, sample_rate=16):
    nrows = ((array.size - window) // stride) + 1
    if window >= stride:
        linspace_end = window
    else:
        linspace_end = stride
    return array[stride * np.arange(nrows)[:, None] + np.linspace(0, linspace_end - 1, num=sample_rate).astype(int)]


def sample_frames(file_path, window: int = 1,  offset: int = 1, frames_to_sample: int = 16):
    """
    Decode the video with PyAV decoder.
    Args:
        file_path (str): path to video in string format.
        window (int): sampling window in seconds
        offset (int): number of seconds each window should be offset by
        frames_to_sample: (int): number of frames to sample within each time window

    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_windows, 16, height, width, 3).
    """
    # Code was inspired by HF example:
    # https://huggingface.co/docs/transformers/model_doc/videomae#transformers.VideoMAEForVideoClassification.forward.example
    # Official PyAv documentation: https://pyav.org/docs/stable/index.html
    container = av.open(file_path)

    # https://pyav.org/docs/develop/api/stream.html#av.stream.Stream.average_rate
    average_rate = container.streams.video[0].average_rate
    numerator, denominator = average_rate.numerator, average_rate.denominator
    fps = round(numerator / denominator)

    container.seek(0)
    all_frames = np.array(list(container.decode(video=0)))
    container.close()

    all_indices = _create_windows(np.arange(len(all_frames)), window * fps, offset * fps, frames_to_sample)
    all_frames = all_frames[all_indices]
    return np.stack([np.stack([x.to_ndarray(format="rgb24") for x in frames]) for frames in all_frames])
