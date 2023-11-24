import av
import numpy as np


def _sample_indices(num_total_frames: int, window: int = 1, fps: int = 30, num_to_sample: int = 16):
    """
    Samples gifs from video in specified time window
    Args:
        num_total_frames (int): total number of frames in video
        window (int): sampling window in seconds
        fps (int): frames per second
        num_to_sample (int): number of frames to sample from each window

    Returns:
        result (List[List[int]]) Nested list of indices with shape [NUM_WINDOWS, num_to_sample]
        Examples:
            Video has 1800 frames, sampling window is 1s, fps is 30, 16 frames sampled for each window
            result will have shape [60, 16]. NUM_WINDOWS = num_total_frames / (fps * window) = 1800 / (30 * 1) = 60
            num_to_sample = 16
    """
    num_windows = round(num_total_frames / (fps * window))
    all_indices = []

    for idx in np.arange(num_windows):
        first_frame = fps * window * idx
        last_frame = first_frame + fps * window

        if last_frame >= num_total_frames:
            last_frame = num_total_frames - 1

        indices = np.linspace(first_frame, last_frame, num=num_to_sample).astype(int)
        all_indices.append(indices)

    return all_indices


def sample_frames(file_path, window: int = 1, frames_to_sample: int = 16):
    """
    Decode the video with PyAV decoder.
    Args:
        file_path (str): path to video in string format.
        window (int): sampling window in seconds
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

    all_indices = _sample_indices(len(all_frames), window, fps, frames_to_sample)
    all_frames = all_frames[all_indices]
    return np.stack([np.stack([x.to_ndarray(format="rgb24") for x in frames]) for frames in all_frames])
