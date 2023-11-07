import numpy as np
from plotting import *
from utility_functions import *
import pandas as pd

TW_SIZE = 250 # Milliseconds
NORMALIZE = True # MinMax Normalization
MIN_CHANGES = 3 # Minimum amount of changes for annotation to be considered valid.

idg_annotators_session1 = ["BD7CE04E-99E3-7FA4-A15B-5625CD981638", "F868E6ED-CA85-FD16-942C-BE70BB997450", "1D8DFC94-778B-0969-9390-9F8A5B9C33EE", "89DA2498-EB31-04AF-2921-AEA70D626881", "49CAE400-6726-5DE5-398E-179FAD35B00A"]
idg_annotators_session2 = ['2EEEFB7F-9312-F08D-97CA-28A3B631D29E', "3865D7ED-91D3-6EF6-DB13-DD7C46D9034E", "BA3206C6-52F9-5900-5A81-2188D3E88B59", "ED4B536F-21B5-262C-61BC-A4396AFC016B", "62FF5C7F-4E6B-BB00-3FE0-F8752641A074"]
idg_annotators_session3 = ['5B89C3FA-A4AB-D90C-3FEF-885016FFB732', '738D09B9-A39F-819B-A237-C09448A2EB62', '9B3BB3E4-2AEB-482E-7316-F9715621C362', 'AE85275E-9C4D-D381-25F3-D823E24C0EF8', "F64F50A5-9F45-9753-0C9E-12AD4E483081"]
bad_annotators = [] # Catch any bad annotators (missing annotation) and remove them from the dataset.

MA_SMOOTH = False
MA_SIZE = 20

def avgfilter(signal, N=4):
  
  if signal.shape == ():
      return signal
  
  # Initialize the output array with NaN values
  output = np.empty_like(signal, dtype=np.float64)
  
  # Apply a moving average filter to each window of size k in the signal
  for k in range(1, N + 1):
      output[k - 1] = np.mean(signal[:k])
  for i in range(N, len(signal)):
      output[i] = np.mean(signal[i - N:i])
  return output


def get_max_times(participants):
    game_max_times = {}
    for _, participant_df in participants:
        games = participant_df.groupby('DatabaseName')
        for _, game_df in games:
            try:
                clean_game_name = game_df['OriginalName'].values[0].split("_")[1].split(".")[0]
            except:
                print(game_df['OriginalName'].values[0])
            if clean_game_name not in game_max_times or game_df["VideoTime"].max() > game_max_times[clean_game_name]:
                game_max_times[clean_game_name] = game_df["VideoTime"].max()
    return game_max_times


def separate_by_session_and_participant_engagement(df, engagement=False):
    pagan_sessions = df.groupby("PaganSession")
    data_dict = {}
    for session_name, session_df in pagan_sessions:

        participants = session_df.groupby('Participant')

        data_dict[session_name] = {"Expert": {}, "Mturk": {}}
        game_max_times = get_max_times(participants)

        for participant_id, participant_df in participants:
            groups = participant_df.groupby("Group")
            for group, group_df in groups:

                if group == "Expert" and session_name == "Session-1" and participant_id not in idg_annotators_session1:
                    continue
                elif group == "Expert" and session_name == "Session-2" and participant_id not in idg_annotators_session2:
                    continue
                elif group == "Expert" and session_name == "Session-3" and participant_id not in idg_annotators_session3:
                    continue

                games = group_df.groupby('DatabaseName')
                data_dict[session_name][group][participant_id] = {}
                game_counter = 0
                
                for _, game_df in games:
                    clean_game_name = game_df['OriginalName'].values[0].split("_")[1].split(".")[0]           
                    data_dict[session_name][group][participant_id][clean_game_name] = {
                        "VideoTime": game_df["VideoTime"].values,
                        "Value": game_df["Value"].values,
                        "StartTime": game_df['Timestamp'].values[0]
                    }
                    if data_dict[session_name][group][participant_id][clean_game_name]["VideoTime"][-1] < game_max_times[clean_game_name]:
                        data_dict[session_name][group][participant_id][clean_game_name]["VideoTime"] = np.append(game_df["VideoTime"].values, game_max_times[clean_game_name])
                        data_dict[session_name][group][participant_id][clean_game_name]["Value"] = np.append(game_df["Value"].values, 0)   
                    game_counter += 1
                if game_counter < 30 and engagement:
                    del data_dict[session_name][group][participant_id]
    return data_dict


def interpolate_data(data_dict, tw_size):
    interpolated_dict = {}
    counter = 0
    invalid = 0

    for session_id, session_data in data_dict.items():
        interpolated_dict[session_id] = {"Expert": {}, "Mturk": {}}
        for group_name, group_data in session_data.items():
            
            # print(f'{session_id}-{group_name}')
            lengths = []

            for participant_id, participant_data in group_data.items():
                interpolated_dict[session_id][group_name][participant_id] = {}
                for game_name, game_data in participant_data.items():
                    values = game_data['Value']
                    video_times = game_data['VideoTime']
                    df = pd.DataFrame({"VideoTime": video_times, "Value":values})
                    counter += 1
                    lengths.append(len(df))
                    if MIN_CHANGES != -1 and len(df) < MIN_CHANGES:
                        invalid += 1
                        interpolated_dict[session_id][group_name][participant_id][game_name] = None
                    else:
                        interpolated_dict[session_id][group_name][participant_id][game_name] = pagan_fulltrace(df, tw_size)

            # print(np.mean(lengths))
    print(f"Number of traces interpolated: {counter}")
    print(f"Number of invalid traces: {invalid}")
    return interpolated_dict


def pagan_fulltrace(pagan_trace, tw_size, time_col = 'VideoTime'):

    if len (pagan_trace) == 1:
        return None
    
    pagan_trace.loc[pagan_trace.index[-1], 'Value'] = pagan_trace.loc[pagan_trace.index[-2], 'Value']
    df = pagan_trace.copy(deep=True)
    df.loc[:, '[control]time_index'] = pd.to_timedelta((df[time_col]).astype('int32'), 'ms')
    df = df.set_index(df['[control]time_index'], drop=True)
    annotation = df.copy(deep=True)
    annotation = annotation.resample('{}ms'.format(tw_size)).mean(numeric_only=True)
    annotation = annotation.ffill(axis=0)

    removed_trailing = annotation['Value'].values[:-1]
    if removed_trailing[-1] == 0:
        removed_trailing = annotation['Value'].values[:-2]

    return removed_trailing

def normalize_data(data_dict):
    normalized_dict = {}
    for session_id, session_data in data_dict.items():
        normalized_dict[session_id] = {"Mturk": {}, "Expert": {}}
        for group_id, group_data in session_data.items():
            for participant_id, participant_data in group_data.items():
                normalized_dict[session_id][group_id][participant_id] = {}
                for game_name, game_data in participant_data.items():
                    values = np.array(game_data)
                    if np.min(values) != np.max(values):
                        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
                    else:
                        normalized_values = values

                    if MA_SMOOTH:
                        normalized_values = avgfilter(normalized_values)

                    normalized_dict[session_id][group_id][participant_id][game_name] = normalized_values.tolist()
    return normalized_dict


def remove_bad_sessions(visual_data, audio_data, engagement_data, sessions=[]):
    remove = {"Visual": [], "Audio": [], "Engagement": []}
    for session_id in audio_data.keys():

        for group_id in audio_data[session_id].keys():
            # Remove participants from the visual data dict who do not appear in the other two
            for participant_id in visual_data[session_id][group_id].keys():
                if participant_id not in engagement_data[session_id][group_id].keys() or participant_id not in audio_data[session_id][group_id].keys():
                    remove["Visual"].append((session_id, group_id, participant_id))

            # Remove participants from the audio data dict who do not appear in the other two
            for participant_id in audio_data[session_id][group_id].keys():
                if participant_id not in engagement_data[session_id][group_id].keys() or participant_id not in visual_data[session_id][group_id].keys():
                    remove["Audio"].append((session_id, group_id, participant_id))

            # Remove participants from the engagement data dict who do not appear in the other two (this should never happen but this is a sanity check)
            for participant_id in engagement_data[session_id][group_id].keys():
                if participant_id not in audio_data[session_id][group_id].keys() or participant_id not in visual_data[session_id][group_id].keys():
                    remove["Engagement"].append((session_id, group_id, participant_id))
              
    for ids in remove["Audio"]:
        del audio_data[ids[0]][ids[1]][ids[2]]
    for ids in remove["Visual"]:
        del visual_data[ids[0]][ids[1]][ids[2]]

    for ids in remove["Engagement"]:  
        del engagement_data[ids[0]][ids[1]][ids[2]]

    if len(sessions) != 0:
        for session in list(engagement_data.keys()):
            if session not in sessions:
                del audio_data[session]
                del visual_data[session]
                del engagement_data[session]

    return visual_data, audio_data, engagement_data

def format_time_difference(seconds):
    minutes, seconds = divmod(seconds, 60)
    try:
        return f"{int(minutes)} minutes and {int(seconds)} seconds"
    except:
        print("Error")
        return 


if __name__ == "__main__":

    engagement_df = pd.read_csv("./Raw_Engagement_Logs.csv")
    green_brightness_df = pd.read_csv("./Raw_Visual_Logs.csv")
    sound_pitch_df = pd.read_csv("./Raw_Audio_Logs.csv")
        
    engagement_data_dict = separate_by_session_and_participant_engagement(engagement_df, True)
    sound_pitch_data_dict = separate_by_session_and_participant_engagement(sound_pitch_df)
    green_brightness_data_dict = separate_by_session_and_participant_engagement(green_brightness_df)

    engagement_data = interpolate_data(engagement_data_dict, TW_SIZE)
    visual_data = interpolate_data(green_brightness_data_dict, TW_SIZE)
    audio_data = interpolate_data(sound_pitch_data_dict, TW_SIZE)

    if NORMALIZE:
        visual_data = normalize_data(visual_data)
        audio_data = normalize_data(audio_data)
        engagement_data = normalize_data(engagement_data)

    sessions = ['Session-1', 'Session-2']

    visual_data, audio_data, engagement_data = remove_bad_sessions(visual_data, audio_data, engagement_data)
    
    print(engagement_data.keys())

    np.save("Audio_Task.npy", audio_data)
    np.save("Visual_Task.npy", visual_data)
    np.save("Engagement_Task.npy", engagement_data)