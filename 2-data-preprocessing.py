import numpy as np
from utility_functions import *
import pandas as pd

"""
Basic signal processing script. Takes the CSV files outputted by the data loader and builds a dictionary of all the data, split by session, group, participant and game.
Bad participants (i.e. those that have not completed all three tasks) are removed from the dictionary.
This script also interpolates the data to a fixed time window size (1 second by default) and smoothes the data using a moving average filter (1 sample by default).
Finally, the traces can be normalized using min-max normalization on a trace-by-trace basis (enabled by default).
Note: The annotator ID's for the expert annotators are also hardcoded below, these are used to separate them from the crowdworkers who annotated in the same session.
This script outputs three dictionaries containing the processed signals as numpy file, one for each annotation type (visual, audio, engagement).
"""
idg_annotators_session1 = ["BD7CE04E-99E3-7FA4-A15B-5625CD981638", "F868E6ED-CA85-FD16-942C-BE70BB997450", "1D8DFC94-778B-0969-9390-9F8A5B9C33EE", "89DA2498-EB31-04AF-2921-AEA70D626881", "49CAE400-6726-5DE5-398E-179FAD35B00A"]
idg_annotators_session2 = ['2EEEFB7F-9312-F08D-97CA-28A3B631D29E', "3865D7ED-91D3-6EF6-DB13-DD7C46D9034E", "BA3206C6-52F9-5900-5A81-2188D3E88B59", "ED4B536F-21B5-262C-61BC-A4396AFC016B", "62FF5C7F-4E6B-BB00-3FE0-F8752641A074"]
idg_annotators_session3 = ['5B89C3FA-A4AB-D90C-3FEF-885016FFB732', '738D09B9-A39F-819B-A237-C09448A2EB62', '9B3BB3E4-2AEB-482E-7316-F9715621C362', 'AE85275E-9C4D-D381-25F3-D823E24C0EF8', "F64F50A5-9F45-9753-0C9E-12AD4E483081"]

def build_session_dict(df, engagement=False):
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
                if game_counter < 15 and engagement:
                    del data_dict[session_name][group][participant_id]
    return data_dict


def interpolate_data(data_dict, tw_size, MIN_CHANGES):
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
                        interpolated_dict[session_id][group_name][participant_id][game_name] = interpolate_trace(df, tw_size)
    # print(f"Number of traces interpolated: {counter}")
    # print(f"Number of invalid traces: {invalid}")
    return interpolated_dict


def interpolate_trace(pagan_trace, tw_size, time_col = 'VideoTime'):
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

def normalize_data(data_dict, MA_SIZE):
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
                        normalized_values = np.zeros_like(values)
                    if MA_SIZE > 1:
                        normalized_values = avgfilter(normalized_values, MA_SIZE)
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


def add_missing_games(engagement_data):
    all_games = sorted(set(game_name for session_data in engagement_data.values() for participant_data in session_data.values() for group_data in participant_data.values() for game_name in group_data.keys()))
    for session_id, groups in engagement_data.items():
        for group_id, participants in groups.items():
            for participant_id, games in participants.items():
                for game in all_games:
                    if game not in games:
                        engagement_data[session_id][group_id][participant_id][game] = None
    return engagement_data    


def execute(NORMALIZE, TW_SIZE, MIN_CHANGES, MA_SIZE, SESSIONS):

    engagement_df = pd.read_csv("./Processed Data/Raw_Engagement_Logs.csv")
    green_brightness_df = pd.read_csv("./Processed Data/Raw_Visual_Logs.csv")
    sound_pitch_df = pd.read_csv("./Processed Data/Raw_Audio_Logs.csv")
        
    engagement_data_dict = build_session_dict(engagement_df, True)
    sound_pitch_data_dict = build_session_dict(sound_pitch_df)
    green_brightness_data_dict = build_session_dict(green_brightness_df)

    engagement_data = interpolate_data(engagement_data_dict, TW_SIZE, MIN_CHANGES)
    visual_data = interpolate_data(green_brightness_data_dict, TW_SIZE, MIN_CHANGES)
    audio_data = interpolate_data(sound_pitch_data_dict, TW_SIZE, MIN_CHANGES)

    if NORMALIZE:
        visual_data = normalize_data(visual_data, MA_SIZE)
        audio_data = normalize_data(audio_data, MA_SIZE)
        engagement_data = normalize_data(engagement_data, MA_SIZE)

    visual_data, audio_data, engagement_data = remove_bad_sessions(visual_data, audio_data, engagement_data, SESSIONS)
    engagement_data = add_missing_games(engagement_data)

    np.save("./Processed Data/Session_Dict(Audio_Task).npy", audio_data)
    np.save("./Processed Data/Session_Dict(Visual_Task).npy", visual_data)
    np.save("./Processed Data/Session_Dict(Engagement_Task).npy", engagement_data)
