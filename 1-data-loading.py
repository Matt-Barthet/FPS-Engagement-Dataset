import pandas as pd
import os
import numpy as np


def latest_session(participant_data):
    all_sessions = participant_data['SessionID'].unique()
    timesteps = []
    for i in all_sessions:
        session_trace = participant_data[participant_data['SessionID'] == i]
        timesteps.append(session_trace['Timestamp'].to_numpy()[0])
    return all_sessions[np.argmax(timesteps)]


def filter_duplicate_sessions(df, multipleGames=False):
    sessions = df['PaganSession'].unique()
    latest_sessions_df = pd.DataFrame()  
    for session in sessions:
        session_df = df[df['PaganSession'] == session]
        participants = session_df['Participant'].unique()
        for participant in participants:
            participant_df = session_df[session_df['Participant'] == participant]

            if multipleGames: # if engagement vs qa tests
                games = participant_df['DatabaseName'].unique()
                for game in games:
                    game_df = participant_df[participant_df['DatabaseName'] == game]
                    latest_session_df = game_df[game_df['SessionID'] == latest_session(game_df)]
                    latest_sessions_df = pd.concat([latest_sessions_df, latest_session_df], ignore_index=True)
            else:
                latest_session_df = participant_df[participant_df['SessionID'] == latest_session(participant_df)]
                latest_sessions_df = pd.concat([latest_sessions_df, latest_session_df], ignore_index=True)
    return latest_sessions_df


def execute(FILTER_DUPLICATES=True):

    engagement_df_list = []
    brightness_df_list = []
    pitch_df_list = []

    for filename in os.listdir("./Pagan_CSVs/"):
        if filename.endswith('.csv'):
            session_name = filename.split("_")[0]
            group_name = session_name.split("-")[-1]
            session_name = session_name.removesuffix(f'-{group_name}')
            df = pd.read_csv(f"./Pagan_CSVs/{filename}").reset_index(drop=True)
            df['PaganSession'] = session_name 
            df['Group'] = group_name
            if 'Engagement' in filename:
                engagement_df_list.append(df)
            elif 'Green-Brightness' in filename:
                brightness_df_list.append(df)
            elif 'Sound-Pitch' in filename:
                pitch_df_list.append(df)

    engagement_df = pd.concat(engagement_df_list, ignore_index=True)
    brightness_df = pd.concat(brightness_df_list, ignore_index=True)
    pitch_df = pd.concat(pitch_df_list, ignore_index=True)

    if FILTER_DUPLICATES:
        green_brightness_df = filter_duplicate_sessions(brightness_df)
        sound_pitch_df = filter_duplicate_sessions(pitch_df)
        engagement_df = filter_duplicate_sessions(engagement_df, True)

    engagement_df.to_csv("./Processed Data/Raw_Engagement_Logs.csv")
    green_brightness_df.to_csv("./Processed Data/Raw_Visual_Logs.csv")
    sound_pitch_df.to_csv("./Processed Data/Raw_Audio_Logs.csv")
    