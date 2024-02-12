import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def latest_session(participant_data):
    all_sessions = participant_data['SessionID'].unique()
    timesteps = []
    for i in all_sessions:
        session_trace = participant_data[participant_data['SessionID'] == i]
        timesteps.append(session_trace['Timestamp'].to_numpy()[0])
    return all_sessions[np.argmax(timesteps)]


def per_participant_analysis(engagement_data, raw_data, game_names):
    participant_dict = {}
    for game in game_names:
        game_raw = raw_data[raw_data["OriginalName"].str.contains(game)]
        sessions = game_raw["PaganSession"].unique()
        for session in sessions:
            session_raw = game_raw[game_raw["PaganSession"] == session]
            participants = session_raw["Participant"].unique()
            for participant in participants:
                if participant in engagement_data[session]['Expert']:
                    participant_raw = session_raw[session_raw["Participant"] == participant]
                    latest_session_df = participant_raw[participant_raw['SessionID'] == latest_session(participant_raw)]
                    if f"{session}-{participant}" not in participant_dict.keys():
                        # participant_dict[f"{session}-{participant}"] = [np.max(latest_session_df["Value"].values) - np.min(latest_session_df["Value"].values)]
                        participant_dict[f"{session}-{participant}"] = [len(latest_session_df['Value'])]
                    else:
                        # participant_dict[f"{session}-{participant}"].append(np.max(latest_session_df["Value"].values) - np.min(latest_session_df["Value"].values))
                        participant_dict[f"{session}-{participant}"].append(len(latest_session_df['Value']))

    sorted_keys = sorted(participant_dict.keys())    
    for game in sorted_keys:
        print(f"{game},", end="")

    print()
    for game in sorted_keys:
        to_print = ""
        for value in participant_dict[game]:
            to_print+= f"{value},"
        print(f"{to_print}")

    print()
    for game in sorted_keys:
        print(f"{np.round(1.96*np.std(participant_dict[game])/np.sqrt(len(participant_dict[game])), 2)},", end="")

    exit()
    games = {}
    for session_id, session_data in engagement_data.items():
        counter = 0
        group_data = session_data["Expert"].items()
        for participant_id, participant_data in group_data:
            for game in participant_data.keys():
                if f"{session_id}-{participant_id}" not in games.keys():
                    games[f"{session_id}-{participant_id}"] = [np.std(participant_data[game])]
                else:
                    games[f"{session_id}-{participant_id}"].append(np.std(participant_data[game]))

    for game in games.keys():
        print(f"{game},", end="")

    print()
    for game in games.values():
        print(f"{np.round(np.mean(game), 4)},", end="")

    print()
    for game in games.values():
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)},", end="")

if __name__ == "__main__":
    pagan_sessions = ["Session-1", "Session-2", "Session-3", "Session-7"]
    raw_data = pd.read_csv("Processed Data/Raw_Engagement_Logs.csv")
    raw_data = raw_data[raw_data["Group"] == "Expert"]
    raw_data = raw_data[raw_data["PaganSession"].isin(pagan_sessions)]

    engagement_data = np.load("Processed Data/Session_Dict(Engagement_Task).npy", allow_pickle=True).item()
    
    game_names = set()
    for session_id, session_data in engagement_data.items():
        counter = 0
        group_data = session_data["Expert"].items()
        for participant_id, participant_data in group_data:
            for game in participant_data.keys():
                game_names.add(game)
    
    per_participant_analysis(engagement_data, raw_data, game_names)
    exit()

    game_names = sorted(game_names)
    changes = {}
    for game in game_names:
        game_raw = raw_data[raw_data["OriginalName"].str.contains(game)]
        sessions = game_raw["PaganSession"].unique()
        for session in sessions:
            session_raw = game_raw[game_raw["PaganSession"] == session]
            participants = session_raw["Participant"].unique()
            for participant in participants:
                participant_raw = session_raw[session_raw["Participant"] == participant]
                if game not in changes.keys():
                    # changes[game] = [np.max(participant_raw["Value"].values) - np.min(participant_raw["Value"].values)]
                    changes[game] = [len(participant_raw)]
                else:
                    changes[game].append(len(participant_raw))
                    # changes[game].append(np.max(participant_raw["Value"].values) - np.min(participant_raw["Value"].values))

    means = [np.round(np.mean(game), 2) for game in changes.values()]
    cis = [np.round(1.96*np.std(game)/np.sqrt(len(game)), 2) for game in changes.values()]

    for game in sorted(changes.keys()):
        print(f"{game},", end="")
    print()
    
    for game in changes.values():
        print(game)
        # print(f"{np.round(np.mean(game), 4)},", end="")

    print()
    for game in changes.values():
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)},", end="")

    """games = {}
    for session_id, session_data in engagement_data.items():
        counter = 0
        group_data = session_data["Expert"].items()
        for participant_id, participant_data in group_data:
            for game in participant_data.keys():
                if game not in games.keys():
                    games[game] = [np.std(participant_data[game])]
                else:
                    games[game].append(np.std(participant_data[game]))

    for game in sorted(games.keys()):
        # print(f"{game},", end="")
        print(f"{game}-{np.round((games[game]), 4)},")
    print()
    
    for game in games.values():
        print(f"{np.round(np.mean(game), 4)},", end="")

    print()
    for game in games.values():
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)},", end="")
    """

    """participant = {}
    for session_id, session_data in engagement_data.items():
        counter = 0
        group_data = session_data["Expert"].items()
        for participant_id, participant_data in group_data:
            for game in participant_data.keys():
                if game not in games.keys():
                    games[game] = [np.std(participant_data[game])]
                else:
                    games[game].append(np.std(participant_data[game]))

    for game in games.keys():
        print(f"{game},", end="")

    print()
    for game in games.values():
        print(f"{np.round(np.mean(game), 4)},", end="")


    print()
    for game in games.values():
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)},", end="")"""