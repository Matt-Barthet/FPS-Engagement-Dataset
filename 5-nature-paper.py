import pandas as pd
import numpy as np

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
                    changes[game] = [np.max(participant_raw["Value"].values) - np.min(participant_raw["Value"].values)]
                else:
                    changes[game].append(np.max(participant_raw["Value"].values) - np.min(participant_raw["Value"].values))

    for game in changes.keys():
        print(f"{game},", end="")

    print()
    for game in changes.values():
        print(f"{np.round(np.mean(game), 2)},", end="")


    print()
    for game in changes.values():
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 2)},", end="")

    exit()
    games = {}
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
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)},", end="")


    participant = {}
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
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)},", end="")