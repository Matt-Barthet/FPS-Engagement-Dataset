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


def extract_raw_frequency(game_names, raw_data):
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

    # per_participant_analysis(engagement_data, raw_data, game_names)

    games = {}
    for session_id, session_data in engagement_data.items():
        counter = 0
        group_data = session_data["Expert"].items()
        for participant_id, participant_data in group_data:

            for game in participant_data.keys():
                if game not in games.keys():

                    trace = participant_data[game][:-1]
                    ordinal_trace = np.array([trace[item] - trace[item-1] for item in range(1, len(trace))])

                    zero_values = ordinal_trace[ordinal_trace == 0].tolist()
                    negative_values = ordinal_trace[ordinal_trace < 0].tolist()
                    positive_values = ordinal_trace[ordinal_trace > 0].tolist()

                    games[game] = {"Increase": [len(positive_values)], 
                                   "Decrease": [len(negative_values)], 
                                   "Stable": [len(zero_values)]}
                else:
                    trace = participant_data[game][:-1]
                    ordinal_trace = np.array([trace[item] - trace[item-1] for item in range(1, len(trace))])

                    zero_values = ordinal_trace[ordinal_trace == 0].tolist()
                    negative_values = ordinal_trace[ordinal_trace < 0].tolist()
                    positive_values = ordinal_trace[ordinal_trace > 0].tolist()
                    games[game]['Increase'].append(len(positive_values))
                    games[game]['Decrease'].append(len(negative_values))
                    games[game]['Stable'].append(len(zero_values))


    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming games is already populated
    # Number of games
    n_games = len(games.keys())
    # Setup for subplot grid
    cols = 5  # Number of columns in the subplot grid
    rows = np.ceil(n_games / cols)  # Calculate rows needed based on the number of games

    fig, ax = plt.subplots(int(rows), int(cols), constrained_layout=True)  # Adjust the figure size as needed
    axs = ax.flatten()
    # Iterate over each game to create a subplot
    for i in range(len(game_names)):
        game_name = game_names[i]
        data = games[game_name]
        # Data for the histogram
        labels = ['Decrease', 'Stable', 'Increase', ]
        counts = [np.mean(data['Decrease']), np.mean(data['Stable']), np.mean(data['Increase'])]

        axs[i].bar(labels, counts)  # Create a bar chart
        axs[i].set_title(game_name)  # Set the title of the subplot to the game name
        axs[i].set_ylim([0, 45])
        if i % cols == 0:
            axs[i].set_ylabel('Count')  # Set the y-axis label

        if i < 25:
            axs[i].set_xticks([0, 1, 2], ['','',''])

    plt.show()  # Display the plot
    
    for game in sorted(games.keys()):
        print(f"{game},", end="")
    print()
    
    for game in games.values():
        print(f"{np.round(np.mean(game['Stable']), 4)}")

    print()
    for game in games.values():
        print(f"{np.round(1.96*np.std(game['Stable'])/np.sqrt(len(game['Stable'])), 4)}")


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