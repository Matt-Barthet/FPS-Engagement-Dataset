import numpy as np
import pandas as pd
import os
import matplotlib
from plotting import *
from utility_functions import *
from scipy.stats import pearsonr

PARTICIPANTS_PER_SESSION = -1 # Set to -1 if you don't want to filter based on this variable
DESIRED_SESSIONS = ['Session-1', 'Session-2', 'Session-3', 'Session-7']

def compute_session_sda(participant_sda_dict, session_data, function, keep_best_instead_mean=False):
    for participant_id, participant_data in session_data.items():
        participant_sda_dict[f"{session_id}"][f"{participant_id}"]["Engagement"] = {}
        for game in participant_data.keys():
            game_sda_list = []
            for other_id in session_data.keys():
                if participant_id != other_id and game in session_data[other_id]:
                    game_sda = function(participant_data[game], session_data[other_id][game])
                    game_sda_list.append(game_sda)
            if game_sda_list:
                participant_sda_dict[f"{session_id}"][f"{participant_id}"]["Engagement"][game] = np.max(game_sda_list)
    return participant_sda_dict

def participant_sda_gt_signal(participant_data, function, gt_signal):
    game_sda_dict = {}
    game_names = set(game for game in participant_data.keys())
    for game in game_names:
        if game in participant_data:
            game_sda = function(participant_data[game], gt_signal)
            game_sda_dict[game] = game_sda
    return game_sda_dict

def compute_correlations(sda_scores, label):
    correlations = {}
    for session_id, session_data in sda_scores.items():
        for _, group_data in session_data.items():
            game_names = set(game for participant_data in group_data.values() for game in participant_data['Engagement'].keys())
            for game in game_names:
                game_sdas = [participant_data['Engagement'].get(game) for participant_data in group_data.values()
                            if game in participant_data['Engagement']]
                label_sdas = [participant_data[label] for participant_data in group_data.values()
                            if game in participant_data['Engagement']]
                for i in range(len(game_sdas)):
                    if label_sdas == np.nan:
                        del game_sdas[i]
                        del label_sdas[i]
                correlation, _ = pearsonr(game_sdas, label_sdas)
                correlations[f"{session_id}-{game}"] = correlation
    return correlations

def count_changes(arr):
    arr = np.asarray(arr)
    if len(arr) < 2:
        return 0
    
    change_count = 0
    for i in range(len(arr) - 1):
        if arr[i] != arr[i + 1]:
            change_count += 1
    return change_count

if __name__ == "__main__":

    font = {'size': 14}

    matplotlib.rc('font', **font)

    # tasks we are interested to evaluate in this work
    pagan_task_names = ['Brightness', 'Pitch', 'Engagement']

    # all session files and pagan annotations are stored on shared google drive
    pagan_data_link = './'

    time_windows = 250

    # ground truth data for pitch task
    pitch_gt_trace = pd.read_csv(os.path.join(pagan_data_link, f'{pagan_task_names[1]}_values.csv'))['Value'].to_numpy()
    pitch_gt_trace = (np.interp(pitch_gt_trace, (pitch_gt_trace.min(), pitch_gt_trace.max()), (0, 1)))[::int(15 * (time_windows / 250))]

    green_gt_trace = pd.read_csv(os.path.join(pagan_data_link, f'{pagan_task_names[0]}_values.csv'))['Value'].to_numpy()
    green_gt_trace = (np.interp(green_gt_trace, (green_gt_trace.min(), green_gt_trace.max()), (0, 1)))[::int(15 * (time_windows / 250))]

    engagement_data = np.load("./Engagement_Task.npy", allow_pickle=True).item()
    visual_data = np.load("./Visual_Task.npy", allow_pickle=True).item()
    audio_data = np.load("./Audio_Task.npy", allow_pickle=True).item()
    median_data = np.load("./Engagement_Median.npy", allow_pickle=True).item()

    #plot_brightness_and_pitch_data(visual_data, ["Session-1", "Session-2", "Session-3", "Session-7"], "Expert", "", green_gt_trace)
    #plot_brightness_and_pitch_data(visual_data, ["Session-1", "Session-2", "Session-3"], "Mturk", "", green_gt_trace)

    #plot_brightness_and_pitch_data(audio_data, ["Session-1", "Session-2", "Session-3", "Session-7"], "Expert", "", pitch_gt_trace, True)
    #plot_brightness_and_pitch_data(audio_data, ["Session-1", "Session-2", "Session-3"], "Mturk", "", pitch_gt_trace, True)

    sdas = {}

    for session_id, session_data in visual_data.items():
        sdas[f"{session_id}"] = {"Expert": {}, "Mturk": {}}
        for group_name, group_data in session_data.items():
            participants = list(group_data.keys())
            for participant_id in participants:
                if (len(participants) < PARTICIPANTS_PER_SESSION and PARTICIPANTS_PER_SESSION != -1) or (len(DESIRED_SESSIONS) != 0 and session_id not in DESIRED_SESSIONS):
                    try:
                        del visual_data[session_id][group_name][participant_id]
                        del audio_data[session_id][group_name][participant_id]
                        del engagement_data[session_id][group_name][participant_id]
                    except KeyError:
                        print(session_id, group_name, participant_id)
                else:
                    sdas[f"{session_id}"][group_name][f"{participant_id}"]= {}
                    try: 
                        sdas[f"{session_id}"][group_name][f"{participant_id}"]["Visual_SDA"] = np.round(sda(list(visual_data[session_id][group_name][participant_id].values())[0], green_gt_trace), 4)   
                    except TypeError:
                        sdas[f"{session_id}"][group_name][f"{participant_id}"]["Visual_SDA"] = 0
                    try:
                        sdas[f"{session_id}"][group_name][f"{participant_id}"]["Audio_SDA"] = np.round(sda(list(audio_data[session_id][group_name][participant_id].values())[0], pitch_gt_trace), 4)
                    except TypeError:
                        sdas[f"{session_id}"][group_name][f"{participant_id}"]["Audio_SDA"] = 0

    for session_id, participants in engagement_data.items(): 
        for group_name, group_data in participants.items():
            for participant_id, participant_data in group_data.items():
                sdas[f"{session_id}"][group_name][participant_id]['Engagement'] = {}
                lengths = []
                ranges = []
                for game_id, game_data in participant_data.items():
                    if game_id in median_data[session_id][group_name][participant_id]:
                        median_trace = median_data[session_id][group_name][participant_id][game_id]
                        try:
                            sdas[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = sda(game_data, median_trace)
                            lengths.append(count_changes(game_data))
                            ranges.append(np.max(game_data) - np.min(game_data))
                        except TypeError:
                            sdas[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = 0
                            lengths.append(0)
                            ranges.append(0)

                print(lengths)
                sdas[f"{session_id}"][group_name][participant_id]['Lengths'] = np.ceil(np.mean(lengths))
                sdas[f"{session_id}"][group_name][participant_id]['Ranges'] = np.ceil(np.mean(ranges))


    visual_sdas, audio_sdas, engagement_sdas = [], [], []
    lengths = []
    ranges = []
    
    for session_id, session_data in sdas.items():
        for group_name, group_data in session_data.items():
            print(f"\nSESSION:{session_id}")
            for participant_id, participant_data in group_data.items():
                sda_list = list(participant_data['Engagement'].values())
                mean_sda, ci_sda = compute_confidence_interval(sda_list)
                print(f"{participant_id}: {np.round(participant_data['Visual_SDA'], 4)},{np.round(participant_data['Audio_SDA'], 4)},{mean_sda}±{ci_sda}")
                visual_sdas.append(np.round(participant_data['Visual_SDA'], 4))
                audio_sdas.append(np.round(participant_data['Audio_SDA'], 4))
                engagement_sdas.append(mean_sda)
                lengths.append(participant_data['Lengths'])
                ranges.append(participant_data['Ranges'])

    plt.figure()

    # Creating the scatter plot
    plt.scatter(lengths, engagement_sdas)

    # Adding titles and labels
    plt.xlabel('Number of Changes in Trace')
    plt.ylabel('SDA')


    plt.figure()

    # Creating the scatter plot
    plt.scatter(ranges, engagement_sdas)

    # Adding titles and labels
    plt.xlabel('Range of the Trace (Maximum vs Minimum)')
    plt.ylabel('SDA')


    # Displaying the plot
    plt.show()

    deleted = 0

    for i in range(len(visual_sdas)):
        if visual_sdas[i - deleted] == 0 or audio_sdas[i - deleted] == 0:
            del visual_sdas[i - deleted]
            del audio_sdas[i - deleted]
            del engagement_sdas[i - deleted]
            deleted += 1
                

    correlation_visual = pearsonr(visual_sdas, engagement_sdas)
    correlation_audio = pearsonr(audio_sdas, engagement_sdas)

    visual_correlations = compute_correlations(sdas, "Visual_SDA")
    audio_correlations = compute_correlations(sdas, "Audio_SDA")
    
    plot_correlations(audio_correlations, visual_correlations, True, "")
    plot_sda_scatter_grouped(sdas)
