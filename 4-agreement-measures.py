import numpy as np
import pandas as pd
from utility_functions import *
from scipy.stats import pearsonr
from plotting import *
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations

def compute_correlations(sda_scores, label):
    correlations = {}
    for session_id, session_data in sda_scores.items():
        group_data = session_data['Expert']
        game_names = set(game for participant_data in group_data.values() for game in participant_data['Engagement'].keys())
        for game in game_names:
            game_sdas = [participant_data['Engagement'].get(game) for participant_data in group_data.values()
                        if game in participant_data['Engagement']]
            print(game_sdas)
            label_sdas = [participant_data[label] for participant_data in group_data.values()
                        if game in participant_data['Engagement']]
            print(len(game_sdas))
            """for i in range(len(game_sdas)):
                if np.isnan(game_sdas[i]) or np.isnan(label_sdas[i]):
                    del game_sdas[i]
                    del label_sdas[i]"""
            game_sdas = np.nan_to_num(game_sdas, True, 0)
            label_sdas = np.nan_to_num(label_sdas, True, 0)
            correlation, _ = pearsonr(game_sdas, label_sdas)
            correlations[f"{session_id}-{game}"] = correlation
    return correlations


def agreement_with_gold_standard(engagement_data, gold_standard, distance_function):
    agreement_dict = {}
    for session_id, participants in engagement_data.items(): 
        for group_name, group_data in participants.items():
            for participant_id, participant_data in group_data.items():
                agreement_dict[f"{session_id}"][group_name][participant_id]['Engagement'] = {}
                lengths = []
                ranges = []
                for game_id, game_data in participant_data.items():
                    if game_id in gold_standard[session_id][group_name][participant_id]:
                        median_trace = gold_standard[session_id][group_name][participant_id][game_id]
                        try:
                            agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = distance_function(game_data, median_trace)
                            lengths.append(count_changes(game_data))
                            ranges.append(np.max(game_data) - np.min(game_data))
                        except TypeError:
                            agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = 0
                            lengths.append(0)
                            ranges.append(0)

                agreement_dict[f"{session_id}"][group_name][participant_id]['Lengths'] = np.ceil(np.mean(lengths))
                agreement_dict[f"{session_id}"][group_name][participant_id]['Ranges'] = np.ceil(np.mean(ranges))
    return agreement_dict


def build_agreement_dict(visual_data, audio_data, engagement_data, distance_function, gold_standard=None, tw=1000):
    pitch_gt_trace = pd.read_csv('./Processed Data/QA_Audio_GT.csv')['Value'].to_numpy()
    pitch_gt_trace = (np.interp(pitch_gt_trace, (pitch_gt_trace.min(), pitch_gt_trace.max()), (0, 1)))[::int(15 * (tw / 250))]

    green_gt_trace = pd.read_csv('./Processed Data/QA_Visual_GT.csv')['Value'].to_numpy()
    green_gt_trace = (np.interp(green_gt_trace, (green_gt_trace.min(), green_gt_trace.max()), (0, 1)))[::int(15 * (tw / 250))]

    agreement_dict = {}
    for session_id, session_data in visual_data.items():
        agreement_dict[f"{session_id}"] = {"Expert": {}, "Mturk": {}}
        for group_name, group_data in session_data.items():
            participants = list(group_data.keys())
            for participant_id in participants:
                agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]= {}
                try: 
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Visual_SDA"] = np.round(distance_function(list(visual_data[session_id][group_name][participant_id].values())[0], green_gt_trace), 4)   
                except TypeError:
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Visual_SDA"] = 0
                try:
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Audio_SDA"] = np.round(distance_function(list(audio_data[session_id][group_name][participant_id].values())[0], pitch_gt_trace), 4)
                except TypeError:
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Audio_SDA"] = 0
    if gold_standard is None:
        agreement_dict = pairwise_agreement(agreement_dict, engagement_data, distance_function)
    else:
        agreement_dict = agreement_with_gold_standard(engagement_data, gold_standard, distance_function)
    return agreement_dict


def pairwise_agreement(agreement_dict, engagement_data, distance_function):
    for session_id, participants in engagement_data.items(): 
        for group_name, group_data in participants.items():
            for participant_id, participant_data in group_data.items():
                agreement_dict[f"{session_id}"][group_name][participant_id]['Engagement'] = {}
                lengths = []
                ranges = []
                for game_id, game_data in participant_data.items():
                    dtws = []
                    try:
                        lengths.append(count_changes(game_data))
                        ranges.append(np.max(game_data) - np.min(game_data))
                    except TypeError:
                        lengths.append(0)
                        ranges.append(0)

                    for other_id in group_data.keys():
                        try:
                            dtws.append(distance_function(game_data, engagement_data[session_id][group_name][other_id][game_id]))
                        except ValueError:
                            dtws.append(np.nan)
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = np.mean(dtws)
                agreement_dict[f"{session_id}"][group_name][participant_id]['Lengths'] = np.ceil(np.mean(lengths))
                agreement_dict[f"{session_id}"][group_name][participant_id]['Ranges'] = np.ceil(np.mean(ranges))
    return agreement_dict


def create_distance_matrix(engagement_data, distance_function, groups = ['Expert']):
    distance_matrices = {}
    for session_id, participants in engagement_data.items():
        distance_matrices[session_id] = {}
        for group_name, group_data in participants.items():
            if group_name not in groups:
                continue
            distance_matrices[session_id][group_name] = {}
            if len(group_data) < 1: # Ignore small sessions
                continue
            participant_ids = list(group_data.keys())
            for game_id in group_data[participant_ids[0]].keys():
                distance_matrices[session_id][group_name][game_id] = {}
                distance_matrix = np.zeros((len(participant_ids), len(participant_ids)))
                for (i, pid), (j, other_pid) in combinations(enumerate(participant_ids), 2):
                    try:
                        distance = distance_function(group_data[pid][game_id], group_data[other_pid][game_id])
                        distance_matrix[i][j] = distance
                    except ValueError:
                        distance_matrix[i][j] = np.nan
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                distance_matrices[session_id][group_name][game_id] = distance_matrix
    return distance_matrices


def create_distance_dict(engagement_data, distance_function, groups=['Expert']):
    distance_dict = {}
    for session_id, participants in engagement_data.items():
        distance_dict[session_id] = {}
        for group_name, group_data in participants.items():
            if group_name not in groups or len(group_data) < 1:
                continue
            distance_dict[session_id][group_name] = {}
            participant_ids = list(group_data.keys())
            for game_id in group_data[participant_ids[0]].keys():
                for participant_id in participant_ids:
                    other_distances = []
                    for other_participant_id in participant_ids:
                        if participant_id != other_participant_id:
                            other_distances.append(distance_function(group_data[participant_id][game_id], group_data[other_participant_id][game_id]))
                    if group_name not in distance_dict[session_id]:
                        distance_dict[session_id][group_name] = {}
                    if participant_id not in distance_dict[session_id][group_name]:
                        distance_dict[session_id][group_name][participant_id] = {}
                    distance_dict[session_id][group_name][participant_id][game_id] = np.min(other_distances)
    return distance_dict


def ignore_unwanted_sessions(visual_data, audio_data, engagement_data, gold_standard_data, DESIRED_SESSIONS):
    for session_id, session_data in visual_data.items():
        for group_name, group_data in session_data.items():
            participants = list(group_data.keys())
            for participant_id in participants:
                if len(DESIRED_SESSIONS) != 0 and session_id not in DESIRED_SESSIONS:
                    del visual_data[session_id][group_name][participant_id]
                    del audio_data[session_id][group_name][participant_id]
                    del engagement_data[session_id][group_name][participant_id]
                    # del gold_standard_data[session_id][group_name][participant_id]
    return visual_data, audio_data, engagement_data, # gold_standard_data


def calculate_distance_thresholds(session_matrices):
    min_distances = []
    for _, games in session_matrices.items():
        for _, matrix in games.items():
            for i in range(len(matrix)):
                distances = matrix[i, :]
                non_zero_distances = distances[np.nonzero(distances)]
                if non_zero_distances.size > 0:
                    min_distances.append(np.min(non_zero_distances))
    mean_val = np.nanmean(min_distances)
    std_dev = np.nanstd(min_distances)
    threshold_1std = mean_val + std_dev
    threshold_2std = mean_val + 2 * std_dev
    return threshold_1std, threshold_2std

from copy import deepcopy

def output_filtered_data(engagement_data, distance_dict, distance_matrices):

    std1_filtered_data = deepcopy(engagement_data)
    std2_filtered_data = deepcopy(engagement_data)

    for session_id, session_data in distance_dict.items():
        std1, std2 = calculate_distance_thresholds(distance_matrices[session_id])
        std1 *= 2
        std2 *= 2
        for group_id, group_data in session_data.items():
            std1counter, std2counter = 0, 0

            distances = []

            for participant_id, participant_data in group_data.items():
                for game_name, _ in participant_data.items():
                    participant_distance = distance_dict[session_id][group_id][participant_id][game_name]
                    distances.append(participant_distance)
                    if participant_distance > std2:
                        std2counter += 1
                        del std2_filtered_data[session_id][group_id][participant_id][game_name]
                    if participant_distance > std1:
                        std1counter += 1
                        del std1_filtered_data[session_id][group_id][participant_id][game_name]

            print(f"{session_id}, Group {group_id}, removed {std2counter} outliers using the 2STD threshold")
            print(f"{session_id}, Group {group_id}, removed {std1counter} outliers using the 1STD threshold")

    np.save("Session_Dict(Engagement)_Filtered_1STD.npy", std1_filtered_data)
    np.save("Session_Dict(Engagement)_Filtered_2STD.npy", std2_filtered_data)
    return std1_filtered_data, std2_filtered_data


def execute(time_windows, DESIRED_SESSIONS, distance_function):

    plt.rcParams['font.size'] = 10

    engagement_data = np.load("./Processed Data/Session_Dict(Engagement_Task).npy", allow_pickle=True).item()
    visual_data = np.load("./Processed Data/Session_Dict(Visual_Task).npy", allow_pickle=True).item()
    audio_data = np.load("./Processed Data/Session_Dict(Audio_Task).npy", allow_pickle=True).item()
    gold_standard_data = np.load("./Processed Data/Engagement_Gold_Standard(Median).npy", allow_pickle=True).item()
    
    visual_data, audio_data, engagement_data = ignore_unwanted_sessions(visual_data, audio_data, engagement_data, gold_standard_data, DESIRED_SESSIONS)
    agreement_dict = build_agreement_dict(visual_data, audio_data, engagement_data, distance_function, tw=time_windows)

    """games = {}
    for session_id, session_data in agreement_dict.items():
        group_data = session_data["Expert"].items()
        for participant_id, participant_data in group_data:
            type_data = participant_data['Engagement']
            for game in type_data.keys():
                if participant_id not in games.keys():
                    games[participant_id] = [type_data[game]]
                else:
                    games[participant_id].append(type_data[game])

    for game in games.keys():
        print(f"{game}, {len(games[game])}")

    print()
    for game in games.values():
        print(f"{np.round(np.mean(game), 4)}")


    print()
    for game in games.values():
        print(f"{np.round(1.96*np.std(game)/np.sqrt(len(game)), 4)}")"""

    # plot_engagement_data(engagement_data, None)
    all_distance_matrices = create_distance_matrix(engagement_data, distance_function)
    min_distance_dict = create_distance_dict(engagement_data, distance_function)
    # plot_minimum_distance_histogram(all_distance_matrices)
    # plot_highlighted_engagement_data(engagement_data, all_distance_matrices)
    num_participants = 5  # Number of participants
    cmap = plt.get_cmap('viridis', 5)  # Getting the Viridis colormap for the number of participants
    colors = [cmap(i) for i in range(num_participants)]  # Generating colors for each participant
    markers = ['o', 'v', 's', '^', 'x']
    plot_counter = 0
    
    participant_mapping = {}    
    counter = 0
    for paricipant in engagement_data['Session-1']['Expert'].keys():
        participant_mapping[paricipant] = counter
        counter+=1

    std1_filtered, std2_filtered = output_filtered_data(engagement_data, min_distance_dict, all_distance_matrices)
    for session_id, session_data in std1_filtered.items():
        plot_counter = 0
        for group_id, group_data in session_data.items():
            for participant_id, participant_data in group_data.items():
                for game_name, game_data in participant_data.items():
                    if session_id == "Session-1" and group_id == "Expert" and "void" in game_name:
                        plt.errorbar(x=range(len(std2_filtered[session_id][group_id][participant_id][game_name])), y=std2_filtered[session_id][group_id][participant_id][game_name], color=colors[participant_mapping[participant_id]], marker=markers[participant_mapping[participant_id]], markevery=4, markeredgecolor='black', linewidth=2)
                        plot_counter+=1

        if plot_counter > 0:
            plt.ylabel("Value")
            plt.xlabel("Time (s)")
            plt.show()
    
    exit()
    # plot_filtered_traces(engagement_data, std1_filtered, std2_filtered)
    # plot_engagement_data(std1_filtered, None)
    # plot_distance_matrices(all_distance_matrices)

    visual_sdas, audio_sdas, engagement_sdas = [], [], []
    for _, session_data in agreement_dict.items():
        group_name='Expert'
        lengths = []
        ranges = []
        session_sdas = []
        for _, participant_data in session_data[group_name].items():
            sda_list = list(participant_data['Engagement'].values())
            mean_sda, _ = compute_confidence_interval(sda_list)
            visual_sdas.append(np.round(participant_data['Visual_SDA'], 4))
            audio_sdas.append(np.round(participant_data['Audio_SDA'], 4))
            engagement_sdas.append(mean_sda)
            session_sdas.append(mean_sda)
            lengths.append(participant_data['Lengths'])
            ranges.append(participant_data['Ranges'])
                
    # correlation_visual = pearsonr(visual_sdas, engagement_sdas)
    # correlation_audio = pearsonr(audio_sdas, engagement_sdas)
    visual_correlations = compute_correlations(agreement_dict, "Visual_SDA")
    audio_correlations = compute_correlations(agreement_dict, "Audio_SDA")
    
    # game_dtw_scatter(agreement_dict)
    # plot_correlations(audio_correlations, visual_correlations, True, "")
    plot_correlations_group(audio_correlations, visual_correlations, True, "")
    # plot_sda_scatter_grouped(agreement_dict)

sessions = ['Session-1', 'Session-2', 'Session-3', 'Session-7'] # + [f'Session-{i}' for i in range(8, 16)]
execute(1000, sessions, dtw_distance)