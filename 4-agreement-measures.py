import numpy as np
import pandas as pd
import os
from utility_functions import *
from scipy.stats import pearsonr
from plotting import plot_distance_matrices, plot_engagement_data, plot_distance_histogram, plot_minimum_distance_histogram, plot_highlighted_engagement_data
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations

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


def create_distance_matrix(engagement_data, distance_function, groups = ['Expert']):
    distance_matrices = {}
    for session_id, participants in engagement_data.items():
        distance_matrices[session_id] = {}
        for group_name, group_data in participants.items():
            if group_name not in groups:
                continue
            distance_matrices[session_id][group_name] = {}
            if len(group_data) == 0: # Ignore empty sessions
                continue
            participant_ids = list(group_data.keys())
            for game_id in group_data[participant_ids[0]].keys():
                distance_matrices[session_id][group_name][game_id] = {}
                distance_matrix = np.zeros((len(participant_ids), len(participant_ids)))
                for (i, pid), (j, other_pid) in combinations(enumerate(participant_ids), 2):
                    distance = distance_function(group_data[pid][game_id], group_data[other_pid][game_id])
                    distance_matrix[i][j] = distance
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                distance_matrices[session_id][group_name][game_id] = distance_matrix
    return distance_matrices


def execute(time_windows, DESIRED_SESSIONS, distance_function):

    pitch_gt_trace = pd.read_csv('./Processed Data/QA_Audio_GT.csv')['Value'].to_numpy()
    pitch_gt_trace = (np.interp(pitch_gt_trace, (pitch_gt_trace.min(), pitch_gt_trace.max()), (0, 1)))[::int(15 * (time_windows / 250))]

    green_gt_trace = pd.read_csv('./Processed Data/QA_Visual_GT.csv')['Value'].to_numpy()
    green_gt_trace = (np.interp(green_gt_trace, (green_gt_trace.min(), green_gt_trace.max()), (0, 1)))[::int(15 * (time_windows / 250))]

    engagement_data = np.load("./Processed Data/Session_Dict(Engagement_Task).npy", allow_pickle=True).item()
    visual_data = np.load("./Processed Data/Session_Dict(Visual_Task).npy", allow_pickle=True).item()
    audio_data = np.load("./Processed Data/Session_Dict(Audio_Task).npy", allow_pickle=True).item()
    median_data = np.load("./Processed Data/Engagement_Gold_Standard(Median).npy", allow_pickle=True).item()

    agreement_dict = {}

    for session_id, session_data in visual_data.items():
        agreement_dict[f"{session_id}"] = {"Expert": {}, "Mturk": {}}
        for group_name, group_data in session_data.items():
            participants = list(group_data.keys())
            for participant_id in participants:
                if len(DESIRED_SESSIONS) != 0 and session_id not in DESIRED_SESSIONS:
                    del visual_data[session_id][group_name][participant_id]
                    del audio_data[session_id][group_name][participant_id]
                    del engagement_data[session_id][group_name][participant_id]
                else:
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]= {}
                    try: 
                        agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Visual_SDA"] = np.round(distance_function(list(visual_data[session_id][group_name][participant_id].values())[0], green_gt_trace), 4)   
                    except TypeError:
                        agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Visual_SDA"] = 0
                    try:
                        agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Audio_SDA"] = np.round(distance_function(list(audio_data[session_id][group_name][participant_id].values())[0], pitch_gt_trace), 4)
                    except TypeError:
                        agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Audio_SDA"] = 0

    # Example for one distance matrix

    # plot_engagement_data(engagement_data, None)
    all_distance_matrices = create_distance_matrix(engagement_data, distance_function)
    plot_minimum_distance_histogram(all_distance_matrices)
    plot_highlighted_engagement_data(engagement_data, all_distance_matrices)

    # plot_distance_matrices(all_distance_matrices)

    """for session_id, participants in engagement_data.items(): 
        for group_name, group_data in participants.items():
            for participant_id, participant_data in group_data.items():
                for game_id, game_data in participant_data.items():

                    # Retrieve the distance matrix for the current game
                    distance_matrix = all_distance_matrices[session_id][group_name][game_id]

                    # Perform hierarchical clustering
                    linked = linkage(squareform(distance_matrix), 'single')

                    # Determine the clusters
                    max_d = 6  # Example threshold value, adjust based on your data
                    clusters = fcluster(linked, max_d, criterion='distance')

                    # Plotting the dendrogram with clusters
                    plt.figure(figsize=(10, 7))
                    dendrogram(linked, 
                            orientation='top', 
                            distance_sort='descending', 
                            show_leaf_counts=True,
                            color_threshold=0)  # Color threshold for cluster visualization
                    plt.title(f'{session_id}-{game_id} Dendrogram')
                    plt.show()"""

    for session_id, participants in engagement_data.items(): 
        for group_name, group_data in participants.items():
            for participant_id, participant_data in group_data.items():
                agreement_dict[f"{session_id}"][group_name][participant_id]['Engagement'] = {}
                lengths = []
                ranges = []
                for game_id, game_data in participant_data.items():
                    dtws = []
                    lengths.append(count_changes(game_data))
                    ranges.append(np.max(game_data) - np.min(game_data))
                    for other_id in group_data.keys():
                        dtws.append(distance_function(game_data, engagement_data[session_id][group_name][other_id][game_id]))
                    agreement_dict[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = np.mean(dtws)
                agreement_dict[f"{session_id}"][group_name][participant_id]['Lengths'] = np.ceil(np.mean(lengths))
                agreement_dict[f"{session_id}"][group_name][participant_id]['Ranges'] = np.ceil(np.mean(ranges))

    """
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
                            sdas[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = distance_function(game_data, median_trace)
                            lengths.append(count_changes(game_data))
                            ranges.append(np.max(game_data) - np.min(game_data))
                        except TypeError:
                            sdas[f"{session_id}"][group_name][f"{participant_id}"]["Engagement"][game_id] = 0
                            lengths.append(0)
                            ranges.append(0)

                sdas[f"{session_id}"][group_name][participant_id]['Lengths'] = np.ceil(np.mean(lengths))
                sdas[f"{session_id}"][group_name][participant_id]['Ranges'] = np.ceil(np.mean(ranges))
    """

    visual_sdas, audio_sdas, engagement_sdas = [], [], []
    for session_id, session_data in agreement_dict.items():
        for group_name, group_data in session_data.items():
            lengths = []
            ranges = []
            session_sdas = []
            for participant_id, participant_data in group_data.items():
                sda_list = list(participant_data['Engagement'].values())
                mean_sda, _ = compute_confidence_interval(sda_list)
                visual_sdas.append(np.round(participant_data['Visual_SDA'], 4))
                audio_sdas.append(np.round(participant_data['Audio_SDA'], 4))
                engagement_sdas.append(mean_sda)
                session_sdas.append(mean_sda)
                lengths.append(participant_data['Lengths'])
                ranges.append(participant_data['Ranges'])

    deleted = 0
    for i in range(len(visual_sdas)):
        if visual_sdas[i - deleted] == 0 or audio_sdas[i - deleted] == 0:
            del visual_sdas[i - deleted]
            del audio_sdas[i - deleted]
            del engagement_sdas[i - deleted]
            deleted += 1
                
    correlation_visual = pearsonr(visual_sdas, engagement_sdas)
    correlation_audio = pearsonr(audio_sdas, engagement_sdas)
    visual_correlations = compute_correlations(agreement_dict, "Visual_SDA")
    audio_correlations = compute_correlations(agreement_dict, "Audio_SDA")
    
    # game_dtw_scatter(agreement_dict)
    # plot_correlations(audio_correlations, visual_correlations, True, "")
    # plot_sda_scatter_grouped(agreement_dict)

execute(3000, ['Session-1', 'Session-2', 'Session-3', 'Session-7'], dtw_distance)