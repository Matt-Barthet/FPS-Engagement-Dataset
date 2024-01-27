import matplotlib.pyplot as plt
import numpy as np
from utility_functions import *
import matplotlib
from matplotlib.colors import ListedColormap
import seaborn as sns
import math
time_windows = 1000

from matplotlib.cm import coolwarm
from matplotlib.colors import LinearSegmentedColormap

def plot_colored_engagement_data(data_dict, distance_matrices, groups=['Expert']):
    all_game_names = sorted(set(game_name for session_data in data_dict.values() for group_data in session_data.values() for participant_data in group_data.values()
                                for game_name in participant_data.keys()))
    game_to_index = {game_name: index for index, game_name in enumerate(all_game_names)}

    for session_id, session_data in distance_matrices.items():
        print(session_id)
        min_distances = []
        for _, games in session_data.items():
            for game_id, matrix in games.items():
                for participant_id in range(len(matrix)):
                    distances = matrix[participant_id, :]
                    non_zero_distances = distances[np.nonzero(distances)]
                    if non_zero_distances.size > 0:
                        min_distances.append(np.min(non_zero_distances))

        # Create a dictionary to store normalized distances for this session
        normalized_distances_dict = {}
        distances_dict = {}

        normalized_list = []
        for _, games in session_data.items():
            for game_id, matrix in games.items():
                for participant_id in range(len(matrix)):
                    distances = matrix[participant_id, :]
                    non_zero_distances = distances[np.nonzero(distances)]
                    if non_zero_distances.size > 0:
                        min_distance_of_participant = np.min(non_zero_distances)
                        normalized_distance = (min_distance_of_participant - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
                        normalized_distances_dict[(game_id, participant_id)] = normalized_distance
                        distances_dict[(game_id, participant_id)] = min_distance_of_participant
                        normalized_list.append(normalized_distance)

        mean_val = np.nanmean(normalized_list)
        std_dev = np.nanstd(normalized_list)
        distance_threshold = mean_val + 2 * std_dev

        # Create a custom segmented colormap
        cdict = {'red':   [(0.0, 0.0, 0.0),
                        (distance_threshold, 1.0, 1.0),
                        (1.0, 1.0, 1.0)],

                'green': [(0.0, 0.0, 0.0),
                        (distance_threshold, 0.0, 0.0),
                        (1.0, 0.0, 0.0)],

                'blue':  [(0.0, 1.0, 1.0),
                        (distance_threshold, 0.0, 0.0),
                        (1.0, 0.0, 0.0)]}

        custom_colormap = LinearSegmentedColormap('CustomMap', cdict)

        for group_id, group_data in data_dict[session_id].items():
            if group_id not in groups:
                continue
            
            fig, axes = plt.subplots(5, 6, figsize=(14, 12), constrained_layout=True)
            fig.suptitle(f"Highlighted Engagement Data for {session_id}", fontsize=16)
            axes = axes.flatten()
            for ids, (participant_id, participant_data) in enumerate(group_data.items()):
                for game_name, game_values in participant_data.items():
                    try:
                        ax = axes[game_to_index[game_name]]
                        time_values = np.arange(0, len(game_values))
                        # Ensure that the color is correctly selected from the colormap
                        normalized_distance = normalized_distances_dict.get((game_name, ids), 0)
                        color = custom_colormap(normalized_distance)
                        ax.plot(time_values, game_values, label=participant_id, color=color)
                    except TypeError:
                        pass
                    except ValueError:
                        pass

            sm = plt.cm.ScalarMappable(cmap=custom_colormap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
            cbar.set_label('Normalized DTW Distance')
            cbar.ax.axvline(distance_threshold, color='black', linewidth=4, label='Threshold')
            cbar.ax.legend(bbox_to_anchor=(1.2,0.5), loc='center right')

            try:
                plt.savefig(f"./Data Analysis/Figures/{session_id}-{group_id}_highlighted.png")
            except ValueError:
                pass
            plt.close()

def plot_brightness_and_pitch_data(session, session_ids, group, title, gt_signal, sound=False,start=0):

    labels = [f'P{i+1+start}' for i in range(20)]
    counter = 0
    for session_id, session_data in session.items():
        if session_id in session_ids:

            plt.figure(figsize=(8, 4))

            if not sound:
                plt.plot(gt_signal[:int(61 * (1000 / time_windows))], color ="black", lw=2, marker='o', markevery=4, label="GT")
                # plt.xticks([0, 60, 120, 180, 240], [0, 15, 30, 45, 60])
                plt.xlim(xmin=-1, xmax=int(61 * (1000 / time_windows)))
                for _, participant_data in session_data[group].items():
                    plt.plot(list(participant_data.values())[0][:int(61 * (1000 / time_windows))], alpha=0.75, label=labels[counter])
                    counter+=1
            else:
                # plt.xticks([0, 40, 80, 120], [0, 10, 20, 30])
                plt.plot(gt_signal[:int(31 * (1000 / time_windows))], color ="black", lw=2, marker='o', markevery=4, label='GT')
                plt.xlim(xmin=-1, xmax=int(31 * (1000 / time_windows)))
                for _, participant_data in session_data[group].items():
                    plt.plot(list(participant_data.values())[0][:int(31 * (1000 / time_windows))], alpha=0.75, label=labels[counter])
                    counter+=1

            if sound:
                title = "Audio"
            else:
                title = "Visual"
            plt.title(f'{session_id}: {title} QA ({group})')
            plt.xlabel(f'Time ({time_windows} ms TW)')
            plt.ylabel('Value')
            plt.tight_layout()
            plt.legend(bbox_to_anchor=(1.2,1.03), loc='upper right')
            plt.subplots_adjust(right=0.846)
            plt.savefig(f"./Figures/{session_id}-{title}-QA({group}).png")

def plot_game_traces(traces, median, start=0):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(colors)
    labels = [f'P{i+1+start}' for i in range(10)]
    plt.figure(figsize=(8, 4))
    plt.plot(median[:60], color ="black", lw=2, marker='o', markevery=4, label='GT')
    for trace in range(len(traces)):
        plt.plot(traces[trace][:60], alpha=0.75, label=labels[trace], color=colors[trace+start % 10])
    plt.xlim(xmin=-1, xmax=61)
    plt.legend(bbox_to_anchor=(1.2,0.78), loc='upper right')
    plt.subplots_adjust(right=0.846)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()


def plot_highlighted_engagement_data(data_dict, distance_matrices, groups=['Expert']):
    all_game_names = sorted(set(game_name for session_data in data_dict.values() for group_data in session_data.values() for participant_data in group_data.values()
                                for game_name in participant_data.keys()))
    game_to_index = {game_name: index for index, game_name in enumerate(all_game_names)}

    for session_id, session_data in distance_matrices.items():
        min_distances = []
        for _, games in session_data.items():
            for _, matrix in games.items():
                for i in range(len(matrix)):
                    distances = matrix[i, :]
                    non_zero_distances = distances[np.nonzero(distances)]
                    if non_zero_distances.size > 0:
                        min_distances.append(np.min(non_zero_distances))

        mean_val = np.nanmean(min_distances)
        std_dev = np.nanstd(min_distances)
        distance_threshold = mean_val + 2 * std_dev

        min_distances_dict = {}
        for _, games in session_data.items():
            for game_id, matrix in games.items():
                for participant_id in range(len(matrix)):
                    distances = matrix[participant_id, :]
                    non_zero_distances = distances[np.nonzero(distances)]
                    if non_zero_distances.size > 0:
                        min_distance = np.min(non_zero_distances)
                        if min_distance > distance_threshold:
                            min_distances_dict[(game_id, participant_id)] = min_distance

        for group_id, group_data in data_dict[session_id].items():
            if group_id not in groups:
                continue
            fig, axes = plt.subplots(5, 6, figsize=(14, 10), constrained_layout=True)
            fig.suptitle(f"Highlighted Engagement Outliers for {session_id} - DTW > {distance_threshold}", fontsize=16)
            axes = axes.flatten()
            for ids, (participant_id, participant_data) in enumerate(group_data.items()):
                for game_name, game_values in participant_data.items():
                    try:
                        ax = axes[game_to_index[game_name]]
                        time_values = np.arange(0, len(game_values))
                        alpha_value = 0.3 if (game_name, ids) not in min_distances_dict else 1.0
                        ax.plot(time_values, game_values, label=participant_id, alpha=alpha_value)
                        ax.set_title(game_name)
                    except TypeError:
                        pass
            plt.savefig(f"./Data Analysis/Figures/{session_id}-{group_id}_highlighted_threshold{distance_threshold}.png")
            plt.close()


def plot_engagement_data(data_dict, median_signals_dict):
    all_game_names = sorted(set(game_name for session_data in data_dict.values() for group_data in session_data.values() for participant_data in group_data.values()
                                for game_name in participant_data.keys()))
    game_to_index = {game_name: index for index, game_name in enumerate(all_game_names)}

    for session_id, session_data in data_dict.items():
        for group_id, group_data in session_data.items():
            fig, axes = plt.subplots(5, 6, figsize=(14, 10), constrained_layout=True)
            fig.suptitle(f"Resampled Engagement Data for Session {session_id}", fontsize=16)
            axes = axes.flatten()
            for participant_id, participant_data in group_data.items():
                for game_name, game_values in participant_data.items():
                    try:
                        ax = axes[game_to_index[game_name]]
                        time_values = np.arange(0, len(game_values))
                        ax.plot(time_values, game_values, label=participant_id)
                        if median_signals_dict is not None and game_name in median_signals_dict[session_id][group_id]:
                            median_signal = median_signals_dict[session_id][group_id][game_name]
                            median_time_values = np.arange(0, len(median_signal))
                            ax.plot(median_time_values, median_signal, color='black', linewidth=2)   
                        ax.set_title(game_name)
                        # ax.legend()

                    except TypeError:
                        pass
            plt.savefig(f"./Data Analysis/Figures/{session_id}-{group_id}.png")
            plt.close()


def plot_distance_histogram(distance_matrices, bins=30, xlabel='DTW Distance', ylabel='Frequency'):
    for session_id, _ in distance_matrices.items():
        all_distances = []
        for _, games in distance_matrices[session_id].items():
            for _, matrix in games.items():
                distances = matrix[np.triu_indices_from(matrix, k=1)]
                all_distances.extend(distances)
        all_distances = np.array(all_distances)
        plt.figure(figsize=(8, 6))
        plt.hist(all_distances, bins=bins, color='blue', edgecolor='black')
        plt.title(f'Distance Histogram for {session_id}-Experts')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(f"./Data Analysis/Figures/DTW_Hist_{session_id}.png")
       

def plot_minimum_distance_histogram(distance_matrices, bins=30, xlabel='DTW Distance', ylabel='Frequency'):
    for session_id, _ in distance_matrices.items():
        min_distances = []
        for _, games in distance_matrices[session_id].items():
            for _, matrix in games.items():
                for i in range(len(matrix)):
                    distances = matrix[i, :]
                    non_zero_distances = distances[np.nonzero(distances)]
                    if non_zero_distances.size > 0:
                        min_distances.append(np.min(non_zero_distances))
        
        print(session_id, len(min_distances))
        plt.figure(figsize=(8, 6))
        plt.hist(min_distances, bins=bins, color='blue', edgecolor='black')
        plt.title(f'Minimum DTW Histogram for {session_id}-Experts')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(f"./Data Analysis/Figures/DTW_Hist_Minimums_{session_id}.png")
        plt.close()


def plot_minimum_distance_histogram(distance_matrices, bins=30, xlabel='DTW Distance', ylabel='Frequency'):
    for session_id, _ in distance_matrices.items():
        min_distances = []
        for _, games in distance_matrices[session_id].items():
            for _, matrix in games.items():
                for i in range(len(matrix)):
                    distances = matrix[i, :]
                    non_zero_distances = distances[np.nonzero(distances)]
                    if non_zero_distances.size > 0:
                        min_distances.append(np.min(non_zero_distances))

        plt.figure(figsize=(8, 6))
        plt.hist(min_distances, bins=bins, color='#9C0C35', edgecolor='black')

        # Calculate mean and standard deviation
        mean_val = np.nanmean(min_distances)
        std_dev = np.nanstd(min_distances)
        
        # Plot mean, and 1 and 2 standard deviations from the mean
        plt.axvline(mean_val, color='blue', linestyle='solid', linewidth=4, label='Mean')
        plt.axvline(mean_val + std_dev, color='green', linestyle='dashed', linewidth=4, label='Mean + 1 SD')
        plt.axvline(mean_val + 2 * std_dev, color='purple', linestyle='dashdot', linewidth=4, label='Mean + 2 SD')

        # plt.title(f'Minimum DTW Histogram for {session_id}-Experts')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        plt.tight_layout()
        plt.savefig(f"./Data Analysis/Figures/DTW_Hist_Minimums_{session_id}.png")
        # plt.show()


def plot_distance_matrices(distance_matrices, output_dir='./Data Analysis/Figures', rows_per_figure=4):

    colors = ['blue', 'orange', 'green', 'red', 'purple']

    for session_id, groups in distance_matrices.items():
        for group_id, games in groups.items():
            num_games = len(games)
            cols = math.ceil(num_games / rows_per_figure)
            fig, axes = plt.subplots(rows_per_figure, cols, figsize=(cols * 4, rows_per_figure * 4), constrained_layout=True)
            
            if rows_per_figure == 1 or cols == 1:  # Handle case for a single row or column
                axes = axes.flatten()

            fig.suptitle(f"Distance Matrices for Session {session_id}, Group {group_id}", fontsize=16)

            # Determine the global min and max values for all matrices
            global_min = min(matrix.min() for matrix in games.values())
            global_max = max(matrix.max() for matrix in games.values())

            ax_iter = iter(axes.flatten())
            for game_id, matrix in games.items():
                ax = next(ax_iter)
                cax = ax.imshow(matrix, cmap='copper', interpolation='nearest', vmin=global_min, vmax=global_max)
                ax.set_title(game_id)
                ax.set_xlabel('Participant')
                ax.set_ylabel('Participant')
                ax.set_xticklabels( ['', 'blue', 'orange', 'green', 'red', 'purple'])

                ax.set_yticklabels( ['', 'blue', 'orange', 'green', 'red', 'purple'])

                # Annotate each cell with the distance value
                for i in range(len(matrix)):
                    for j in range(len(matrix)):
                        text_color = 'w' if matrix[i, j] > (global_max - global_min) / 2 else 'w'
                        ax.text(j, i, f'{matrix[i, j]:.2f}', ha="center", va="center", color=text_color)

            # Turn off axes for any unused subplots
            for ax in ax_iter:
                ax.axis('off')

            # Create a single color bar
            fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.01)

            plt.savefig(f"{output_dir}/{session_id}-{group_id}-Matrix.png")
            plt.close()


def game_dtw_scatter(data_dict):
    all_game_names = sorted(set(game_name for session_data in data_dict.values() for group_data in session_data.values() for participant_data in group_data.values()
                                for game_name in participant_data['Engagement'].keys()))
    game_to_index = {game_name: index for index, game_name in enumerate(all_game_names)}

    for session_id, session_data in data_dict.items():
        for group_id, group_data in session_data.items():
            fig, axes = plt.subplots(5, 6, figsize=(14, 10), constrained_layout=True)
            # fig.suptitle(f"Resampled Engagement Data for Session {session_id}", fontsize=16)
            axes = axes.flatten()
            for participant_id, participant_data in group_data.items():
                for game_name, game_values in participant_data["Engagement"].items():
                    try:
                        ax = axes[game_to_index[game_name]]
                        engagement_agreement = game_values
                        qa_agreement = np.mean([session_data[group_id][participant_id]['Visual_SDA'], session_data[group_id][participant_id]['Audio_SDA']])
                        ax.scatter(qa_agreement, engagement_agreement, label=participant_id)
                        ax.set_title(game_name)
                        ax.set_xlim([0, 7])
                        ax.set_ylim([0, 40])
                        # ax.legend()
                    except TypeError:
                        pass
            plt.savefig(f"./Figures/{session_id}-{group_id}.png")
            plt.close()


def plot_game_sda_histogram(game_sda_list, session_mean, session_ci, session_id):
    plt.figure(figsize=(10,6))
    plt.hist(game_sda_list, bins=np.linspace(-1, 1, 21), edgecolor='black')
    plt.title(f'SDA Histogram - {session_id} ({session_mean}±{session_ci})', fontsize=15)
    plt.ylabel('Frequency', fontsize=12)        
    plt.xlabel('SDA', fontsize=12)
    plt.savefig(f"./Figures/SDA_Histograms/SDA_Hist_{session_id}.png")

def plot_game_kappa_histogram(game_sda_list, session_mean, session_ci, session_id):
    plt.figure(figsize=(10,6))
    plt.hist(game_sda_list, bins=np.linspace(-1, 1, 21), edgecolor='black')
    plt.title(f'Cohen\'s Kappa Histogram - {session_id} ({session_mean}±{session_ci})', fontsize=15)
    plt.xlabel('Cohen\'s Kappa', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)        
    plt.savefig(f"./Figures/Cohen's Kappa/Cohens_Hist_{session_id}.png")

def plot_sda_scatter(sda_scores, title=""):
    engagement_mean_sdas = [np.mean(list(participant_data['Engagement'].values())) for session_data in sda_scores.values() for participant_data in session_data.values()]
    visual_sdas = [participant_data['Visual_SDA'] for session_data in sda_scores.values() for participant_data in session_data.values()]
    audio_sdas = [participant_data['Audio_SDA'] for session_data in sda_scores.values() for participant_data in session_data.values()]
    plt.figure(figsize=(10, 6))
    plt.scatter(visual_sdas, engagement_mean_sdas, label="Visual Task")
    plt.scatter(audio_sdas, engagement_mean_sdas, label="Audio Task")
    plt.ylabel('SDA (Engagement Task)')
    plt.xlabel('SDA (QA Task)')
    # plt.xlim([0,1])
    # plt.ylim([-1,1])
    plt.title(f'{title}: Scatter Plot of SDA')
    plt.legend()
    plt.show()

def plot_sda_scatter_grouped(sda_scores):

    visual_mturks = [participant_data['Visual_SDA'] for session_data in sda_scores.values() for participant_data in session_data['Mturk'].values()]
    audio_mturks = [participant_data['Audio_SDA'] for session_data in sda_scores.values() for participant_data in session_data['Mturk'].values()]
    engagement_mturks = [np.mean(list(participant_data['Engagement'].values()))  for session_data in sda_scores.values() for participant_data in session_data['Mturk'].values()]
    mturks_ci = [compute_confidence_interval(list(participant_data['Engagement'].values()))[1] for session_data in sda_scores.values() for participant_data in session_data['Mturk'].values()]
    qa_mturks = [np.mean([visual, audio]) for (visual, audio) in zip(visual_mturks, audio_mturks)]

    visual_experts = [participant_data['Visual_SDA'] for session_data in sda_scores.values() for participant_data in session_data['Expert'].values()]
    audio_experts = [participant_data['Audio_SDA'] for session_data in sda_scores.values() for participant_data in session_data['Expert'].values()]
    engagement_experts = [np.mean(list(participant_data['Engagement'].values())) for session_data in sda_scores.values() for participant_data in session_data['Expert'].values()]
    experts_ci = [compute_confidence_interval(list(participant_data['Engagement'].values()))[1] for session_data in sda_scores.values() for participant_data in session_data['Expert'].values()]
    qa_experts = [np.mean([visual, audio]) for visual, audio in zip(visual_experts, audio_experts)]

    plt.figure(figsize=(7, 5))
    plt.errorbar(qa_mturks, engagement_mturks, label="Crowdworkers", fmt="D", yerr=mturks_ci, markeredgecolor="black")
    plt.errorbar(qa_experts, engagement_experts, label="Experts", fmt="o", yerr=experts_ci, markeredgecolor="black")
    plt.ylabel('Mean DTW (Engagement Tasks)')
    plt.xlabel('Mean DTW (QA Tasks)')
    plt.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.15))
    plt.show()

def plot_correlations(audio_correlations, visual_correlations, sort=True, title=""):
    if sort:
        audio_sorted_items = sorted(audio_correlations.items(), key=lambda x: x[1])
        game_names = [item[0] for item in audio_sorted_items]
        audio_correlation_values = [item[1] for item in audio_sorted_items]
        visual_correlation_values = [visual_correlations[game] for game in game_names]
    else:
        game_names = list(visual_correlations.keys())
        audio_correlation_values = list(audio_correlations.values())
        visual_correlation_values = list(visual_correlations.values())
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(range(len(game_names)), audio_correlation_values, marker='o', label="Audio Correlation")
    plt.scatter(range(len(game_names)), visual_correlation_values, marker='^', label="Visual Correlation")
    plt.xticks(rotation=90)  # Rotate the x-axis labels for readability
    fig.axes[0].get_xaxis().set_ticks([])
    plt.xlabel('Stimuli')
    plt.ylabel('Correlation')
    plt.title(f'{title}: Correlation of Audio/Visual SDA with Engagement SDA')
    plt.tight_layout()
    plt.legend(ncols=2, loc="upper center")
    plt.show()


def plot_correlations_grouped(mturk_correlations, expert_correlations, sort=True, title=""):

    if sort:
        expert_sorted_items = sorted(expert_correlations.items(), key=lambda x: x[1])
        game_names = [item[0] for item in expert_sorted_items]
        expert_correlation_values = [item[1] for item in expert_sorted_items]
        mturk_correlation_values = [mturk_correlations[game] for game in game_names]
    else:
        game_names = list(mturk_correlations.keys())
        expert_correlation_values = list(expert_correlations.values())
        mturk_correlation_values = list(mturk_correlations.values())

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(range(len(game_names)), expert_correlation_values, marker='o', label="Expert Correlation")
    plt.scatter(range(len(game_names)), mturk_correlation_values, marker='^', label="Mturk Correlation")
    plt.xticks(rotation=90) 
    fig.axes[0].get_xaxis().set_ticks([])
    plt.xlabel('Stimuli')
    plt.ylabel('Correlation')
    plt.title(f'{title}: Correlation of Audio/Visual SDA with Engagement SDA')
    plt.tight_layout()
    plt.legend(ncols=2, loc="upper center")
    plt.show()


def plot_agreement_matrix(agreement_matrix):
    plt.figure(figsize=(10, 10))
    sns.heatmap(agreement_matrix, xticklabels=False, yticklabels=range(len(agreement_matrix)), cmap="vlag")
    plt.title('Agreement Matrix Heatmap')
    plt.ylabel('Participant ID')
    plt.xlabel('Time Point')
    plt.show()

def plot_individual_matrix(IM, title):
    fig = plt.figure(figsize=(10, 10))
    # fig.patch.set_alpha(0)
    plt.title(title)
    cmap = ListedColormap(['white', 'white', 'white', 'white'])
    value_map = {"↓": 0, "=": 1, "↑": 2, "": 3, "x": 4}
    numeric_IM = np.vectorize(value_map.get)(IM)
    plt.imshow(numeric_IM, cmap=cmap)
    for i in range(numeric_IM.shape[0]):
        for j in range(numeric_IM.shape[1]):
            plt.text(j, i, IM[i, j], ha='center', va='center', color='k')
    plt.tight_layout()
    plt.show()

def plot_trace(trace1, trace2, sda):
    plt.figure()
    plt.plot(range(len(trace1)), trace1)
    plt.plot(range(len(trace2)), trace2)
    for i in range(1, len(sda)):
        color = 'green' if sda[i] >= 0 else 'red'
        plt.fill_between([i-1, i], trace1[i-1:i+1], trace2[i-1:i+1], 
                         color=color, alpha=0.3)
    green_patch = plt.Rectangle((0,0),1,1,fc="green", edgecolor = 'none', alpha=0.3)
    red_patch = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none', alpha=0.3)
    plt.legend([green_patch, red_patch], ['Agree', 'Disagree'], loc="upper left")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

from matplotlib.cm import get_cmap

def plot_filtered_traces(original, std1, std2):

    traces = original['Session-1']['Expert']
    std1_traces = std1['Session-1']['Expert']
    std2_traces = std2['Session-1']['Expert']

    fig, axes = plt.subplots(1, 3, constrained_layout=False, figsize=(16, 5))
    axes = axes.flatten()

    participant_mapping = {}    
    name_mapping = {}
    num_participants = len(traces)  # Number of participants
    cmap = get_cmap('viridis', 5)  # Getting the Viridis colormap for the number of participants
    colors = [cmap(i) for i in range(num_participants)]  # Generating colors for each participant
    markers = ['o', 'v', 's', '^', 'x']
    counter = 0
    for paricipant in traces.keys():
        participant_mapping[paricipant] = counter
        name_mapping[paricipant] = f"P{counter+1}"
        counter += 1
        trace = traces[paricipant]['wolf3d']
        time_values = np.arange(0, len(trace))
        ax = axes[0]
        ax.plot(time_values, trace, label=name_mapping[paricipant], color=colors[participant_mapping[paricipant]], marker=markers[participant_mapping[paricipant]], markevery=5, markeredgecolor='black', linewidth=2)
        ax.set_title("Original Traces")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
    for paricipant in std2_traces.keys():
        try:
            trace = std2_traces[paricipant]['wolf3d']
            time_values = np.arange(0, len(trace))
            ax = axes[1]
            ax.plot(time_values, trace, label=name_mapping[paricipant], color=colors[participant_mapping[paricipant]], marker=markers[participant_mapping[paricipant]], markevery=5, markeredgecolor='black', linewidth=2)
            ax.set_title("2 St.Dev Filter")
            ax.set_xlabel("Time (s)")
        except KeyError:
            continue

    for paricipant in std1_traces.keys():
        try:
            trace = std1_traces[paricipant]['wolf3d']
            time_values = np.arange(0, len(trace))
            ax = axes[2]
            ax.set_title("1 St.Dev Filter")
            ax.set_xlabel("Time (s)")
            ax.plot(time_values, trace, label=name_mapping[paricipant], color=colors[participant_mapping[paricipant]], marker=markers[participant_mapping[paricipant]], markevery=5, markeredgecolor='black', linewidth=2)
        except KeyError:
            continue
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    fig.legend(loc="upper center", ncol=5, labels=['P1', 'P2', 'P3', 'P4', 'P5'], bbox_to_anchor=(0.5, 1.015))
    plt.show()
    return

    for session_id, session_data in data_dict.items():
        for group_id, group_data in session_data.items():
            fig, axes = plt.subplots(5, 6, figsize=(14, 10), constrained_layout=True)
            fig.suptitle(f"Resampled Engagement Data for Session {session_id}", fontsize=16)
            axes = axes.flatten()
            for participant_id, participant_data in group_data.items():
                for game_name, game_values in participant_data.items():
                    try:
                        ax = axes[game_to_index[game_name]]
                        time_values = np.arange(0, len(game_values))
                        ax.plot(time_values, game_values, label=participant_id)
                        if median_signals_dict is not None and game_name in median_signals_dict[session_id][group_id]:
                            median_signal = median_signals_dict[session_id][group_id][game_name]
                            median_time_values = np.arange(0, len(median_signal))
                            ax.plot(median_time_values, median_signal, color='black', linewidth=2)   
                        ax.set_title(game_name)
                        # ax.legend()

                    except TypeError:
                        pass
            plt.savefig(f"./Data Analysis/Figures/{session_id}-{group_id}.png")
            plt.close()

def execute():
    font = {'size': 14}
    matplotlib.rc('font', **font)

    # plot_brightness_and_pitch_data(visual_data, ["Session-1", "Session-2", "Session-3", "Session-7"], "Expert", "", green_gt_trace)
    # plot_brightness_and_pitch_data(visual_data, ["Session-1", "Session-2", "Session-3"], "Mturk", "", green_gt_trace)

    # plot_brightness_and_pitch_data(audio_data, ["Session-1", "Session-2", "Session-3", "Session-7"], "Expert", "", pitch_gt_trace, True)
    # plot_brightness_and_pitch_data(audio_data, ["Session-1", "Session-2", "Session-3"], "Mturk", "", pitch_gt_trace, True)
