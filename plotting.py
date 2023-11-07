import matplotlib.pyplot as plt
import numpy as np
from utility_functions import *
from matplotlib.colors import ListedColormap
import seaborn as sns


def plot_brightness_and_pitch_data(session, session_ids, group, title, gt_signal, sound=False,start=0):
    plt.figure(figsize=(8, 4))


    labels = [f'P{i+1+start}' for i in range(10)]

    counter = 0
    if not sound:
        plt.plot(gt_signal[:240], color ="black", lw=2, marker='o', markevery=4, label="GT")
        plt.xticks([0, 60, 120, 180, 240], [0, 15, 30, 45, 60])
        plt.xlim(xmin=-1, xmax=241)
        for session_id, session_data in session.items():
            if session_id in session_ids:
                for _, participant_data in session_data[group].items():
                    plt.plot(list(participant_data.values())[0][:240], alpha=0.75, label=labels[counter])
                    counter+=1
    else:
        plt.xticks([0, 40, 80, 120], [0, 10, 20, 30])
        plt.plot(gt_signal[:120], color ="black", lw=2, marker='o', markevery=4, label='GT')
        plt.xlim(xmin=-1, xmax=121)
        for session_id, session_data in session.items():
            if session_id in session_ids:
                for _, participant_data in session_data[group].items():
                    plt.plot(list(participant_data.values())[0][:120], alpha=0.75, label=labels[counter])
                    counter+=1

    plt.title(f'{title}')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.2,1.03), loc='upper right')
    plt.subplots_adjust(right=0.846)
    plt.show()

def plot_game_traces(traces, median, start=0):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(colors)
    labels = [f'P{i+1+start}' for i in range(10)]
    plt.figure(figsize=(8, 4))
    plt.plot(median[:240], color ="black", lw=2, marker='o', markevery=4, label='GT')
    for trace in range(len(traces)):
        plt.plot(traces[trace][:240], alpha=0.75, label=labels[trace], color=colors[trace+start % 10])
    plt.xticks([0, 60, 120, 180, 240], [0, 15, 30, 45, 60])
    plt.xlim(xmin=-1, xmax=241)
    plt.legend(bbox_to_anchor=(1.2,0.78), loc='upper right')
    plt.subplots_adjust(right=0.846)
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()


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
    plt.ylabel('Mean SDA (Engagement Tasks)')
    plt.xlabel('Mean SDA (QA Tasks)')
    # plt.title('SDA Scatter (Visual Task)')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.15))
    plt.hlines(y=0, xmin=-1, xmax=1, color="black")
    plt.vlines(x=0, ymin=-1, ymax=1, color="black")
    plt.show()

    """plt.figure(figsize=(8, 6))
    plt.scatter(visual_mturks, engagement_mturks, label="Crowdworkers")
    plt.scatter(visual_experts, engagement_experts, label="Experts")
    plt.ylabel('Mean SDA (Engagement Tasks)')
    plt.xlabel('SDA (Visual Task)')
    plt.title('SDA Scatter (Visual Task)')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(audio_mturks, engagement_mturks, label="Crowdworkers")
    plt.scatter(audio_experts, engagement_experts, label="Experts")
    plt.ylabel('Mean SDA (Engagement Tasks)')
    plt.xlabel('SDA (Audio Task)')
    plt.title('SDA Scatter (Audio Task)')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend()
    plt.show()"""

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

