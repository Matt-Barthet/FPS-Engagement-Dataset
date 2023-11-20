import numpy as np
from plotting import *
from collections import defaultdict, Counter

def calculate_median_signal(data_dict):
    all_game_names = sorted(set(game_name for session_data in data_dict.values() 
                                for participant_data in session_data.values() 
                                for game_name in participant_data.keys()))
    median_signals = {}
    for session_id, session_data in data_dict.items():
        median_signals[session_id] = {}
        for game_name in all_game_names:
            all_game_values = [participant_data[game_name] 
                               for participant_id, participant_data in session_data.items()
                               if game_name in participant_data]
            if all_game_values:
                max_length = max(len(game_values) for game_values in all_game_values)
                padded_game_values = [np.pad(game_values, (0, max_length - len(game_values)), 'edge')
                                     for game_values in all_game_values]
                median_signal = np.median(np.stack(padded_game_values, axis=-1), axis=1)
                median_signals[session_id][game_name] = median_signal
    return median_signals

def calculate_median_signal_excluding(data_dict):
    all_game_names = sorted(set(game_name for session_data in data_dict.values() for group_data in session_data.values() for participant_data in group_data.values() for game_name in participant_data.keys()))
    median_signals = {}
    for session_id, session_data in data_dict.items():
        median_signals[session_id] = {"Expert": {}, "Mturk": {}}

        for group_name, group_data in session_data.items():
            for participant_id in group_data.keys():
                median_signals[session_id][group_name][participant_id] = {}
                for game_name in all_game_names:
                    all_game_values = [other_data[game_name] for other_id, other_data in group_data.items() if game_name in other_data and other_id != participant_id]
                    if all_game_values:

                        lengths = []
                        for value in all_game_values:
                            if value is not None:
                                lengths.append(len(value))

                        if len(lengths) != 0:
                            max_length = max(lengths)
                            padded_game_values = []
                            for value in all_game_values:
                                if value is not None:
                                    padded_game_values.append(np.pad(value, (0, max_length - len(value)), 'edge'))

                            median_signal = np.median(np.stack(padded_game_values, axis=-1), axis=1)
                            median_signals[session_id][group_name][participant_id][game_name] = median_signal
                        else:
                            median_signals[session_id][group_name][participant_id][game_name] = np.nan
    return median_signals


def create_individual_matrices(ordinal_data, threshold=0.0):
    all_game_names = sorted(set(game_name for session_data in ordinal_data.values() for participant_data in session_data.values() for group_data in participant_data.values() for game_name in group_data.keys()))
    individual_matrices = {}
    for session_id, session_data in ordinal_data.items():
        individual_matrices[session_id] = {}
        for group_id, group_data in session_data.items():

            individual_matrices[session_id][group_id] = {}
            for game_name in all_game_names:
                individual_matrices[session_id][group_id][game_name] = {}

                # Collect all game values for current game_name and store individual matrices
                all_game_values = [participant_data[game_name] for participant_data in group_data.values()]
                all_individual_matrices = [create_individual_matrix(game_values, threshold) for game_values in all_game_values if game_values is not None]

                if len(all_individual_matrices) == 0:
                    continue

                # Find max_length among all individual matrices
                max_length = max(matrix.shape[0] for matrix in all_individual_matrices)

                for participant_id, game_values in zip(group_data.keys(), all_game_values):

                    if game_values is not None and len(game_values) != 0:
                        individual_matrix = create_individual_matrix(game_values, threshold)
                        
                        # Apply padding to individual matrix
                        pad_width = max_length - individual_matrix.shape[0]
                        padded_individual_matrix = np.pad(individual_matrix, ((0, pad_width), (0, pad_width)), 'edge')

                        individual_matrices[session_id][group_id][game_name][participant_id] = padded_individual_matrix
                        # plot_individual_matrix(padded_individual_matrix, f"{session_id}-{group_id}-{game_name}-{participant_id}")

    return individual_matrices


def create_individual_matrix(trace, tthreshold):
    N = len(trace)
    IM = np.full((N, N), "")  # initialize the matrix with no change
    for i in range(N):
        for j in range(i, N):
            difference = trace[i] - trace[j]
            if difference > tthreshold:
                IM[i, j] = "↑"
            elif difference < -tthreshold:
                IM[i, j] = "↓"
            else:
                IM[i, j] = "="
    return IM


def calculate_agreement_matrices(individual_matrices):
    agreement_matrices = {}
    for session_id, session_data in individual_matrices.items():
        agreement_matrices[session_id] = {}
        for group_id, group_data in session_data.items():
            agreement_matrices[session_id][group_id] = {}
            for game_name, game_data in group_data.items():
                num_participants = len(game_data)
                if num_participants < 2:
                    agreement_matrix = list(game_data.values())
                else:
                    num_timepoints = list(game_data.values())[0].shape[0]
                    agreement_matrix = np.full((num_timepoints, num_timepoints), "")                
                    for i in range(num_timepoints):
                        for j in range(num_timepoints):
                            direction_counts = defaultdict(int)
                            for individual_matrix in game_data.values():
                                direction_counts[individual_matrix[i, j]] += 1
                            max_count_direction = max(direction_counts.items(), key=lambda x: x[1])

                            if max_count_direction[1] >= num_participants / 2: # If half the participants agree on the direction change.
                                agreement_matrix[i, j] = max_count_direction[0]
                agreement_matrices[session_id][group_id][game_name] = np.asarray(agreement_matrix)
            # plot_individual_matrix(agreement_matrix, f"{session_id}-{game_name}")
    return agreement_matrices

if __name__ == "__main__":
    
    engagement_data = np.load("./Engagement_Task.npy", allow_pickle=True).item()    
    sessions_to_use = ['Session-1', 'Session-2', 'Session-3', 'Session-7']
    groups_to_use = ['Expert', 'Mturk']

    for session_id in list(engagement_data.keys()):
        if session_id not in sessions_to_use:
            del engagement_data[session_id]
            continue
        for group_id in list(engagement_data[session_id].keys()):
            if group_id not in groups_to_use:
                del engagement_data[session_id][group_id]
    
    median_signals_dict = calculate_median_signal_excluding(engagement_data)
    # plot_engagement_data(engagement_data, None)
    np.save("./Engagement_Median.npy", median_signals_dict)

    individual_matrices = create_individual_matrices(engagement_data)
    agreement_matrices = calculate_agreement_matrices(individual_matrices)
    # plot_engagement_data(ordinal_data, None)

    # Define the categories
    categories = ['↑', '↓', '=', '']

    # Initialize a counter for each category
    counts = Counter({category: 0 for category in categories})

    # Iterate over the agreement matrices
    for session_id, session_data in agreement_matrices.items():
        for group_id, group_data in session_data.items():
            for game_name, agreement_matrix in group_data.items():
                diagonal_list = []
                diagonal_matrix = np.full(agreement_matrix.shape, "")
                for i in range(agreement_matrix.shape[0]):
                    for offset in range(1,5):
                        try:
                            diagonal_list.append(agreement_matrix[i][i+offset])
                            diagonal_matrix[i][i+offset] = diagonal_list[-1]
                        except IndexError:
                            pass
                counts.update(Counter(diagonal_list))
                # plot_individual_matrix(diagonal_matrix, "") # @Kosmas: I double checked this and it works well now.

    # Create a bar plot of the counts
    plt.bar(counts.keys(), counts.values())
    plt.xticks(['↑', '↓', '=', ''], ["Increase", "Decrease", "Stable", "Disagreement"])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Agreement Labels - Frequencies within 5 TW of Diagonal')
    plt.show()


