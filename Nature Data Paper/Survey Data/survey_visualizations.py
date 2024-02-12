import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":

    survey_data = pd.read_csv("./Nature Data Paper/Raw CSVs/Survey Responses - Experts.csv")

    columns_we_want = ['Age', 'Gender', 'Ethnicity', 'Handedness', 'Education', 'Familiarity with Videogames', 'Familiarity with First person shooter games', 'Familiarity with affect annotation tasks']

    plt.rcParams['font.size'] = 16
    


    ethnicity_data = survey_data['Ethnicity'].values

    for id in range(len(ethnicity_data)):
        if ethnicity_data[id] == "Asian / Pacific Islander":
            ethnicity_data[id] = 0
        elif ethnicity_data[id] == "Caucasian":
            ethnicity_data[id] = 1
        elif ethnicity_data[id] == "African":
            ethnicity_data[id] = 2
        else:
            print(ethnicity_data[id])

    plt.figure()
    plt.hist(ethnicity_data, bins=[0, 1, 2, 3], edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    plt.xticks([0.5, 1.5, 2.5], ["Asian", "Caucasian", "African",])
    plt.xlabel("Ethnicity")
    plt.ylabel("Frequency (%)")
    plt.yticks(np.linspace(0, 0.9, 10), np.linspace(0, 90, 10, dtype=int))
    plt.tight_layout()
    plt.savefig("./Nature Data Paper/ethnicity_distribution.png")

    handedness_data = survey_data['Handedness'].values
    for id in range(len(handedness_data)):
        if handedness_data[id] == "Left-handed":
            handedness_data[id] = 0
        elif handedness_data[id] == "Right-handed":
            handedness_data[id] = 1
        elif handedness_data[id] == "Ambidextrous":
            handedness_data[id] = 2

    plt.figure()
    plt.hist(handedness_data, bins=[0, 1, 2, 3], edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    plt.xticks([0.5, 1.5, 2.5], ["Left-handed", "Right-handed", "Ambidextrous"])
    plt.xlabel("Handedness")
    plt.ylabel("Frequency (%)")
    plt.yticks(np.linspace(0, 0.9, 10), np.linspace(0, 90, 10, dtype=int))
    plt.tight_layout()
    plt.savefig("./Nature Data Paper/handedness_distribution.png")


    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(16, 10))

    age_data = survey_data['Age'].values

    for id in range(len(age_data)):
        if age_data[id] == "18-25":
            age_data[id] = 0
        elif age_data[id] == "25-35":
            age_data[id] = 1
        elif age_data[id] == "35-45":
            age_data[id] = 2
        elif age_data[id] == "45+":
            age_data[id] = 3

    ax = axs[0][0]
    ax.hist(age_data, bins=[0, 1, 2, 3, 4], edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], ["18-25", "25-35", "35-45", "45+"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency (%)")
    ax.set_yticks(np.linspace(0, 0.9, 10), np.linspace(0, 90, 10, dtype=int))
    
    gender_data = survey_data['Gender'].values

    for id in range(len(gender_data)):
        if gender_data[id] == "Male":
            gender_data[id] = 1
        elif gender_data[id] == "Female":
            gender_data[id] = 0
        elif gender_data[id] == "Non-Binary":
            gender_data[id] = 2

    ax = axs[0][1]
    ax.hist(gender_data, bins=[0, 1, 2, 3], edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    ax.set_xticks([0.5, 1.5, 2.5], ["Female", "Male", "Non-Binary",])
    ax.set_xlabel("Gender")
    ax.set_yticks(np.linspace(0, 0.9, 10), ['' for i in range(10)])

    # Education
    education_data = survey_data['Education'].values
    unique_education = sorted(set(education_data))
    education_mapping = {edu: i for i, edu in enumerate(unique_education)}
    education_data = [education_mapping[edu] for edu in education_data]

    ax = axs[0][2]
    ax.hist(education_data, bins=range(len(unique_education)+1), edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    ax.set_xticks([0.5, 1.5], ['Secondary', 'University'])
    ax.set_xlabel("Education Completed")
    ax.set_yticks(np.linspace(0, 0.9, 10),  ['' for i in range(10)])


    # Familiarity with Videogames
    video_games_data = survey_data['Familiarity with Videogames'].values
    ax = axs[1][0]
    ax.hist(video_games_data, bins=np.arange(1, max(video_games_data)+2)-0.5, edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    ax.set_xticks([1,2,3,4,5])
    ax.set_xlabel("Familiarity with Videogames")
    ax.set_ylabel("Frequency (%)")
    ax.set_yticks(np.linspace(0, 0.9, 10), np.linspace(0, 90, 10, dtype=int))

    # Familiarity with First person shooter games
    fps_games_data = survey_data['Familiarity with First person shooter games'].values
    ax = axs[1][1]
    ax.hist(fps_games_data, bins=np.arange(1, max(fps_games_data)+2)-0.5, edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    ax.set_xlabel("Familiarity with FPS Games")
    ax.set_xticks([1,2,3,4,5])
    ax.set_yticks(np.linspace(0, 0.9, 10),['' for i in range(10)])


    # Familiarity with affect annotation tasks
    affect_annotation_data = survey_data['Familiarity with affect annotation tasks'].values
    ax = axs[1][2]
    ax.hist(affect_annotation_data, bins=np.arange(1, max(affect_annotation_data)+2)-0.5, edgecolor='black', color='#9C0C35', rwidth=0.9, density=True)
    ax.set_xlabel("Familiarity with Affect Annotation Tasks")
    ax.set_xticks([1,2,3,4,5])
    ax.set_yticks(np.linspace(0, 0.9, 10),['' for i in range(10)])

    plt.savefig("./Nature Data Paper/survey_data.png")
