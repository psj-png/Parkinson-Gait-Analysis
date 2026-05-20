import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Gait_Analysis\extracted_data\gait_cleaned_labeled.csv')
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x='frame', y='knee_angle', hue='label')
    plt.title("Gait ROM Analysis")
    plt.show()