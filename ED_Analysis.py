from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X, y):
    from imblearn.over_sampling import KMeansSMOTE
    from sklearn.cluster import KMeans

    kmeans_smote = KMeansSMOTE(
        kmeans_estimator=KMeans(n_clusters=250, random_state=42),
        cluster_balance_threshold=0.005,
        random_state=42
    )
    X_res, y_res = kmeans_smote.fit_resample(X, y)

    return X_res,y_res

import numpy as np
import matplotlib.pyplot as plt


def visualization(DB,temp_label,lb):
    if DB=="UNSW-NB15":

        if lb=="before":
            df = pd.DataFrame(temp_label)
            import matplotlib.pyplot as plt
            x = df["attack_cat"].unique()
            x.sort()
            x.tolist()
            x1 = x
            y = df["attack_cat"].value_counts()
            new_x = []
            for i in range(len(x)):
                new_x.append(y[i])
            y1 = new_x

            x_indices = list(range(len(y1)))
            plt.figure(figsize=(10, 8))
            plt.bar(x1, y1)
            plt.xlabel("classes in UNSW dataset")
            plt.xticks(x_indices, labels=x1)
            plt.ylabel("count of each classes in UNSW datset")
            plt.title("class distribution before smote in UNSW-15 dataset")
            plt.savefig(f"ImageResults/Class_Distribution/Before_smote/{DB}_class_distribution", dpi=600)
            plt.show()
        if lb=="after":
            df = pd.DataFrame(temp_label)
            import matplotlib.pyplot as plt
            x = df["attack_cat"].unique()
            x.sort()
            x.tolist()
            x1 = x
            y = df["attack_cat"].value_counts()
            new_x = []
            for i in range(len(x)):
                new_x.append(y[i])
            y1 = new_x

            x_indices = list(range(len(y1)))
            plt.figure(figsize=(10, 8))
            plt.bar(x1, y1)
            plt.xlabel("classes in UNSW dataset")
            plt.xticks(x_indices, labels=x1)
            plt.ylabel("count of each classes in UNSW dataset")
            plt.title("class distribution After smote in UNSW-15 dataset")
            plt.savefig(f"ImageResults/Class_Distribution/After_smote/{DB}_class_distribution",dpi=600)
            plt.show()

    if DB=="N-BaIoT":

        if lb=="before":
            df = pd.DataFrame(temp_label)
            # important needed
            import matplotlib.pyplot as plt
            x = df["category"].unique()
            x.sort()
            x.tolist()
            x1 = x
            y = df["category"].value_counts()
            new_x = []
            for i in range(len(x)):
                new_x.append(y[i])
            y1 = new_x

            x_indices = list(range(len(y1)))
            plt.figure(figsize=(10, 8))
            plt.bar(x1, y1)
            plt.xlabel("classes in N-BaIoT dataset")
            plt.xticks(x_indices, labels=x1)
            plt.ylabel("count of each classes in N-BaIoT datset")
            plt.title("class distribution before smote in N-BaIoT dataset")
            plt.savefig(f"ImageResults/Class_Distribution/Before_smote/{DB}_class_distribution", dpi=600)
            plt.show()
        if lb=="after":
            df = pd.DataFrame(temp_label)
            # important needed
            import matplotlib.pyplot as plt
            x = df["category"].unique()
            x.sort()
            x.tolist()
            x1 = x
            y = df["category"].value_counts()
            new_x = []
            for i in range(len(x)):
                new_x.append(y[i])
            y1 = new_x

            x_indices = list(range(len(y1)))
            plt.figure(figsize=(10, 8))
            plt.bar(x1, y1)
            plt.xlabel("classes in N-BaIoT dataset")
            plt.xticks(x_indices, labels=x1)
            plt.ylabel("count of each classes in N-BaIoT datset")
            plt.title("class distribution After smote in N-BaIoT dataset")
            plt.savefig(f"ImageResults/Class_Distribution/After_smote/{DB}_class_distribution", dpi=600)
            plt.show()

    if DB=="CICIDS2015":
        if lb=="before":

            df = pd.DataFrame(temp_label)
            # important needed
            import matplotlib.pyplot as plt
            x = df["Label"].unique()
            x.sort()
            x.tolist()
            x1 = x
            y = df["Label"].value_counts()
            new_x = []
            for i in range(len(x)):
                new_x.append(y[i])
            y1 = new_x

            x_indices = list(range(len(y1)))
            plt.figure(figsize=(10, 8))
            plt.bar(x1, y1)
            plt.xlabel("classes in CICIDS2015 dataset")
            plt.xticks(x_indices, labels=x1)
            plt.ylabel("count of each classes in CICIDS2015 datset")
            plt.title("class distribution before smote in CICIDS2015 dataset")
            plt.savefig(f"ImageResults/Class_Distribution/Before_smote/{DB}_class_distribution", dpi=600)
            plt.show()

        if lb=="after":

            df = pd.DataFrame(temp_label)
            # important needed
            import matplotlib.pyplot as plt
            x = df["Label"].unique()
            x.sort()
            x.tolist()
            x1 = x
            y = df["Label"].value_counts()
            new_x = []
            for i in range(len(x)):
                new_x.append(y[i])
            y1 = new_x

            x_indices = list(range(len(y1)))
            plt.figure(figsize=(10, 8))
            plt.bar(x1, y1)
            plt.xlabel("classes in CICIDS2015 dataset")
            plt.xticks(x_indices, labels=x1)
            plt.ylabel("count of each classes in CICIDS2015 datset")
            plt.title("class distribution After smote in CICIDS2015 dataset")
            plt.savefig(f"ImageResults/Class_Distribution/After_smote/{DB}_class_distribution", dpi=600)
            plt.show()

import random


def simulate(node, DB, data):
    if DB == "UNSW-NB15":
        base_station = np.array([np.random.randint(0, 100), np.random.randint(0, 100)])
        nodes = np.random.rand(node, 2) * [100, 100]

        # Load and clean data
        # data = np.load("data_loader/UNSW-NB15_features.npy")

        # data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0, neginf=0, posinf=0)

        # Assign a random data row to each node
        if data.shape[0] < node:
            raise ValueError("Not enough data rows to assign to each node.")

        node_data = random.sample(list(data), node)

        # Plot setup
        plt.figure(figsize=(15, 8))
        plt.scatter(base_station[0], base_station[1], color="red", s=500, marker="^", label="Base Station")
        plt.scatter(nodes[:, 0], nodes[:, 1], color="green", s=40, marker="o", label="IoT Nodes")

        for ii in range(nodes.shape[0]):
            plt.text(nodes[ii][0] + 2, nodes[ii][1] + 3, f"IOT{ii + 1}",
                     bbox=dict(fill=False, edgecolor='green', linewidth=1))

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.axis("off")
        plt.legend()
        plt.pause(2)

        for i in range(nodes.shape[0]):
            positions_list = np.linspace(nodes[i], base_station, 50)
            green_dots = []

            for pos in positions_list:
                dot = plt.scatter(pos[0], pos[1], color="green", s=2, marker="o")
                green_dots.append(dot)
                plt.draw()
                plt.pause(0.01)

            plt.pause(0.5)

            # Simulate "data transmission" (here we just print it)
            print(f"\nNode {i + 1} transmitting data to base station:")
            print(node_data[i])  # or log to file / process as needed

            plt.pause(0.5)

            # Remove movement dots
            for dot in green_dots:
                dot.remove()

        plt.pause(3)
        plt.close()

    if DB == "N-BaIoT":

        base_station = np.array([np.random.randint(0, 100), np.random.randint(0, 100)])
        nodes = np.random.rand(node, 2) * [100, 100]

        # Load and clean data
        # data = np.load("data_loader/UNSW-NB15_features.npy")
        # data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0, neginf=0, posinf=0)

        # Assign a random data row to each node
        if data.shape[0] < node:
            raise ValueError("Not enough data rows to assign to each node.")

        node_data = random.sample(list(data), node)

        # Plot setup
        plt.figure(figsize=(15, 8))
        plt.scatter(base_station[0], base_station[1], color="red", s=500, marker="^", label="Base Station")
        plt.scatter(nodes[:, 0], nodes[:, 1], color="green", s=40, marker="o", label="IoT Nodes")

        for ii in range(nodes.shape[0]):
            plt.text(nodes[ii][0] + 2, nodes[ii][1] + 3, f"IOT{ii + 1}",
                     bbox=dict(fill=False, edgecolor='green', linewidth=1))

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.axis("off")
        plt.legend()
        plt.pause(2)

        for i in range(nodes.shape[0]):
            positions_list = np.linspace(nodes[i], base_station, 50)
            green_dots = []

            for pos in positions_list:
                dot = plt.scatter(pos[0], pos[1], color="green", s=2, marker="o")
                green_dots.append(dot)
                plt.draw()
                plt.pause(0.01)

            plt.pause(0.5)

            # Simulate "data transmission" (here we just print it)
            print(f"\nNode {i + 1} transmitting data to base station:")
            print(node_data[i])  # or log to file / process as needed

            plt.pause(0.5)

            # Remove movement dots
            for dot in green_dots:
                dot.remove()

        plt.pause(3)
        plt.close()

    if DB == "CICIDS2015":
        base_station = np.array([np.random.randint(0, 100), np.random.randint(0, 100)])
        nodes = np.random.rand(node, 2) * [100, 100]

        # data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0, neginf=0, posinf=0)

        # Assign a random data row to each node
        if data.shape[0] < node:
            raise ValueError("Not enough data rows to assign to each node.")

        node_data = random.sample(list(data), node)

        # Plot setup
        plt.figure(figsize=(15, 8))
        plt.scatter(base_station[0], base_station[1], color="red", s=500, marker="^", label="Base Station")
        plt.scatter(nodes[:, 0], nodes[:, 1], color="green", s=40, marker="o", label="IoT Nodes")

        for ii in range(nodes.shape[0]):
            plt.text(nodes[ii][0] + 2, nodes[ii][1] + 3, f"IOT{ii + 1}",
                     bbox=dict(fill=False, edgecolor='green', linewidth=1))

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.axis("off")
        plt.legend()
        plt.pause(2)

        for i in range(nodes.shape[0]):
            positions_list = np.linspace(nodes[i], base_station, 50)
            green_dots = []

            for pos in positions_list:
                dot = plt.scatter(pos[0], pos[1], color="green", s=2, marker="o")
                green_dots.append(dot)
                plt.draw()
                plt.pause(0.01)

            plt.pause(0.5)

            # Simulate "data transmission" (here we just print it)
            print(f"\nNode {i + 1} transmitting data to base station:")
            print(node_data[i])  # or log to file / process as needed

            plt.pause(0.5)

            # Remove movement dots
            for dot in green_dots:
                dot.remove()

        plt.pause(3)
        plt.close()


def features_relation(DB,temp_features,d1,d2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd


    df = pd.DataFrame(temp_features)
    selected_features = [d1, d2]

    # Create pairplot
    g = sns.pairplot(df[selected_features], diag_kind='kde')

    # Add centralized title
    g.fig.suptitle(f"Relationship between {d1} and {d2} in UNSW-15 dataset", y=1.02, fontsize=14)

    # Show the plot
    plt.show()

    g.savefig(f"ImageResults/Relationships_between_features/{DB}/{d1}_{d2}.png", dpi=600)

    print(f"{d1}_{d2}.png saved succesfully")


def features_relation2(DB,temp_features,d1,d2,d3,d4,d5):
    import seaborn as sns
    df = pd.DataFrame(temp_features)
    selected_features = [d1, d2 ,d3,d4,d5]

    # Create pairplot
    g = sns.pairplot(df[selected_features], diag_kind='kde')

    # Add centralized title
    g.fig.suptitle(f"Relationship between {d1} {d2} {d3} {d4} and {d5} in UNSW-15 dataset", y=1.02, fontsize=14)

    # Show the plot
    plt.show()

    g.savefig(f"ImageResults/Relationships_between_features/{DB}/{d1}_{d2}_{d3}_{d4}_{d5}.png", dpi=600)

    print(f"{d1}_{d2}_{d3}_{d4}_{d5}.png saved successfully")

def correlation_map(df,DB):
    # Correlation heatmap
    import seaborn as sns
    plt.figure(figsize=(16, 12))
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title(f'Feature Correlation Heatmap for {DB} dataset features')
    plt.savefig(f"ImageResults/Correlation_map/{DB}_corr.png",dpi=600)
    plt.show()

    # if DB=="CICIDS2015":
    #     plt.figure(figsize=(10, 5))
    #     sns.histplot(data=df, x=df['Fwd Packet Length Mean'], kde=True, element="step")
    #     plt.title('Distribution of Fwd Packet Length Mean by Class')
    #     plt.savefig(f"ImageResults/histplot/{DB}_hist.png", dpi=600)
    #     plt.show()
    #
    #     # Scatter plot for Packet Length vs Flow Duration
    #     plt.figure(figsize=(10, 6))
    #     sns.scatterplot(data=df, x='Flow Duration', y='Total Length of Fwd Packets', alpha=0.5)
    #     plt.title('Flow Duration vs Fwd Packet Length by Label')
    #     plt.savefig(f"ImageResults/scatter_plot/{DB}_scatter.png", dpi=600)
    #     plt.show()
    #
import os
import pandas as pd

def save_sample_csvs(temp_features, output_dir="Test_data/UNSW-NB15", num_samples=50):
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle and take `num_samples` rows
    sampled_df = temp_features.sample(n=num_samples, random_state=42).reset_index(drop=True)

    # Save each row as separate CSV
    for i in range(num_samples):
        row_df = sampled_df.iloc[i:i+1]  # single row as DataFrame
        row_df.to_csv(f"{output_dir}/sample{i+1}.csv", index=False)  # Save with column names

    print(f"✅ {num_samples} sample files saved in '{output_dir}/' folder.")

# Example usage after preprocessing:
# After temp_features is ready inside your Preprocessing() function:
# save_sample_csvs(temp_features)
