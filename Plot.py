import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from termcolor import colored
from Sub_Functions.Concat_epochs import Concat_epochs


class ALL_GRAPH_PLOT:
    def __init__(self,Comp_Analysis=None,Perf_Analysis=None,KF_Analysis=None,Roc_Analysis=None,Show=True,Save=True):

        self.Comp_Analysis=Comp_Analysis
        self.Perf_Analysis=Perf_Analysis
        self.KF_Analysis=KF_Analysis
        self.ROC_Analysis=Roc_Analysis
        self.bar_width=0.1
        self.color = ["#1f77b4",  "#ff7f0e",  "#2ca02c",  "#d62728",  "#9467bd",  "#8c564b",  "#e377c2",  "#7f7f7f"]
        self.models=["BBO_CFAT","CNN","CNN_LSTM","ELAI","PM_WA","RNN","SAE_CNN","proposed_RES_EN_DE_CNN"]
        self.percentage=["TP_40","TP_60","TP_60","TP_70","TP_80","TP_90"]
        self.save=Save
        self.show=Show
        self.models2 = [ "RES_EN-DE_CNN at Epochs=100","RES_EN-DE_CNN at Epochs=200","RES_EN-DE_CNN at Epochs=300","RES_EN-DE_CNNat Epochs=400","RES_EN-DE_CNN at Epochs=500"]

    def Load_Comp_data(self,DB):
        Concat_epochs(DB)
        self.perf_concat(DB)

        A=np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/ACC_1.npy")
        B=np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/SEN_1.npy")
        C=np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/SPE_1.npy")
        D=np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/PRE_1.npy")
        E=np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/REC_1.npy")
        F=np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/F1score_1.npy")

        return A,B,C,D,E,F

    def Comp_figure(self,DB,perf,x_label,y_label,colors,bar_width):
        n_models=perf.shape[0]
        n_percentage=perf.shape[1]
        Model=self.models
        Models=Model[:n_models]
        percntage=[40,50,60,70,80,90]
        Percentages=percntage[:n_percentage]
        x=np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Comp_Analysis Bar Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12,8))
        for i in range(perf.shape[0]):
            plt.bar(x+i*bar_width ,perf[i],width=bar_width,label=Models[i],color=colors[i],alpha=1)
        center_shift = (perf.shape[0] * bar_width) / 2 - bar_width / 2
        plt.xticks(x + center_shift, Percentages)
        plt.xlabel(x_label,weight="bold", fontsize="15")
        plt.ylabel(y_label,weight="bold", fontsize="15")

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="upper left", prop=legend_properties)

        if self.save:
            os.makedirs(f"Results/{DB}/Comparative_Analysis/Bar",exist_ok=True)
            df.to_csv(f"Results/{DB}/Comparative_Analysis/Bar/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Comparative_Analysis/Bar/{y_label}_Graph.png", dpi=600)
            print(colored(f'Comp_Analysis Bar Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def Comp_figure_Line(self,DB,perf,x_label,y_label,colors):
        n_models = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models
        Models = Model[:n_models]
        percentage = [40, 50, 60, 70, 80, 90]
        Percentages = percentage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Comp_Analysis Line Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12,8))
        for i in range(perf.shape[0]):
            plt.plot(Percentages,perf[i],marker='o',linestyle="-",alpha=1,label=Models[i],color=colors[i])

        plt.xlabel(x_label, weight="bold", fontsize="15")
        plt.ylabel(y_label, weight="bold", fontsize="15")

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="upper left", prop=legend_properties)

        if self.save:
            os.makedirs(f"Results/{DB}/Comparative_Analysis/Line", exist_ok=True)
            df.to_csv(f"Results/{DB}/Comparative_Analysis/Line/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Comparative_Analysis/Line/{y_label}_Graph.png", dpi=600)
            print(colored(f'Comp_Analysis Line Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def plot_comp_figure(self,DB):
        perf=self.Load_Comp_data(DB)

        x_label="Training_percentage(%)"

        y_label = "Precision (%)"
        Perf_2 = perf[3]
        self.Comp_figure(DB,Perf_2, x_label, y_label,self.color,self.bar_width)
        self.Comp_figure_Line(DB,Perf_2, x_label, y_label, self.color)

        y_label = "Recall (%)"
        Perf_3 = perf[4]
        self.Comp_figure(DB, Perf_3, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_3, x_label, y_label, self.color)

        y_label = "Accuracy (%)"
        Perf_1 = perf[0]
        self.Comp_figure(DB, Perf_1, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_1, x_label, y_label, self.color)

        y_label = "F1 Score (%)"
        Perf_4 = perf[5]
        self.Comp_figure(DB, Perf_4, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_4, x_label, y_label, self.color)

        # Sensitivity
        y_label = "Sensitivity (%)"
        Perf_5 = perf[1]
        self.Comp_figure(DB, Perf_5, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_5, x_label, y_label, self.color)

        # Specificity
        y_label= "Specificity (%)"
        Perf_6 = perf[2]
        self.Comp_figure(DB, Perf_6, x_label, y_label, self.color, self.bar_width)
        self.Comp_figure_Line(DB, Perf_6, x_label, y_label, self.color)

    def perf_concat(self,DB):

        base_path = os.path.join(os.getcwd(), "Analysis", "Comparative_Analysis", DB)
        epoch_path = os.path.join(os.getcwd(), "Analysis", "Performance_Analysis", "Concated_epochs", DB,
                                  "metrics_epochs_500.npy")
        epoch_data = np.load(epoch_path)
        metric_files = ["ACC_1.npy", "SEN_1.npy", "SPE_1.npy", "F1score_1.npy", "REC_1.npy", "PRE_1.npy"]
        for i, file in enumerate(metric_files):
            metric_path = os.path.join(base_path, file)
            metric_data = np.load(metric_path)
            metric_data[-1] = epoch_data[i]
            np.save(metric_path, metric_data)



    def load_perf_values(self,DB):
        Concat_epochs(DB)
        A11 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_100.npy")
        A22 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_200.npy")
        A33 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_300.npy")
        A44 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_400.npy")
        A55 = np.load(f"Analysis/Performance_Analysis/Concated_epochs/{DB}/metrics_epochs_500.npy")

        A1 = np.stack([A11[0], A22[0], A33[0], A44[0], A55[0]], axis=0)
        A2 = np.stack([A11[1], A22[1], A33[1], A44[1], A55[1]], axis=0)
        A3 = np.stack([A11[2], A22[2], A33[2], A44[2], A55[2]], axis=0)
        A4 = np.stack([A11[3], A22[3], A33[3], A44[3], A55[3]], axis=0)
        A5 = np.stack([A11[4], A22[4], A33[4], A44[4], A55[4]], axis=0)
        A6 = np.stack([A11[5], A22[5], A33[5], A44[5], A55[5]], axis=0)

        return A1,A2,A3,A4,A5,A6

    def perf_figure(self,DB,perf,x_label,y_label,colors,bar_width):
        n_epochs = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models2
        Models = Model[:n_epochs]
        percentage = [40, 50, 60, 70, 80, 90]
        Percentages = percentage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Perf_Analysis Bar Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12, 6))
        for i in range(perf.shape[0]):
            plt.bar(x + i * bar_width, perf[i], width=bar_width, label=Models[i], color=colors[i], alpha=1)
        center_shift = (perf.shape[0] * bar_width) / 2 - bar_width / 2
        plt.xticks(x + center_shift, Percentages)
        plt.xlabel(x_label, weight="bold", fontsize="15")
        plt.ylabel(y_label, weight="bold", fontsize="15")

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="upper left", prop=legend_properties)

        if self.save:
            os.makedirs(f"Results/{DB}/Performance_Analysis/Bar", exist_ok=True)
            df.to_csv(f"Results/{DB}/Performance_Analysis/Bar/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Performance_Analysis/Bar/{y_label}_Graph.png", dpi=600)
            print(colored(f'Perf_Analysis Bar Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def perf_figure_line(self,DB,perf,x_label,y_label,colors):
        n_epochs = perf.shape[0]
        n_percentage = perf.shape[1]
        Model = self.models2
        Models = Model[:n_epochs]
        percentage = [40, 50, 60, 70, 80, 90]
        Percentages = percentage[:n_percentage]
        x = np.arange(len(Percentages))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=Percentages)
        df.index = Models
        print(colored(f'Perf_Analysis Line Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12, 6))
        for i in range(perf.shape[0]):
            plt.plot(Percentages, perf[i], marker='o', linestyle="-", alpha=1, label=Models[i], color=colors[i])

        plt.xlabel(x_label, weight="bold", fontsize="15")
        plt.ylabel(y_label, weight="bold", fontsize="15")

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="upper left", prop=legend_properties)

        if self.save:
            os.makedirs(f"Results/{DB}/Performance_Analysis/Line", exist_ok=True)
            df.to_csv(f"Results/{DB}/Performance_Analysis/Line/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/Performance_Analysis/Line/{y_label}_Graph.png", dpi=600)
            print(colored(f'Perf_Analysis Line Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def plot_perf_figure(self,DB):
        data=self.load_perf_values(DB)

        x_label = "Training_percentage(%)"

        y_label = "Precision (%)"
        Perf_2 = data[5]
        self.perf_figure(DB, Perf_2, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_2, x_label, y_label, self.color)

        y_label = "Recall (%)"
        Perf_3 = data[4]
        self.perf_figure(DB, Perf_3, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_3, x_label, y_label, self.color)

        y_label = "Accuracy (%)"
        Perf_1 = data[0]
        self.perf_figure(DB, Perf_1, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_1, x_label, y_label, self.color)

        y_label = "F1 Score (%)"
        Perf_4 = data[3]
        self.perf_figure(DB, Perf_4, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_4, x_label, y_label, self.color)

        # Sensitivity
        y_label = "Sensitivity (%)"
        Perf_5 = data[1]
        self.perf_figure(DB, Perf_5, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_5, x_label, y_label, self.color)

        # Specificity
        y_label = "Specificity (%)"
        Perf_6 = data[2]
        self.perf_figure(DB, Perf_6, x_label, y_label, self.color, self.bar_width)
        self.perf_figure_line(DB, Perf_6, x_label, y_label, self.color)

    def load_kf_values(self,DB):
        A = np.load(f"Analysis/KF_Analysis/{DB}/ACC_2.npy")
        B = np.load(f"Analysis/KF_Analysis/{DB}/SEN_2.npy")
        C = np.load(f"Analysis/KF_Analysis/{DB}/SPE_2.npy")
        D = np.load(f"Analysis/KF_Analysis/{DB}/F1score_2.npy")
        E = np.load(f"Analysis/KF_Analysis/{DB}/REC_2.npy")
        F = np.load(f"Analysis/KF_Analysis/{DB}/PRE_2.npy")

        return A,B,C,D,E,F
    def kf_figure(self,DB,perf,x_label,y_label,colors,bar_width):
        n_models = perf.shape[0]
        n_folds = perf.shape[1]
        Model = self.models
        Models = Model[:n_models]
        fold = [6,7,8,9,10]
        folds = fold[:n_folds]
        x = np.arange(len(folds))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=folds)
        df.index = Models
        print(colored(f'KF_Analysis Bar Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12,8))
        for i in range(perf.shape[0]):
            plt.bar(x + i * bar_width, perf[i], width=bar_width, label=Models[i], color=colors[i], alpha=1)
        center_shift = (perf.shape[0] * bar_width) / 2 - bar_width / 2
        plt.xticks(x + center_shift, folds)
        plt.xlabel(x_label, weight="bold", fontsize="15")
        plt.ylabel(y_label, weight="bold", fontsize="15")

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="upper left", prop=legend_properties)

        if self.save:
            os.makedirs(f"Results/{DB}/KF_Analysis/Bar", exist_ok=True)
            df.to_csv(f"Results/{DB}/KF_Analysis/Bar/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/KF_Analysis/Bar/{y_label}_Graph.png", dpi=600)
            print(colored(f'KF_Analysis Bar Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

    def kf_figure_line(self,DB,perf,x_label,y_label,colors):
        n_models = perf.shape[0]
        n_folds = perf.shape[1]
        Model = self.models
        Models = Model[:n_models]
        percentage = [6,7,8,9,10]
        folds = percentage[:n_folds]
        x = np.arange(len(folds))
        sns.set(style="darkgrid")
        df = pd.DataFrame(perf, columns=folds)
        df.index = Models
        print(colored(f'KF_Analysis Line Graph values for {DB} of {y_label} saved as CSV', 'yellow'))
        plt.figure(figsize=(12,8))
        for i in range(perf.shape[0]):
            plt.plot(folds, perf[i], marker='o', linestyle="-", alpha=1, label=Models[i], color=colors[i])

        plt.xlabel(x_label, weight="bold", fontsize="15")
        plt.ylabel(y_label, weight="bold", fontsize="15")

        legend_properties = {'weight': 'bold', 'size': 12}
        plt.legend(loc="upper left", prop=legend_properties)

        if self.save:
            os.makedirs(f"Results/{DB}/KF_Analysis/Line", exist_ok=True)
            df.to_csv(f"Results/{DB}/KF_Analysis/Line/{y_label}__Graph.csv")
            plt.savefig(f"Results/{DB}/KF_Analysis/Line/{y_label}_Graph.png", dpi=600)
            print(colored(f'KF_Analysis Line Graph Image for {DB} of {y_label} saved', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def plot_kf_figure(self,DB):
        perf = self.load_kf_values(DB)

        x_label = "KFOLD"

        y_label = "Precision (%)"
        Perf_2 = perf[5]
        self.kf_figure(DB, Perf_2, x_label, y_label, self.color, self.bar_width)
        self.kf_figure_line(DB, Perf_2, x_label, y_label, self.color)

        y_label = "Recall (%)"
        Perf_3 = perf[4]
        self.kf_figure(DB, Perf_3, x_label, y_label, self.color, self.bar_width)
        self.kf_figure_line(DB, Perf_3, x_label, y_label, self.color)

        y_label = "Accuracy (%)"
        Perf_1 = perf[0]
        self.kf_figure(DB, Perf_1, x_label, y_label, self.color, self.bar_width)
        self.kf_figure_line(DB, Perf_1, x_label, y_label, self.color)

        y_label = "F1 Score (%)"
        Perf_4 = perf[5]
        self.kf_figure(DB, Perf_4, x_label, y_label, self.color, self.bar_width)
        self.kf_figure_line(DB, Perf_4, x_label, y_label, self.color)

        # Sensitivity
        y_label = "Sensitivity (%)"
        Perf_5 = perf[1]
        self.kf_figure(DB, Perf_5, x_label, y_label, self.color, self.bar_width)
        self.kf_figure_line(DB, Perf_5, x_label, y_label, self.color)

        # Specificity
        y_label = "Specificity (%)"
        Perf_6 = perf[2]
        self.kf_figure(DB, Perf_6, x_label, y_label, self.color, self.bar_width)
        self.kf_figure_line(DB, Perf_6, x_label, y_label, self.color)

        plt.clf()
        plt.close()

    def Load_Comparative_ROC_values(self,DB):

        TPR_ROC_1 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/TPR_1.npy")
        FPR_ROC_2 = np.load(f"{os.getcwd()}/Analysis/Comparative_Analysis/{DB}/FPR_1.npy")

        return [TPR_ROC_1 ,FPR_ROC_2]

    def plot_ROC_from_comparative_model(self, DB, perf):
        df = pd.DataFrame(perf)

        model_names = ["KNN", "CNN", "CNN_Resnet", "SVM", "DiT", "HGNN", "WA", "proposed_model"]
        labels = ["ROC Curve for KNN", "ROC Curve for CNN", "ROC Curve for CNN_Resnet", "ROC Curve for SVM",
                  "ROC Curve for DiT", "ROC Curve for HGNN", "ROC Curve for WA", "ROC Curve for Proposed model"]

        # Load TPR and FPR values from your function
        Com_1 = self.Load_Comparative_ROC_values(DB)
        True_positive_rate = Com_1[0]  # shape: (8, N)
        False_positive_rate = Com_1[1]  # shape: (8, N)

        xlab = "False Positive Rate"
        ylab = "True Positive Rate"

        # Start plotting
        plt.figure(figsize=(10, 8))

        for i in range(len(labels)):
            tpr = True_positive_rate[i]
            fpr = False_positive_rate[i]

            # Ensure both are numpy arrays for safety
            tpr = np.array(tpr)
            fpr = np.array(fpr)

            plt.plot(fpr, tpr, label=labels[i], linewidth=2)

        # Add diagonal reference line (random guess)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

        plt.xlabel(xlab, weight="bold", fontsize=15)
        plt.ylabel(ylab, weight="bold", fontsize=15)
        plt.title("Comparative ROC Curves", fontsize=16, weight="bold")
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(True)

        if self.save:
            os.makedirs(f"Analysis/ROC_Analysis/{DB}", exist_ok=True)
            df.to_csv(f"Analysis/ROC_Analysis/{DB}/comparative_roc_metrics.csv", index=False)
            plt.savefig(f"Analysis/ROC_Analysis/{DB}/comparative_roc_plot.png", dpi=600)
            print(colored(f'Comparative ROC Curve for {DB} saved as PNG and CSV.', 'green'))

        if self.show:
            plt.show()

        plt.clf()
        plt.close()

    def Load_Performance_ROC_values(self, DB):
        base_path = f"{os.getcwd()}/Analysis/Performance_Analysis/Concated_epochs/{DB}"

        # Load ROC metrics from all 5 epoch files
        epoch_files = ["metrics_epochs_100.npy", "metrics_epochs_200.npy",
                       "metrics_epochs_300.npy", "metrics_epochs_400.npy",
                       "metrics_epochs_500.npy"]

        TPR_list = []
        FPR_list = []

        for file in epoch_files:
            data = np.load(os.path.join(base_path, file))  # shape: (8, 6)

            # Get the last two rows: TPR (row -2) and FPR (row -1)
            TPR = data[-2]  # shape: (6,)
            FPR = data[-1]  # shape: (6,)

            TPR_list.append(TPR)
            FPR_list.append(FPR)

        return TPR_list, FPR_list

    def plot_ROC_from_proposed_model(self, DB,color):

        # Load TPR and FPR values from all 5 files
        TPR_list, FPR_list = self.Load_Performance_ROC_values(DB)
        epochs = [100, 200, 300, 400, 500]

        xlab = "False Positive Rate"
        ylab = "True Positive Rate"

        for i in range(5):  # For each epoch
            plt.figure(figsize=(8, 6))
            plt.plot(FPR_list[i], TPR_list[i], color=color[i], label=f"Epochs {epochs[i]}", linewidth=2)

            plt.xlabel(xlab, weight="bold", fontsize=14)
            plt.ylabel(ylab, weight="bold", fontsize=14)
            plt.title(f"ROC Curve - Epoch {epochs[i]}", weight="bold", fontsize=15)
            plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 11})
            plt.grid(True)

            out_dir = f"Analysis/ROC_Analysis/{DB}/Epoch_{epochs[i]}"
            os.makedirs(out_dir, exist_ok=True)
            print(colored(f'Saved ROC plot and CSV for {DB} for Epoch {epochs[i]} in {out_dir}', 'green'))

            if self.save:
                out_dir = f"Analysis/ROC_Analysis/{DB}/Epoch_{epochs[i]}"
                os.makedirs(out_dir, exist_ok=True)

                # Save CSV
                df = pd.DataFrame({'FPR': FPR_list[i], 'TPR': TPR_list[i]})
                df.to_csv(f"{out_dir}/ROC_Epoch_{epochs[i]}.csv", index=False)

                # Save PNG
                plt.savefig(f"{out_dir}/ROC_Epoch_{epochs[i]}.png", dpi=600)
                print(colored(f'Saved ROC plot and CSV forfor {DB} Epoch {epochs[i]} in {out_dir}', 'green'))

            if self.show:
                plt.show()

            plt.clf()
            plt.close()


    def plot_missing_values(self,a, b=None):
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Plot nulls before imputation
        plt.figure(figsize=(10, 5))
        sns.heatmap(a.isnull(), cbar=False)
        plt.title("Missing Values BEFORE KNN Imputation")
        plt.show()

        # Plot nulls after imputation
        plt.figure(figsize=(10, 5))
        sns.heatmap(b.isnull(), cbar=False)
        plt.title("Missing Values AFTER KNN Imputation")
        plt.show()

    def plot_ROC_comparative_figure(self,DB):
        perf=self.Load_Comparative_ROC_values(DB)

        if DB=="Zea_maya" or DB=="Solanum_pennellii":

            self.plot_ROC_from_comparative_model(DB,perf)

    def plot_ROC_performance_figure(self,DB):
        if DB=="Zea_mays" or DB=="Solanum_pennellii":

            self.plot_ROC_from_proposed_model(DB,self.color)


    def GRAPH_RESULT(self,DB):
        self.plot_comp_figure(DB)
        self.plot_perf_figure(DB)
        self.plot_kf_figure(DB)
        self.plot_ROC_performance_figure(DB)