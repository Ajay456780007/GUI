import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from tkinter import ttk
from keras.models import load_model
import numpy as np
import os

# Feature definitions (for input shape consistency)
CICIDS2015_FEATURES = ["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10"]
NBAIOT_FEATURES = ["sensor1", "sensor2", "sensor3", "sensor4", "sensor5", "sensor6", "sensor7", "sensor8", "sensor9", "sensor10"]
UNSWNB15_FEATURES  = [
    'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
       'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts',
       'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
       'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime',
       'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
       'ct_state_ttl', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
       'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
       'ct_dst_src_ltm'
]
MODEL_PATHS = {
    'CICIDS2015': 'Saved_model/CICIDS2015_model.h5',
    'N-BaIoT': 'Saved_model/N-BaIoT_model.h5',
    'UNSW-NB15': 'Saved_model/UNSW-NB15_model.keras'
}


class AttackPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚨 Attack Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f4f8")

        self.model = None
        self.selected_features = []
        self.data_row = None
        self.feature_names = []

        title_font = ("Helvetica", 20, "bold")
        label_font = ("Arial", 12)
        button_font = ("Arial", 10, "bold")

        tk.Label(root, text="Intrusion Detection System", font=title_font, bg="#f0f4f8", fg="#0d47a1").pack(pady=10)

        dataset_frame = tk.Frame(root, bg="#e8f0fe", bd=2, relief="ridge")
        dataset_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(dataset_frame, text="Select Dataset (Load Model):", font=label_font, bg="#e8f0fe").pack(pady=5)

        btn_frame = tk.Frame(dataset_frame, bg="#e8f0fe")
        btn_frame.pack()

        for dataset in MODEL_PATHS.keys():
            ttk.Button(btn_frame, text=dataset, command=lambda d=dataset: self.load_dataset(d)).pack(side="left", padx=10, pady=5)

        # Instruction label
        self.info_label = tk.Label(root, text="After loading the model, choose a CSV sample from Test_data (sample1.csv, sample2.csv...)",
                                   font=label_font, bg="#f0f4f8", wraplength=700, justify="center", fg="#333")
        self.info_label.pack(pady=20)

        # Action buttons
        button_frame = tk.Frame(root, bg="#f0f4f8")
        button_frame.pack(pady=20)

        self.select_data_button = tk.Button(button_frame, text="📂 Select Sample File", font=button_font,
                                            command=self.select_data, state='disabled', bg="#64b5f6", fg="white", width=20)
        self.select_data_button.pack(side="left", padx=20)

        self.predict_button = tk.Button(button_frame, text="🔍 Predict", font=button_font,
                                        command=self.predict, state='disabled', bg="#43a047", fg="white", width=20)
        self.predict_button.pack(side="left", padx=20)

        # Prediction result label
        self.result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), bg="#f0f4f8")
        self.result_label.pack(pady=40)

    def load_dataset(self, dataset_name):
        try:
            model_path = MODEL_PATHS[dataset_name]
            self.model = load_model(model_path, compile=False)

            if dataset_name == 'CICIDS2015':
                self.feature_names = CICIDS2015_FEATURES
            elif dataset_name == 'N-BaIoT':
                self.feature_names = NBAIOT_FEATURES
            elif dataset_name == 'UNSW-NB15':
                self.feature_names = UNSWNB15_FEATURES

            self.select_data_button.config(state='normal')
            self.result_label.config(text="✅ Model loaded. Now select a test sample.", fg="#0d47a1")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model for {dataset_name}\n{e}")

    def select_data(self):
        file_path = filedialog.askopenfilename(initialdir="Test_data", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)

            # Ensure selected file has required columns
            missing = [feat for feat in self.feature_names if feat not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            self.data_row = df[self.feature_names].iloc[0].values.reshape(1, -1)
            self.predict_button.config(state='normal')
            self.result_label.config(text="✅ Sample file loaded. Ready to predict.", fg="#1565c0")

        except Exception as e:
            messagebox.showerror("Sample File Error", f"Error reading sample file:\n{e}")

    def predict(self):
        if self.model is None or self.data_row is None:
            messagebox.showwarning("Missing Info", "Please load a dataset and select data.")
            return
        try:
            # Ensure input shape is (1, 44, 1)
            input_data = self.data_row
            # if input_data.shape[1] == 44:
            input_data = input_data.reshape((1, input_data.shape[1], 1))  # reshape to (1, 44, 1)
            # elif input_data.shape[1] == 43:
            #     messagebox.showerror("Feature Error", "Model expects 44 features, but 43 were given.")
            #     return
            # else:
            #     messagebox.showerror("Shape Error", f"Incompatible input shape: {input_data.shape}")
            #     return

            prediction = self.model.predict(input_data)[0]

            # Classify the prediction
            if isinstance(prediction, np.ndarray) and len(prediction) > 1:
                pred_class = np.argmax(prediction)
            else:
                pred_class = int(round(float(prediction)))

            if pred_class in [1, 'attack', 'malicious', 'DoS', 'Botnet']:
                self.result_label.config(text="🚨 Attack Data Detected!", fg="red")
            else:
                self.result_label.config(text="✅ Normal Data", fg="green")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


# --- Run the App ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AttackPredictorGUI(root)
    root.mainloop()
