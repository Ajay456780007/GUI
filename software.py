import customtkinter as ctk
import tkinter.filedialog as fd
import tkinter.messagebox as msg
import cv2
import numpy as np
import os
from threading import Thread
from PIL import Image
import pathlib
from datetime import datetime
import sys
import tensorflow as tf

# Import full preprocessing Extractor
from Feature_Extraction.Resnet18 import extract_feature_vector_Resnet_18
from Feature_Extraction.Viola_jones import detect_and_crop_face
from Feature_Extraction.Weighted_SIFT import Weighted_SIFT_1
from Feature_Extraction.deep3D import deep_3D_HOG

# Load TFLite model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proposed_model.tflite")
if not os.path.exists(model_path):
    msg.showerror("Model Error", f"Model file not found at: {model_path}")
    sys.exit(1)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Haarcascade setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_enhanced(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
    faces = face_cascade.detectMultiScale(upscaled, scaleFactor=1.1, minNeighbors=5)
    adjusted_faces = [(x//2, y//2, w//2, h//2) for (x, y, w, h) in faces]
    return adjusted_faces

def resize_to_target(image, size=(227, 227)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def predict_image_tflite(face_crop):
    try:
        cropped = detect_and_crop_face(face_crop)
        cropped = resize_to_target(cropped)
        image_pil = Image.fromarray(cropped)

        # Extract features
        deep3d = deep_3D_HOG(cropped)
        sift = Weighted_SIFT_1(cropped)
        resnet = extract_feature_vector_Resnet_18(image_pil)

        # Resize and convert to RGB
        deep3d = resize_to_target(cv2.cvtColor(deep3d, cv2.COLOR_BGR2RGB))
        sift = resize_to_target(cv2.cvtColor(sift, cv2.COLOR_BGR2RGB))
        resnet = resize_to_target(cv2.cvtColor(resnet, cv2.COLOR_BGR2RGB))

        # Convert to shape (H, W, C)
        combined = np.concatenate([deep3d, sift, resnet], axis=2)
        combined = resize_to_target(combined, (227, 227))

        # Final shape (1, 227, 227, 9) if needed
        if input_details[0]['shape'][1:] == [227, 227, 9]:
            combined = np.expand_dims(combined, axis=0).astype(np.float32)
        else:
            # Transpose to match TFLite channel-first model (1, 9, 227, 227)
            combined = np.transpose(combined, (2, 0, 1))
            combined = combined[np.newaxis, :, :, :].astype(np.float32)

        if combined.shape != tuple(input_details[0]['shape']):
            print(f"[Shape Warning] Got {combined.shape}, expected {input_details[0]['shape']}")
            return -1, 0.0

        interpreter.set_tensor(input_details[0]['index'], combined)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return 1 if output_data[0][0] > 0.5 else 0, output_data[0][0]
    except Exception as e:
        print("[Prediction Error]", e)
        return -1, 0.0

class DrowsinessApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.app = ctk.CTk()
        self.app.title("Drowsiness Detection System")
        self.app.geometry("800x600")

        self.tab_view = ctk.CTkTabview(self.app, width=780, height=580)
        self.tab_view.pack(padx=10, pady=10)

        self.tab_home = self.tab_view.add("Home")
        self.tab_live = self.tab_view.add("Live Detection")
        self.tab_test = self.tab_view.add("Image Test")
        self.tab_about = self.tab_view.add("About")

        self.setup_home_tab()
        self.setup_live_tab()
        self.setup_image_test_tab()
        self.setup_about_tab()

    def setup_home_tab(self):
        ctk.CTkLabel(self.tab_home, text="Welcome to Drowsiness Detection System", font=("Arial", 22)).pack(pady=40)
        ctk.CTkLabel(self.tab_home, text="Navigate through tabs to use features", font=("Arial", 16)).pack(pady=10)

    def setup_live_tab(self):
        self.start_button = ctk.CTkButton(self.tab_live, text="Start Detection", command=self.start_detection_thread)
        self.start_button.pack(pady=20)
        self.stop_info_label = ctk.CTkLabel(self.tab_live, text="Press 'Q' to stop detection", font=("Arial", 14), text_color="gray")
        self.stop_info_label.pack(pady=5)

    def setup_image_test_tab(self):
        self.test_label = ctk.CTkLabel(self.tab_test, text="Upload an Image to Test Drowsiness", font=("Arial", 18))
        self.test_label.pack(pady=10)

        self.upload_button = ctk.CTkButton(self.tab_test, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.image_label = ctk.CTkLabel(self.tab_test, text="")
        self.image_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.tab_test, text="", font=("Arial", 16))
        self.status_label.pack(pady=10)

        self.download_button = ctk.CTkButton(self.tab_test, text="Download Image", command=self.download_image)
        self.download_button.pack(pady=10)
        self.download_button.pack_forget()

    def setup_about_tab(self):
        about_text = (
            "This application detects driver drowsiness using:\n"
            "- Deep Learning Model (TFLite)\n"
            "- Face Detection (Haarcascade)\n"
            "- Feature Extractors (ResNet18, SIFT, HOG3D)\n\n"
            "Developed by: GUJAR"
        )
        ctk.CTkLabel(self.tab_about, text=about_text, font=("Arial", 14), justify="left").pack(pady=30)

    def start_detection_thread(self):
        thread = Thread(target=self.detect_drowsiness)
        thread.start()

    def detect_drowsiness(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces_enhanced(frame)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                pred_class, pred_score = predict_image_tflite(face_crop)

                label = "Not Drowsy" if pred_class == 1 else "Drowsy"
                color = (0, 255, 0) if pred_class == 1 else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                break

            cv2.imshow("Live Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def upload_image(self):
        file_path = fd.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            self.status_label.configure(text="No image selected.", text_color="orange")
            return

        frame = cv2.imread(file_path)
        if frame is None:
            self.status_label.configure(text="Failed to load image.", text_color="red")
            return

        faces = detect_faces_enhanced(frame)

        pred_class = -1
        label = "Unknown"

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            pred_class, pred_score = predict_image_tflite(face_crop)

            label = "Not Drowsy" if pred_class == 1 else "Drowsy"
            color = (0, 255, 0) if pred_class == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            break

        if pred_class != -1:
            self.status_label.configure(text=f"Prediction: {label} ({pred_score:.2f})", text_color="red" if pred_class == 0 else "green")
        else:
            self.status_label.configure(text="No face detected.", text_color="orange")

        self.processed_frame = frame
        self.download_button.pack()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        ctk_img = ctk.CTkImage(light_image=img_pil, size=(600, 400))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    def download_image(self):
        downloads_path = str(pathlib.Path.home() / "Downloads")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drowsiness_result_{timestamp}.jpg"
        save_path = os.path.join(downloads_path, filename)

        try:
            cv2.imwrite(save_path, self.processed_frame)
            msg.showinfo("Success", f"Image saved to: {save_path}")
        except Exception as e:
            msg.showerror("Error", f"Failed to save image: {str(e)}")

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    DrowsinessApp().run()
