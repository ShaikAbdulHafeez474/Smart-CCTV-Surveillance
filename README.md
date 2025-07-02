 🎥 Smart CCTV Suspicious Activity Detector

A deep learning-powered real-time **CCTV surveillance system** that detects suspicious activity from both **uploaded videos** and **live webcam feeds**. Built using **PyTorch**, **OpenCV**, and **Streamlit**, the app employs a **CNN + LSTM** architecture to analyze temporal frame sequences.

---

 🚀 Features

* 📹 **Live webcam surveillance** (OpenCV-based frame capture)
* 📽️ **Video file analysis** (Upload and predict)
* 🚨 Alerts user with alarm sound when suspicious activity is detected
* 🔧 Real-time frame processing using PyTorch model
* ✨ Beautiful and interactive **Streamlit UI**

---

🧰 Tech Stack

| Category         | Technologies Used                       |
| ---------------- | --------------------------------------- |
| Frontend UI      | Streamlit                               |
| Video Processing | OpenCV                                  |
| Deep Learning    | PyTorch, TorchVision                    |
| Model Type       | CNN + LSTM Hybrid                       |
| Deployment       | Streamlit Cloud                         |
| Others           | NumPy, Pillow, Base64, HTML audio alert |

---

🔍 Model Overview

The system uses a **CNN** (for spatial feature extraction from individual frames) followed by an **LSTM** (to model temporal dependencies across frames).

* **Input**: Sequence of 32 frames resized to 224x224
* **Output**: Binary classification

  * `normal`
  * `suspicious`

---

👀 How It Works

1. **Upload Video**

   * Extracts 32 evenly spaced frames from uploaded clip
   * Preprocesses and sends through model
   * Displays prediction and triggers alert if suspicious

2. **Webcam Live Feed**

   * Captures 32 frames using your webcam
   * Performs same processing and inference in real-time

---

🚪 Folder Structure

```
smart-cctv-surveillance/
├── app.py                # Streamlit application
├── model.py              # CNN + LSTM model class
├── best_model.pth        # Trained PyTorch model
├── alarm.mp3             # Audio alarm file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

---

📚 Setup Instructions

1. **Clone the repo**

```bash
https://github.com/your-username/smart-cctv-surveillance.git
cd smart-cctv-surveillance
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run Streamlit App**

```bash
streamlit run app.py
```

> Note: Make sure `best_model.pth` is present in the root directory.

---
ScreenShot1 : 

![Screenshot 2025-06-27 223126](https://github.com/user-attachments/assets/839226e6-526e-48d8-b5b4-bd1ddff735d0)

ScreenShot2 : 
![Screenshot 2025-06-27 223439](https://github.com/user-attachments/assets/f01b6c70-62ee-4602-b703-96593744edf1)


🌐 Deployed App

🔗 [Click here to try it out live](https://smart-cctv-surveillance-app.streamlit.app/)

---

🚜 Future Enhancements

* ✨ Add object tracking using YOLO
* ⌚ Real-time multi-camera support
* ⚖️ Model performance dashboard
* 📡 Integrate cloud alert system (email, SMS)

---

 ✨ Credits

* Developed with ❤️ by Hafeez
* Powered by PyTorch + Streamlit

---

✉️ Feedback / Contact

If you have any suggestions, drop me a mail or raise an issue on GitHub.

---

**Protect. Detect. React.**
