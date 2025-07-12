
# 🚀 Player Re-Identification & Tracking in Sports Videos

### **AI Internship Induction Task Submission**

This project presents a robust solution for **player re-identification and tracking in sports footage**, combining state-of-the-art computer vision techniques: **YOLOv8** for detection, **OSNet** for appearance-based feature extraction, and **Deep SORT** for consistent multi-object tracking.

---

## 📁 Project Structure

```

Player-ReID-Tracking/
├── README.md
├── report.md or report.pdf
├── requirements.txt
├── src/
│   ├── tracker.py                  \# Main tracking logic
│   ├── playerDetection.py          \# YOLOv8 detection module
│   ├── embeddingExtractor.py       \# Appearance feature extractor (OSNet)
│   ├── utils.py                    \# Helper functions
├── models/
│   └── best.pt                     \# Custom YOLOv8 model
├── output/
│   └── yolo\_output\_with\_embeddings.mp4
├── assets/
│   └── 15sec\_input\_720p.mp4

````

---

## ⚙️ Setup & Run Instructions

**Python 3.8+ required.**

1.  **Clone the repository and navigate into the directory.**
2.  **(Optional) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Linux/macOS
    source venv/bin/activate
    # Windows
    venv\Scripts\activate.bat
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the tracker script:**
    ```bash
    python src/tracker.py
    ```
    * **Input:** `assets/15sec_input_720p.mp4`
    * **Output:** `output/yolo_output_with_embeddings.mp4`

---

## 🧠 Core Components & Functionality

| Module                | Purpose                                                  |
| :--------------------- | :--------------------------------------------------------- |
| **YOLOv8**            | Detects players, ball, and referees.                     |
| **OSNet**             | Extracts robust player **appearance embeddings**.          |
| **Deep SORT**         | Assigns consistent IDs by integrating **motion and appearance** cues. |
| **Kalman Filter**     | **Smooths trajectories** for each tracked player.          |
| **Cosine Similarity** | Utilized for **re-identification** and feature matching.    |

---

## ✨ Project Highlights

* Achieves **high consistency** in player tracking across frames.
* Integrates **appearance-based re-identification** using OSNet embeddings.
* **Real-time capable logic** adapted for offline video processing.
* Output video visualizes **player ID, label, and bounding boxes** for clear analysis.

---

## ⚠️ Limitations
IDs may switch occasionally if:

Players overlap or occlude each other

Appearance changes due to motion blur

No temporal interpolation (missing detections = missing tracks)

Only appearance-based re-ID, no jersey number recognition yet

Model performs best when players are of decent resolution

##📈 Potential Improvements

Add jersey number OCR for stronger ID persistence

Use temporal smoothing or optical flow

Improve re-ID robustness with augmentation or larger embedding queue

Adapt for real-time processing with webcam or livestream


## 👤 Author

** Satvik Sharma**  
BITS Pilani, Hyderabad Campus  
AI Intern Project- July 2025  
GitHub: https://github.com/satvik-18
````