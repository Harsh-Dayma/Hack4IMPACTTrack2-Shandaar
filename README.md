# Team-Shandaar
Members: Harsh Dayma(Lead)
         ,Abdullah Zahid
         ,Aditya Prasad
         
Domain : Cybersecurity & Ethical AI Systems

Problem Statement : DeepFake Detection Tool


**🧠 Deepfake Detection using ResNet-18 + PyQt5**



A desktop-based deepfake detection system that uses a deep learning model (ResNet-18) to classify images as REAL or FAKE. The application includes a GUI that allows users to select any region of the screen and instantly analyze it.



**🚀 Features**

📸Screen region selection for real-time analysis

🤖Deep learning-based classification (ResNet-18)

🧠Face detection using OpenCV Haar Cascades

⚡Fast inference with PyTorch

🖥️Lightweight PyQt5 GUI overlay



🏗️ Project Structure



├── dataset/                       #Datasetfolder(ImageFolderformat)

│   ├── REAL/

│   └── FAKE/

│

├── train.py                       # Model training script

├── app.py                        # GUI + inference script

├── efficientnet\_deepfake\_best.pth        # Saved model weights

├── README.md



**⚙️ Installation**

1\. Clone the repository

&#x20;  git clone https://github.com/your-username/deepfake-detector.git

&#x20;  cd deepfake-detector

2\. Create virtual environment

&#x20;  python3 -m venv venv

&#x20;  source venv/bin/activate   # Linux/Mac

&#x20;  venv\\Scripts\\activate      # Windows

3\. Install dependencies

&#x20;  pip install torch torchvision opencv-python PyQt5 numpy



**📊 Dataset**

Organized using ImageFolder format:

&#x20; dataset/

&#x20; ├── REAL/

&#x20; └── FAKE/

Images are resized and normalized before training.

Data augmentation applied:

&#x20;  Random Flip

&#x20;  Rotation

&#x20;  Color Jitter

&#x20;  Random Erasing



**🧠 Model Details**

Backbone: ResNet-18

Pretrained: ✅ (ImageNet weights)

Final Layer: Modified for binary classification

Loss Function: CrossEntropyLoss

Optimizer: Adam

Learning Rate Scheduler: StepLR



**🏋️Training**

Run:

python train.py

Trains for 25 epochs

Saves best model based on validation accuracy: efficientnet\_deepfake\_best.pth



**🖥️ Running the Application**

python app.py



**How it works:**

1.Full-screen overlay appears

2.Drag to select any region

3.The model analyzes the selected image

4.Result is displayed:

REAL (92.45%)



**🔍 Inference Pipeline**

5.Capture selected screen region

6.Detect face using Haar Cascade

7.Resize to 128×128

8.Normalize using ImageNet stats

9.Pass through ResNet-18

10.Output prediction with confidence



**⚠️ Known Limitations**

Uses Haar Cascade (not robust for difficult angles)

Only binary classification (REAL vs FAKE)

Performance depends heavily on dataset quality

No video-level temporal analysis



**🔮 Future Improvements**

Replace Haar Cascade with MTCNN or RetinaFace

Use EfficientNet or Vision Transformers

Add video deepfake detection

Deploy as web app (FastAPI + React)

Improve dataset diversity



**📌 Requirements**

Python 3.8+

PyTorch

OpenCV

PyQt5

NumPy



**🧾 License**

This project is open-source and available under the MIT License.



**👤 Team**

Shandaar

GitHub: https://github.com/Harsh-Dayma/Hack4IMPACTTrack2-Shandaar




         
