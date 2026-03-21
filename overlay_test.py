import sys
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from PyQt5 import QtWidgets, QtGui, QtCore

# -----------------------------
# LOAD FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# SAME PREPROCESSING AS TRAINING
# -----------------------------
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def crop_face(img):
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img_np[y:y+h, x:x+w]
    else:
        face = img_np

    face = cv2.resize(face, (128, 128))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face

# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("efficientnet_deepfake_best.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ["REAL", "FAKE"]  # adjust if reversed

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(qt_pixmap):
    image = qt_pixmap.toImage()
    width = image.width()
    height = image.height()

    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)

    rgb = arr[:, :, :3]

    face = crop_face(rgb)
    tensor = val_transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()

# -----------------------------
# OVERLAY UI
# -----------------------------
class ScreenSelector(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None

        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setWindowState(QtCore.Qt.WindowFullScreen)
        self.setWindowOpacity(0.3)
        self.setStyleSheet("background-color: black;")

    def mousePressEvent(self, event):
        self.start = event.pos()
        self.end = self.start
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.end = event.pos()
        self.process_selection()
        self.close()

    def paintEvent(self, event):
        if self.start and self.end:
            painter = QtGui.QPainter(self)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0), 2))
            rect = QtCore.QRect(self.start, self.end)
            painter.drawRect(rect)

    def process_selection(self):
        x1 = min(self.start.x(), self.end.x())
        y1 = min(self.start.y(), self.end.y())
        x2 = max(self.start.x(), self.end.x())
        y2 = max(self.start.y(), self.end.y())

        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(0, x1, y1, x2-x1, y2-y1)

        label, conf = predict_image(screenshot)

        self.show_result(label, conf)

    def show_result(self, label, conf):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Deepfake Result")
        msg.setText(f"{label} ({conf*100:.2f}%)")
        msg.exec_()

# -----------------------------
# MAIN APP
# -----------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    selector = ScreenSelector()
    selector.show()

    sys.exit(app.exec_())