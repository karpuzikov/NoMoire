"""Descreener: standalone halftone/moire remover. Run: python descreener_app.py"""
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from descreener_engine import PEAK_DETECTION_TILE_SIZE, descreenImage

STYLESHEET = """
* { font-family: Inter, 'Segoe UI', sans-serif; font-size: 13px; color: #e6e8eb; }
QMainWindow, QWidget#panel { background: #0e0f11; }
QListWidget { background: #17191c; border: 1px solid #26292e; border-radius: 8px; padding: 8px; }
QListWidget::item { padding: 6px; }
QListWidget::item:selected { background: #b8f23a; color: #0e0f11; border-radius: 4px; }
QPushButton { background: transparent; border: 1px solid #3a3e45; border-radius: 8px; padding: 8px 16px; }
QPushButton:hover { border-color: #b8f23a; }
QPushButton#primary { background: #b8f23a; border: 1px solid #b8f23a; color: #0e0f11; font-weight: 600; }
QPushButton#primary:disabled { background: #26292e; border-color: #26292e; color: #6b7078; }
QLabel#preview { background: #17191c; border: 1px solid #26292e; border-radius: 8px; padding: 8px; }
QLabel#caption, QLabel#status { color: #8a8f98; font-family: 'JetBrains Mono', Consolas, monospace; }
QSlider::groove:horizontal { height: 4px; background: #26292e; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #b8f23a; border-radius: 2px; }
QSlider::handle:horizontal { background: #b8f23a; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
"""

SUPPORTED_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
DEBOUNCE_MILLISECONDS = 250


def convertPilImageToPixmap(image):
    rgbImage = image.convert("RGB")
    qImage = QImage(rgbImage.tobytes(), rgbImage.width, rgbImage.height, rgbImage.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qImage)


class SaveAllWorker(QThread):
    progressed = Signal(int, int)

    def __init__(self, filePaths, sensitivity, sharpenAmount):
        super().__init__()
        self.filePaths, self.sensitivity, self.sharpenAmount = filePaths, sensitivity, sharpenAmount

    def run(self):
        for index, filePath in enumerate(self.filePaths, 1):
            result = descreenImage(Image.open(filePath), self.sensitivity, self.sharpenAmount)
            result.save(filePath.with_name(f"{filePath.stem}_descreened{filePath.suffix}"), quality=95)
            self.progressed.emit(index, len(self.filePaths))


class DescreenerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Descreener")
        self.resize(1280, 760)
        self.filePaths = []
        self.saveAllWorker = None
        self.debounceTimer = QTimer(singleShot=True, interval=DEBOUNCE_MILLISECONDS)
        self.debounceTimer.timeout.connect(self.refreshPreview)

        self.fileList = QListWidget()
        self.fileList.currentRowChanged.connect(lambda _: self.debounceTimer.start())
        self.sensitivitySlider = self.buildSlider(1, 10, 6)
        self.sharpenSlider = self.buildSlider(0, 100, 30)
        self.openButton = QPushButton("Open images")
        self.openButton.clicked.connect(self.openImages)
        self.saveAllButton = QPushButton("Save all")
        self.saveAllButton.setObjectName("primary")
        self.saveAllButton.setEnabled(False)
        self.saveAllButton.clicked.connect(self.saveAll)
        self.statusLabel = QLabel("Saves next to originals as *_descreened")
        self.statusLabel.setObjectName("status")
        self.beforeLabel, self.afterLabel = self.buildPreviewLabel(), self.buildPreviewLabel()

        sidebar = QVBoxLayout()
        sidebar.setSpacing(12)
        sidebar.addWidget(self.openButton)
        sidebar.addWidget(self.fileList, 1)
        for caption, slider in (("Sensitivity", self.sensitivitySlider), ("Edge sharpen", self.sharpenSlider)):
            sidebar.addWidget(self.buildCaption(caption))
            sidebar.addWidget(slider)
        sidebar.addWidget(self.saveAllButton)
        sidebar.addWidget(self.statusLabel)
        sidebarWidget = QWidget(objectName="panel")
        sidebarWidget.setLayout(sidebar)
        sidebarWidget.setFixedWidth(280)

        previews = QHBoxLayout()
        for caption, label in (("BEFORE (100%)", self.beforeLabel), ("AFTER (100%)", self.afterLabel)):
            column = QVBoxLayout()
            column.addWidget(self.buildCaption(caption))
            column.addWidget(label, 1)
            previews.addLayout(column)
        root = QHBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(sidebarWidget)
        root.addLayout(previews, 1)
        central = QWidget(objectName="panel")
        central.setLayout(root)
        self.setCentralWidget(central)

    def buildSlider(self, minimum, maximum, value):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda _: self.debounceTimer.start())
        return slider

    def buildPreviewLabel(self):
        label = QLabel(objectName="preview", alignment=Qt.AlignCenter)
        label.setMinimumSize(300, 300)
        return label

    def buildCaption(self, text):
        return QLabel(text, objectName="caption")

    def openImages(self):
        selectedPaths, _ = QFileDialog.getOpenFileNames(self, "Open images", "", SUPPORTED_FILE_FILTER)
        if not selectedPaths:
            return
        self.filePaths = [Path(p) for p in selectedPaths]
        self.fileList.clear()
        self.fileList.addItems([p.name for p in self.filePaths])
        self.saveAllButton.setEnabled(True)
        self.fileList.setCurrentRow(0)

    def refreshPreview(self):
        row = self.fileList.currentRow()
        if row < 0:
            return
        fullImage = Image.open(self.filePaths[row]).convert("RGB")
        left = (fullImage.width - min(fullImage.width, PEAK_DETECTION_TILE_SIZE)) // 2
        top = (fullImage.height - min(fullImage.height, PEAK_DETECTION_TILE_SIZE)) // 2
        centerCrop = fullImage.crop((left, top, left + min(fullImage.width, PEAK_DETECTION_TILE_SIZE),
                                     top + min(fullImage.height, PEAK_DETECTION_TILE_SIZE)))
        result = descreenImage(centerCrop, self.sensitivitySlider.value(), self.sharpenSlider.value() / 100)
        for label, image in ((self.beforeLabel, centerCrop), (self.afterLabel, result)):
            label.setPixmap(convertPilImageToPixmap(image))

    def saveAll(self):
        self.saveAllButton.setEnabled(False)
        self.saveAllWorker = SaveAllWorker(self.filePaths, self.sensitivitySlider.value(), self.sharpenSlider.value() / 100)
        self.saveAllWorker.progressed.connect(lambda done, total: self.statusLabel.setText(f"Saved {done}/{total}"))
        self.saveAllWorker.finished.connect(lambda: self.saveAllButton.setEnabled(True))
        self.saveAllWorker.start()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLESHEET)
    window = DescreenerWindow()
    window.show()
    sys.exit(application.exec())
