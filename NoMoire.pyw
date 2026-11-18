"""NoMoire: standalone halftone/moire remover.

Install:  pip install numpy scipy Pillow PySide6
Run:      double-click NoMoire.pyw
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import fft as scipyFft
from scipy.ndimage import maximum_filter, uniform_filter
import sys
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidget, QMainWindow,
                               QPushButton, QSlider, QVBoxLayout, QWidget)


PEAK_DETECTION_TILE_SIZE = 1024
MINIMUM_PEAK_FREQUENCY_CYCLES_PER_PIXEL = 0.04
MAXIMUM_PEAKS_PER_CHANNEL = 64
NOTCH_SIGMA_CYCLES_PER_PIXEL = 0.004
NOTCH_RADIUS_IN_SIGMAS = 4
BACKGROUND_FILTER_SIZE = 15


def cropCenterTile(channel):
    height, width = channel.shape
    tileHeight = min(height, PEAK_DETECTION_TILE_SIZE)
    tileWidth = min(width, PEAK_DETECTION_TILE_SIZE)
    top = (height - tileHeight) // 2
    left = (width - tileWidth) // 2
    return channel[top:top + tileHeight, left:left + tileWidth]


def findHalftonePeaks(channel, thresholdInRobustSigmas):
    """Returns [(verticalFrequency, horizontalFrequency)] in cycles per pixel."""
    tile = cropCenterTile(channel)
    tileHeight, tileWidth = tile.shape
    window = np.outer(np.hanning(tileHeight), np.hanning(tileWidth)).astype(np.float32)
    windowedTile = (tile - tile.mean()) * window
    logSpectrum = np.log1p(np.abs(np.fft.fftshift(scipyFft.fft2(windowedTile))))
    residual = logSpectrum - uniform_filter(logSpectrum, BACKGROUND_FILTER_SIZE)
    median = np.median(residual)
    robustSigma = 1.4826 * np.median(np.abs(residual - median)) + 1e-9
    isLocalMaximum = residual == maximum_filter(residual, size=5)
    rows, columns = np.nonzero(isLocalMaximum & (residual > median + thresholdInRobustSigmas * robustSigma))
    verticalFrequencies = (rows - tileHeight // 2) / tileHeight
    horizontalFrequencies = (columns - tileWidth // 2) / tileWidth
    isAboveDc = np.hypot(verticalFrequencies, horizontalFrequencies) > MINIMUM_PEAK_FREQUENCY_CYCLES_PER_PIXEL
    strongestFirst = np.argsort(-residual[rows, columns])
    keptIndexes = [i for i in strongestFirst if isAboveDc[i]][:MAXIMUM_PEAKS_PER_CHANNEL]
    return [(verticalFrequencies[i], horizontalFrequencies[i]) for i in keptIndexes]


def applyNotches(spectrum, peaks):
    """spectrum is fftshifted; attenuates a gaussian patch around every peak in place."""
    height, width = spectrum.shape
    sigmaRows = NOTCH_SIGMA_CYCLES_PER_PIXEL * height
    sigmaColumns = NOTCH_SIGMA_CYCLES_PER_PIXEL * width
    radiusRows = int(np.ceil(NOTCH_RADIUS_IN_SIGMAS * sigmaRows))
    radiusColumns = int(np.ceil(NOTCH_RADIUS_IN_SIGMAS * sigmaColumns))
    for verticalFrequency, horizontalFrequency in peaks:
        centerRow = int(round(height // 2 + verticalFrequency * height))
        centerColumn = int(round(width // 2 + horizontalFrequency * width))
        top, bottom = max(centerRow - radiusRows, 0), min(centerRow + radiusRows + 1, height)
        left, right = max(centerColumn - radiusColumns, 0), min(centerColumn + radiusColumns + 1, width)
        rowOffsets = (np.arange(top, bottom) - centerRow)[:, None] / sigmaRows
        columnOffsets = (np.arange(left, right) - centerColumn)[None, :] / sigmaColumns
        spectrum[top:bottom, left:right] *= 1 - np.exp(-0.5 * (rowOffsets ** 2 + columnOffsets ** 2))


def descreenChannel(channel, thresholdInRobustSigmas):
    peaks = findHalftonePeaks(channel, thresholdInRobustSigmas)
    spectrum = np.fft.fftshift(scipyFft.fft2(channel, workers=-1))
    applyNotches(spectrum, peaks)
    return scipyFft.ifft2(np.fft.ifftshift(spectrum), workers=-1).real


def descreenImage(image, sensitivity, sharpenAmount):
    """sensitivity 1..10 (higher removes more); sharpenAmount 0..1 restores edge crispness."""
    thresholdInRobustSigmas = 12 - sensitivity
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    filteredChannels = [descreenChannel(rgbArray[:, :, i], thresholdInRobustSigmas) for i in range(3)]
    filteredArray = np.clip(np.stack(filteredChannels, axis=2), 0, 255).astype(np.uint8)
    result = Image.fromarray(filteredArray)
    if sharpenAmount > 0:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(sharpenAmount * 200), threshold=2))
    return result


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
