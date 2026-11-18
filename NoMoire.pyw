"""NoMoire: standalone halftone/moire remover. Drag & drop images or folders onto the window.

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


SPECTRUM_TILE_SIZE = 512
SPECTRUM_TILE_GRID = 3
MINIMUM_PEAK_FREQUENCY_CYCLES_PER_PIXEL = 0.04
MAXIMUM_PEAKS = 200
BEAT_PEAK_PROMINENCE = 3.0
FUNDAMENTAL_PEAK_COUNT = 8
BACKGROUND_FILTER_SIZE = 21
MINIMUM_PEAK_PROMINENCE_BASE = 8.0  # natural-log power above local background; strength 5 -> 5.5
NOTCH_MINIMUM_SIGMA_CYCLES_PER_PIXEL = 0.005
NOTCH_SIGMA_FRACTION_OF_PEAK_FREQUENCY = 0.08
NOTCH_RADIUS_IN_SIGMAS = 4
LOWPASS_SIGMA_FRACTION_OF_FUNDAMENTAL = 0.6


class ScreenPeaks:
    """Detected halftone screen: peak frequencies in cycles per pixel and the lowest screen frequency."""

    def __init__(self, peaks, fundamentalFrequency):
        self.peaks = peaks
        self.fundamentalFrequency = fundamentalFrequency


def averageSpectrumPowerOverTiles(rgbArray):
    height, width, _ = rgbArray.shape
    tileHeight, tileWidth = min(height, SPECTRUM_TILE_SIZE), min(width, SPECTRUM_TILE_SIZE)
    window = np.outer(np.hanning(tileHeight), np.hanning(tileWidth)).astype(np.float32)
    topOffsets = np.linspace(0, height - tileHeight, SPECTRUM_TILE_GRID).astype(int)
    leftOffsets = np.linspace(0, width - tileWidth, SPECTRUM_TILE_GRID).astype(int)
    totalPower = np.zeros((tileHeight, tileWidth), np.float64)
    tileCount = 0
    for top in topOffsets:
        for left in leftOffsets:
            for channelIndex in range(3):
                tile = rgbArray[top:top + tileHeight, left:left + tileWidth, channelIndex]
                totalPower += np.abs(scipyFft.fft2((tile - tile.mean()) * window, workers=-1)) ** 2
                tileCount += 1
    return np.fft.fftshift(totalPower / tileCount)


def detectScreenPeaks(rgbArray, strength):
    """strength 1..10: higher accepts weaker peaks. Returns ScreenPeaks (empty peaks if no screen found)."""
    meanPower = averageSpectrumPowerOverTiles(rgbArray)
    tileHeight, tileWidth = meanPower.shape
    logSpectrum = np.log(meanPower + 1e-3)
    residual = logSpectrum - uniform_filter(logSpectrum, BACKGROUND_FILTER_SIZE)
    isLocalMaximum = residual == maximum_filter(residual, size=5)
    hasPeakAbove = lambda threshold: np.nonzero(isLocalMaximum & (residual > threshold))
    verticalFrequencyOf = lambda rows: (rows - tileHeight // 2) / tileHeight
    horizontalFrequencyOf = lambda columns: (columns - tileWidth // 2) / tileWidth
    rows, columns = hasPeakAbove(MINIMUM_PEAK_PROMINENCE_BASE - 0.5 * strength)
    radii = np.hypot(verticalFrequencyOf(rows), horizontalFrequencyOf(columns))
    strongestFirst = [i for i in np.argsort(-residual[rows, columns]) if radii[i] > MINIMUM_PEAK_FREQUENCY_CYCLES_PER_PIXEL]
    if not strongestFirst:
        return ScreenPeaks([], 0.0)
    fundamentalFrequency = min(radii[i] for i in strongestFirst[:FUNDAMENTAL_PEAK_COUNT])
    # beats between the screens of different inks sit below the fundamental and are weaker
    rows, columns = hasPeakAbove(BEAT_PEAK_PROMINENCE)
    radii = np.hypot(verticalFrequencyOf(rows), horizontalFrequencyOf(columns))
    keptIndexes = [i for i in np.argsort(-residual[rows, columns])
                   if radii[i] > MINIMUM_PEAK_FREQUENCY_CYCLES_PER_PIXEL][:MAXIMUM_PEAKS]
    peaks = [(verticalFrequencyOf(rows[i]), horizontalFrequencyOf(columns[i])) for i in keptIndexes]
    return ScreenPeaks(peaks, fundamentalFrequency)


def applyNotches(spectrum, peaks):
    """spectrum is fftshifted; attenuates a gaussian patch around every peak in place."""
    height, width = spectrum.shape
    for verticalFrequency, horizontalFrequency in peaks:
        sigma = max(NOTCH_MINIMUM_SIGMA_CYCLES_PER_PIXEL,
                    NOTCH_SIGMA_FRACTION_OF_PEAK_FREQUENCY * np.hypot(verticalFrequency, horizontalFrequency))
        sigmaRows, sigmaColumns = sigma * height, sigma * width
        centerRow = int(round(height // 2 + verticalFrequency * height))
        centerColumn = int(round(width // 2 + horizontalFrequency * width))
        top = max(int(centerRow - NOTCH_RADIUS_IN_SIGMAS * sigmaRows), 0)
        bottom = min(int(centerRow + NOTCH_RADIUS_IN_SIGMAS * sigmaRows) + 1, height)
        left = max(int(centerColumn - NOTCH_RADIUS_IN_SIGMAS * sigmaColumns), 0)
        right = min(int(centerColumn + NOTCH_RADIUS_IN_SIGMAS * sigmaColumns) + 1, width)
        rowOffsets = (np.arange(top, bottom) - centerRow)[:, None] / sigmaRows
        columnOffsets = (np.arange(left, right) - centerColumn)[None, :] / sigmaColumns
        spectrum[top:bottom, left:right] *= 1 - np.exp(-0.5 * (rowOffsets ** 2 + columnOffsets ** 2))


def buildLowpassGain(height, width, fundamentalFrequency):
    sigma = LOWPASS_SIGMA_FRACTION_OF_FUNDAMENTAL * fundamentalFrequency
    rowGain = np.exp(-0.5 * (((np.arange(height) - height // 2) / height) / sigma) ** 2)
    columnGain = np.exp(-0.5 * (((np.arange(width) - width // 2) / width) / sigma) ** 2)
    return np.outer(rowGain, columnGain).astype(np.float32)


def descreenWithPeaks(image, screenPeaks, sharpenAmount):
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    if not screenPeaks.peaks:
        return image.convert("RGB")
    height, width, _ = rgbArray.shape
    lowpassGain = buildLowpassGain(height, width, screenPeaks.fundamentalFrequency)
    filteredChannels = []
    for channelIndex in range(3):
        spectrum = np.fft.fftshift(scipyFft.fft2(rgbArray[:, :, channelIndex], workers=-1))
        applyNotches(spectrum, screenPeaks.peaks)
        filteredChannels.append(scipyFft.ifft2(np.fft.ifftshift(spectrum * lowpassGain), workers=-1).real)
    result = Image.fromarray(np.clip(np.stack(filteredChannels, axis=2), 0, 255).astype(np.uint8))
    if sharpenAmount > 0:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(sharpenAmount * 300), threshold=1))
    return result


def descreenImage(image, strength, sharpenAmount):
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    return descreenWithPeaks(image, detectScreenPeaks(rgbArray, strength), sharpenAmount)


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

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
DEBOUNCE_MILLISECONDS = 250
PREVIEW_CROP_SIZE = 1024


def cropCenter(image, size):
    width, height = min(image.width, size), min(image.height, size)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def convertPilImageToPixmap(image):
    rgbImage = image.convert("RGB")
    qImage = QImage(rgbImage.tobytes(), rgbImage.width, rgbImage.height, rgbImage.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qImage)


class SaveAllWorker(QThread):
    progressed = Signal(int, int)

    def __init__(self, filePaths, strength, sharpenAmount):
        super().__init__()
        self.filePaths, self.strength, self.sharpenAmount = filePaths, strength, sharpenAmount

    def run(self):
        for index, filePath in enumerate(self.filePaths, 1):
            result = descreenImage(Image.open(filePath), self.strength, self.sharpenAmount)
            result.save(filePath.with_name(f"{filePath.stem}_descreened{filePath.suffix}"), quality=95)
            self.progressed.emit(index, len(self.filePaths))


class DescreenerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Descreener")
        self.resize(1280, 760)
        self.filePaths = []
        self.saveAllWorker = None
        self.setAcceptDrops(True)
        self.cachedPeaksKey = None
        self.debounceTimer = QTimer(singleShot=True, interval=DEBOUNCE_MILLISECONDS)
        self.debounceTimer.timeout.connect(self.refreshPreview)

        self.fileList = QListWidget()
        self.fileList.setAcceptDrops(False)
        self.fileList.currentRowChanged.connect(lambda _: self.debounceTimer.start())
        self.strengthSlider = self.buildSlider(1, 10, 5)
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
        for caption, slider in (("Strength", self.strengthSlider), ("Edge sharpen", self.sharpenSlider)):
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

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        droppedPaths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        imagePaths = [p for path in droppedPaths for p in (sorted(path.iterdir()) if path.is_dir() else [path])
                      if p.suffix.lower() in SUPPORTED_SUFFIXES]
        if imagePaths:
            self.loadImages(imagePaths)

    def openImages(self):
        selectedPaths, _ = QFileDialog.getOpenFileNames(self, "Open images", "", SUPPORTED_FILE_FILTER)
        if selectedPaths:
            self.loadImages([Path(p) for p in selectedPaths])

    def loadImages(self, imagePaths):
        self.filePaths = imagePaths
        self.fileList.clear()
        self.fileList.addItems([p.name for p in self.filePaths])
        self.saveAllButton.setEnabled(True)
        self.fileList.setCurrentRow(0)

    def refreshPreview(self):
        row = self.fileList.currentRow()
        if row < 0:
            return
        filePath, strength = self.filePaths[row], self.strengthSlider.value()
        if self.cachedPeaksKey != (filePath, strength):
            fullImage = Image.open(filePath).convert("RGB")
            self.cachedPeaks = detectScreenPeaks(np.asarray(fullImage, dtype=np.float32), strength)
            self.cachedCrop = cropCenter(fullImage, PREVIEW_CROP_SIZE)
            self.cachedPeaksKey = (filePath, strength)
        result = descreenWithPeaks(self.cachedCrop, self.cachedPeaks, self.sharpenSlider.value() / 100)
        self.statusLabel.setText(f"Screen peaks found: {len(self.cachedPeaks.peaks)}" if self.cachedPeaks.peaks
                                 else "No halftone screen detected")
        for label, image in ((self.beforeLabel, self.cachedCrop), (self.afterLabel, result)):
            label.setPixmap(convertPilImageToPixmap(image))

    def saveAll(self):
        self.saveAllButton.setEnabled(False)
        self.saveAllWorker = SaveAllWorker(self.filePaths, self.strengthSlider.value(), self.sharpenSlider.value() / 100)
        self.saveAllWorker.progressed.connect(lambda done, total: self.statusLabel.setText(f"Saved {done}/{total}"))
        self.saveAllWorker.finished.connect(lambda: self.saveAllButton.setEnabled(True))
        self.saveAllWorker.start()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setStyleSheet(STYLESHEET)
    window = DescreenerWindow()
    window.show()
    sys.exit(application.exec())
