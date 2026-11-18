"""NoMoire: standalone halftone/moire remover. Drag & drop images or folders onto the window.

Install:  pip install numpy scipy Pillow PySide6
Run:      double-click NoMoire.pyw
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import fft as scipyFft
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter
import sys
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton, QSlider, QVBoxLayout,
                               QWidget)


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
NOTCHED_LOWPASS_FRACTION = 0.6
BLURRED_LOWPASS_FRACTION = 0.2
EDGE_GRADIENT_LOW = 6.0
EDGE_GRADIENT_HIGH = 25.0
EDGE_SPREAD_SIGMA = 24.0
FLAT_WINDOW_SIGMA = 3.0
FLAT_DEVIATION_LOW = 2.0
FLAT_DEVIATION_HIGH = 5.0
EDGE_SHARPEN_RADIUS = 3.0
BLEND_STRIP_HEIGHT = 512
EDGE_SHARPEN_PERCENT = 120
REFLECT_PADDING_PIXELS = 64  # avoids FFT wrap-around artifacts at image borders


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


def buildLowpassGain(height, width, fundamentalFrequency, sigmaFraction):
    sigma = sigmaFraction * fundamentalFrequency
    rowGain = np.exp(-0.5 * (((np.arange(height) - height // 2) / height) / sigma) ** 2)
    columnGain = np.exp(-0.5 * (((np.arange(width) - width // 2) / width) / sigma) ** 2)
    return np.outer(rowGain, columnGain).astype(np.float32)


def buildEdgeAndFlatMasks(grayImage):
    """Edge mask: 1 near strong edges (notching rings there). Flat mask: 1 in featureless areas (any faint
    residual lattice shows there). Both computed at quarter resolution."""
    small = np.asarray(grayImage.reduce(4), dtype=np.float32)
    gradientMagnitude = np.hypot(gaussian_filter(small, 1.0, order=(0, 1)), gaussian_filter(small, 1.0, order=(1, 0)))
    edgeStrength = np.clip((gradientMagnitude - EDGE_GRADIENT_LOW) / (EDGE_GRADIENT_HIGH - EDGE_GRADIENT_LOW), 0, 1)
    edgeMask = np.clip(gaussian_filter(edgeStrength, EDGE_SPREAD_SIGMA / 4) * 3, 0, 1)
    localMean = gaussian_filter(small, FLAT_WINDOW_SIGMA)
    localDeviation = np.sqrt(np.maximum(gaussian_filter(small * small, FLAT_WINDOW_SIGMA) - localMean ** 2, 0))
    flatMask = np.clip((FLAT_DEVIATION_HIGH - localDeviation) / (FLAT_DEVIATION_HIGH - FLAT_DEVIATION_LOW), 0, 1)
    upsample = lambda mask: np.asarray(Image.fromarray(mask.astype(np.float32)).resize(grayImage.size, Image.BILINEAR))[..., None]
    return upsample(edgeMask), upsample(flatMask)


def descreenWithPeaks(image, screenPeaks, sharpenAmount):
    """Notch-filtered result by default; plain blur in featureless areas (faint lattice shows there) and
    blur + sharpen near strong edges (notching rings there). Memory-lean: uint8 intermediates, blend in strips."""
    rgbImage = image.convert("RGB")
    if not screenPeaks.peaks:
        return rgbImage
    width, height = rgbImage.size
    padding = min(REFLECT_PADDING_PIXELS, height - 1, width - 1)
    notched = np.empty((height, width, 3), np.uint8)
    blurred = np.empty((height, width, 3), np.uint8)
    paddedHeight, paddedWidth = height + 2 * padding, width + 2 * padding
    notchedGain = buildLowpassGain(paddedHeight, paddedWidth, screenPeaks.fundamentalFrequency, NOTCHED_LOWPASS_FRACTION)
    blurredGain = buildLowpassGain(paddedHeight, paddedWidth, screenPeaks.fundamentalFrequency, BLURRED_LOWPASS_FRACTION)
    toUint8Crop = lambda channel: np.clip(channel[padding:paddedHeight - padding, padding:paddedWidth - padding], 0, 255).astype(np.uint8)
    for channelIndex in range(3):
        channel = np.pad(np.asarray(rgbImage, dtype=np.float32)[:, :, channelIndex], padding, mode="reflect")
        spectrum = np.fft.fftshift(scipyFft.fft2(channel, workers=-1))
        blurred[:, :, channelIndex] = toUint8Crop(scipyFft.ifft2(np.fft.ifftshift(spectrum * blurredGain), workers=-1).real)
        applyNotches(spectrum, screenPeaks.peaks)
        notched[:, :, channelIndex] = toUint8Crop(scipyFft.ifft2(np.fft.ifftshift(spectrum * notchedGain), workers=-1).real)
        del channel, spectrum
    blurredImage = Image.fromarray(blurred)
    blurredSharp = np.asarray(blurredImage.filter(ImageFilter.UnsharpMask(radius=EDGE_SHARPEN_RADIUS, percent=EDGE_SHARPEN_PERCENT, threshold=0)))
    edgeMask, flatMask = buildEdgeAndFlatMasks(blurredImage.convert("L"))
    for top in range(0, height, BLEND_STRIP_HEIGHT):
        rows = slice(top, min(top + BLEND_STRIP_HEIGHT, height))
        edge, flat = edgeMask[rows], flatMask[rows]
        notched[rows] = (notched[rows] * ((1 - edge) * (1 - flat)) + blurred[rows] * ((1 - edge) * flat)
                         + blurredSharp[rows] * edge).astype(np.uint8)
    result = Image.fromarray(notched)
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
QPushButton:hover, QPushButton:pressed { border-color: #b8f23a; }
QPushButton#primary { background: #b8f23a; border: 1px solid #b8f23a; color: #0e0f11; font-weight: 600; }
QPushButton#primary:disabled { background: #26292e; border-color: #26292e; color: #6b7078; }
QGraphicsView { background: #17191c; border: 1px solid #26292e; border-radius: 8px; }
QLabel#caption, QLabel#status { color: #8a8f98; font-family: 'JetBrains Mono', Consolas, monospace; }
QSlider::groove:horizontal { height: 4px; background: #26292e; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #b8f23a; border-radius: 2px; }
QSlider::handle:horizontal { background: #b8f23a; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
"""

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SUPPORTED_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
DEBOUNCE_MILLISECONDS = 600
ZOOM_STEP = 1.25

Image.MAX_IMAGE_PIXELS = None


def convertPilImageToPixmap(image):
    rgbImage = image.convert("RGB")
    qImage = QImage(rgbImage.tobytes(), rgbImage.width, rgbImage.height, rgbImage.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qImage)


class ImageViewer(QGraphicsView):
    """Shows the whole picture fitted to the window. Wheel zooms, drag pans, double-click toggles fit / 100%."""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmapItem = QGraphicsPixmapItem()
        self.pixmapItem.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self.pixmapItem)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

    def showPixmap(self, pixmap, fitToWindow):
        self.pixmapItem.setPixmap(pixmap)
        self.scene().setSceneRect(self.pixmapItem.boundingRect())
        if fitToWindow:
            self.fitInView(self.pixmapItem, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        self.scale(*(2 * [ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP]))

    def mouseDoubleClickEvent(self, event):
        if abs(self.transform().m11() - 1) < 0.001:
            self.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        else:
            self.resetTransform()


class PreviewWorker(QThread):
    processed = Signal(object, object, int)

    def __init__(self, filePath, strength, sharpenAmount):
        super().__init__()
        self.filePath, self.strength, self.sharpenAmount = filePath, strength, sharpenAmount

    def run(self):
        original = Image.open(self.filePath).convert("RGB")
        screenPeaks = detectScreenPeaks(np.asarray(original, dtype=np.float32), self.strength)
        self.processed.emit(original, descreenWithPeaks(original, screenPeaks, self.sharpenAmount),
                            len(screenPeaks.peaks))


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
        self.setWindowTitle("NoMoire")
        self.resize(1280, 760)
        self.setAcceptDrops(True)
        self.filePaths = []
        self.previewWorker = None
        self.saveAllWorker = None
        self.isPreviewRequestPending = False
        self.originalPixmap = self.resultPixmap = None
        self.shouldFitNextPreview = True
        self.debounceTimer = QTimer(singleShot=True, interval=DEBOUNCE_MILLISECONDS)
        self.debounceTimer.timeout.connect(self.requestPreview)

        self.viewer = ImageViewer()
        self.fileList = QListWidget()
        self.fileList.setAcceptDrops(False)
        self.fileList.currentRowChanged.connect(self.onFileSelected)
        self.strengthSlider = self.buildSlider(1, 10, 5)
        self.sharpenSlider = self.buildSlider(0, 100, 30)
        self.openButton = QPushButton("Open images")
        self.openButton.clicked.connect(self.openImages)
        self.compareButton = QPushButton("Hold to see original")
        self.compareButton.pressed.connect(lambda: self.showPixmap(self.originalPixmap))
        self.compareButton.released.connect(lambda: self.showPixmap(self.resultPixmap))
        self.saveAllButton = QPushButton("Save all")
        self.saveAllButton.setObjectName("primary")
        self.saveAllButton.setEnabled(False)
        self.saveAllButton.clicked.connect(self.saveAll)
        self.statusLabel = QLabel("Drop images here. Saves as *_descreened")
        self.statusLabel.setObjectName("status")
        self.statusLabel.setWordWrap(True)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(12)
        sidebar.addWidget(self.openButton)
        sidebar.addWidget(self.fileList, 1)
        sidebar.addWidget(self.compareButton)
        for caption, slider in (("Strength", self.strengthSlider), ("Edge sharpen", self.sharpenSlider)):
            sidebar.addWidget(QLabel(caption, objectName="caption"))
            sidebar.addWidget(slider)
        sidebar.addWidget(self.saveAllButton)
        sidebar.addWidget(self.statusLabel)
        sidebarWidget = QWidget(objectName="panel")
        sidebarWidget.setLayout(sidebar)
        sidebarWidget.setFixedWidth(280)

        root = QHBoxLayout()
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(sidebarWidget)
        root.addWidget(self.viewer, 1)
        central = QWidget(objectName="panel")
        central.setLayout(root)
        self.setCentralWidget(central)

    def buildSlider(self, minimum, maximum, value):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda _: self.debounceTimer.start())
        return slider

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

    def onFileSelected(self, row):
        self.shouldFitNextPreview = True
        self.requestPreview()

    def requestPreview(self):
        if self.fileList.currentRow() < 0:
            return
        if self.previewWorker is not None and self.previewWorker.isRunning():
            self.isPreviewRequestPending = True
            return
        self.statusLabel.setText("Processing...")
        self.previewWorker = PreviewWorker(self.filePaths[self.fileList.currentRow()], self.strengthSlider.value(),
                                           self.sharpenSlider.value() / 100)
        self.previewWorker.processed.connect(self.onPreviewProcessed)
        self.previewWorker.finished.connect(self.onPreviewWorkerFinished)
        self.previewWorker.start()

    def onPreviewProcessed(self, original, result, peakCount):
        self.originalPixmap, self.resultPixmap = convertPilImageToPixmap(original), convertPilImageToPixmap(result)
        self.showPixmap(self.resultPixmap)
        self.statusLabel.setText(f"Screen peaks found: {peakCount}. Wheel zooms, double-click toggles 100%"
                                 if peakCount else "No halftone screen detected")

    def onPreviewWorkerFinished(self):
        if self.isPreviewRequestPending:
            self.isPreviewRequestPending = False
            self.requestPreview()

    def showPixmap(self, pixmap):
        self.viewer.showPixmap(pixmap, self.shouldFitNextPreview)
        self.shouldFitNextPreview = False

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
