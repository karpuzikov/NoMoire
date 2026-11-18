"""NoMoire: standalone halftone/moire remover. Drag & drop images or folders onto the window.

Method (after Sattva Descreen): find each print screen (cyan/magenta/yellow/black angle and frequency) in the
spectrum, then cut cosine-tapered disks out of the spectrum at the screen's lattice harmonics and at the beats
between screens. Everything else in the picture is left untouched.

Install:  pip install numpy scipy Pillow PySide6
Run:      double-click NoMoire.pyw
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import fft as scipyFft
from scipy.ndimage import gaussian_filter, maximum_filter
import sys
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QListWidget, QMainWindow, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

Image.MAX_IMAGE_PIXELS = None

# Harmonics (m, n) of a square screen lattice: peak = fundamental * (m + i n). Same table Sattva Descreen uses.
SCREEN_HARMONICS = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (1, 2), (2, 1), (1, -2), (2, -1), (2, 2), (2, -2)]
DETECTION_TILE_SIZE = 512
DETECTION_TILE_GRID = 3
MAXIMUM_SCREENS = 4  # cyan, magenta, yellow, black
MINIMUM_SCREEN_FREQUENCY_CYCLES_PER_PIXEL = 0.05
MINIMUM_SCREEN_SCORE = 10.0
FILTER_TILE_SIZE = 1024  # 50% overlapped Hann tiles
SCREEN_DISK_RADIUS_BASE = 0.3  # fraction of screen frequency; strength 5 -> 0.55
SCREEN_DISK_RADIUS_PER_STRENGTH = 0.05
BEAT_DISK_RADIUS_PER_STRENGTH = 0.024  # strength 5 -> 0.12
BEAT_MAXIMUM_ORDER = 2
BEAT_MINIMUM_FREQUENCY = 0.03
SHARPEN_RADIUS_PIXELS = 1.5


def averageSpectrumPower(channel):
    height, width = channel.shape
    window = np.outer(np.hanning(DETECTION_TILE_SIZE), np.hanning(DETECTION_TILE_SIZE)).astype(np.float32)
    power = np.zeros((DETECTION_TILE_SIZE,) * 2, np.float64)
    for gridY in range(DETECTION_TILE_GRID):
        for gridX in range(DETECTION_TILE_GRID):
            top = int((height - DETECTION_TILE_SIZE) * (gridY + 0.5) / DETECTION_TILE_GRID)
            left = int((width - DETECTION_TILE_SIZE) * (gridX + 0.5) / DETECTION_TILE_GRID)
            tile = channel[top:top + DETECTION_TILE_SIZE, left:left + DETECTION_TILE_SIZE]
            power += np.abs(scipyFft.fft2((tile - tile.mean()) * window)) ** 2
    return scipyFft.fftshift(power / DETECTION_TILE_GRID ** 2)


def scoreLattice(prominence, fundamentalBins):
    center = DETECTION_TILE_SIZE // 2
    score, counted = 0.0, 0
    for m, n in SCREEN_HARMONICS:
        x = m * fundamentalBins[0] - n * fundamentalBins[1]
        y = n * fundamentalBins[0] + m * fundamentalBins[1]
        if abs(x) >= center - 2 or abs(y) >= center - 2:
            continue
        ix, iy = int(round(x)) + center, int(round(y)) + center
        score += prominence[iy - 2:iy + 3, ix - 2:ix + 3].max()
        counted += 1
    return score, counted


def detectScreens(channel):
    """Returns [(fx, fy, score)] fundamentals in cycles/pixel, strongest lattice first."""
    power = averageSpectrumPower(channel)
    center = DETECTION_TILE_SIZE // 2
    logPower = np.log(power + 1)
    prominence = logPower - gaussian_filter(logPower, 12)
    searchMap = prominence.copy()
    yy, xx = np.mgrid[-center:center, -center:center]
    searchMap[np.hypot(xx, yy) < MINIMUM_SCREEN_FREQUENCY_CYCLES_PER_PIXEL * DETECTION_TILE_SIZE] = -1e9
    screens = []
    for _ in range(MAXIMUM_SCREENS):
        isCandidate = (searchMap == maximum_filter(searchMap, size=5)) & (searchMap > 3.0) & (yy >= 0)
        candidates = np.argwhere(isCandidate)
        if not len(candidates):
            break
        best = None
        for iy, ix in candidates[np.argsort(-searchMap[isCandidate])[:40]]:
            neighborhood = power[iy - 1:iy + 2, ix - 1:ix + 2]
            weights = neighborhood / neighborhood.sum()
            offsetY, offsetX = np.mgrid[-1:2, -1:2]
            bins = (ix - center + float((weights * offsetX).sum()), iy - center + float((weights * offsetY).sum()))
            score, counted = scoreLattice(prominence, bins)
            if counted >= 2 and (best is None or score > best[0]):
                best = (score, bins)
        if best is None or best[0] < MINIMUM_SCREEN_SCORE:
            break
        score, (binX, binY) = best
        screens.append((binX / DETECTION_TILE_SIZE, binY / DETECTION_TILE_SIZE, score))
        for m, n in SCREEN_HARMONICS:
            x, y = m * binX - n * binY, n * binX + m * binY
            for sign in (1, -1):
                ix, iy = int(round(sign * x)) + center, int(round(sign * y)) + center
                if 0 <= ix < DETECTION_TILE_SIZE and 0 <= iy < DETECTION_TILE_SIZE:
                    searchMap[max(iy - 6, 0):iy + 7, max(ix - 6, 0):ix + 7] = -1e9
    return screens


def stampCosineDisk(mask, frequencyX, frequencyY, radius, size):
    """Cosine-tapered disk (1 at the centre, 0 at the rim) at a fractional frequency and its mirror."""
    for sign in (1, -1):
        centerX, centerY, radiusBins = sign * frequencyX * size, sign * frequencyY * size, radius * size
        columns = np.arange(int(np.floor(centerX - radiusBins)), int(np.ceil(centerX + radiusBins)) + 1)
        for row in range(int(np.floor(centerY - radiusBins)), int(np.ceil(centerY + radiusBins)) + 1):
            distance = np.hypot(columns - centerX, row - centerY)
            weight = np.where(distance < radiusBins, np.cos(0.5 * np.pi * distance / radiusBins), 0)
            mask[row % size, columns % size] = np.maximum(mask[row % size, columns % size], weight)


def buildRemovalMask(size, screens, strength):
    mask = np.zeros((size, size), np.float32)
    screenRadius = SCREEN_DISK_RADIUS_BASE + SCREEN_DISK_RADIUS_PER_STRENGTH * strength
    for fundamentalX, fundamentalY, _ in screens:
        frequency = np.hypot(fundamentalX, fundamentalY)
        for m, n in SCREEN_HARMONICS:
            x, y = m * fundamentalX - n * fundamentalY, n * fundamentalX + m * fundamentalY
            if np.hypot(x, y) < 0.5:
                stampCosineDisk(mask, x, y, screenRadius * frequency, size)
    if len(screens) > 1:
        vectors = []
        for index, (fundamentalX, fundamentalY, _) in enumerate(screens):
            vectors += [(index, fundamentalX, fundamentalY), (index, -fundamentalY, fundamentalX)]
        meanFrequency = np.mean([np.hypot(x, y) for x, y, _ in screens])
        stamped = set()
        orders = [o for o in range(-BEAT_MAXIMUM_ORDER, BEAT_MAXIMUM_ORDER + 1) if o]
        for screenA, ax, ay in vectors:
            for screenB, bx, by in vectors:
                if screenA >= screenB:
                    continue
                for orderA in orders:
                    for orderB in orders:
                        x, y = orderA * ax + orderB * bx, orderA * ay + orderB * by
                        if not BEAT_MINIMUM_FREQUENCY <= np.hypot(x, y) < 0.5:
                            continue
                        key = (round(x * size), round(y * size))
                        if key in stamped or (-key[0], -key[1]) in stamped:
                            continue
                        stamped.add(key)
                        stampCosineDisk(mask, x, y, BEAT_DISK_RADIUS_PER_STRENGTH * strength * meanFrequency, size)
    return mask


def removeScreensFromChannel(channel, screens, strength):
    height, width = channel.shape
    size, hop = FILTER_TILE_SIZE, FILTER_TILE_SIZE // 2
    mask = buildRemovalMask(size, screens, strength)
    padded = np.pad(channel.astype(np.float32), ((hop, hop + size), (hop, hop + size)), mode="reflect")
    hann = np.hanning(size + 1)[:-1].astype(np.float32)
    window = np.outer(hann, hann)
    result = np.zeros_like(padded)
    for top in range(0, height + size, hop):
        for left in range(0, width + size, hop):
            tile = padded[top:top + size, left:left + size]
            if tile.shape != (size, size):
                continue
            screenPart = scipyFft.ifft2(scipyFft.fft2(tile - tile.mean()) * mask).real.astype(np.float32)
            result[top:top + size, left:left + size] += (tile - screenPart) * window
    return result[hop:hop + height, hop:hop + width]


def detectScreensInImage(image):
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    return [detectScreens(rgbArray[..., channel]) for channel in range(3)]


def descreenWithScreens(image, screensPerChannel, strength, sharpenAmount):
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    for channel, screens in enumerate(screensPerChannel):
        if screens:
            rgbArray[..., channel] = removeScreensFromChannel(rgbArray[..., channel], screens, strength)
    result = Image.fromarray(np.clip(rgbArray + 0.5, 0, 255).astype(np.uint8))
    if sharpenAmount > 0:
        result = result.filter(ImageFilter.UnsharpMask(radius=SHARPEN_RADIUS_PIXELS, percent=int(sharpenAmount * 300), threshold=1))
    return result


def descreenImage(image, strength, sharpenAmount):
    return descreenWithScreens(image, detectScreensInImage(image), strength, sharpenAmount)


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
        screens = detectScreensInImage(original)
        self.processed.emit(original, descreenWithScreens(original, screens, self.strength, self.sharpenAmount),
                            max(len(channelScreens) for channelScreens in screens))


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
        self.sharpenSlider = self.buildSlider(0, 100, 20)
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

    def onPreviewProcessed(self, original, result, screenCount):
        self.originalPixmap, self.resultPixmap = convertPilImageToPixmap(original), convertPilImageToPixmap(result)
        self.showPixmap(self.resultPixmap)
        self.statusLabel.setText(f"Print screens found: {screenCount}. Wheel zooms, double-click toggles 100%"
                                 if screenCount else "No halftone screen detected")

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
