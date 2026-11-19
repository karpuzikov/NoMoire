"""NoMoire: standalone halftone/moire remover. Drag & drop images or folders onto the window.

Method (decoded from Sattva Descreen): find each print screen (cyan/magenta/yellow/black angle and frequency),
low-pass the picture just under the screen frequency, then fit and subtract the beat (moire) patterns between
screens with a brightness-level regression. The same filter runs on every pixel.

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


def detectScreensInImage(image):
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    return [detectScreens(rgbArray[..., channel]) for channel in range(3)]


SCREEN_RECORD_NAMES = ("cyan", "magenta", "black", "yellow")
LATTICE_HARMONICS = [(1, 0), (0, 1), (1, 1), (1, -1)]  # the four Sattva uses in moire mode
LOWPASS_CUTOFF_OF_SCREEN_FREQUENCY = 0.98  # Sattva table 2d10
LOWPASS_COSINE_AMPLITUDES = (1.1344, -0.1444, 0.0007, 0.0093)
BEAT_COSINE_AMPLITUDES = (0.8061, 0.2048, -0.0076, -0.0033)
EXTRACTION_DISK_RADIUS_OF_SCREEN_FREQUENCY = 0.1367
DIAGONAL_HARMONIC_AMPLITUDE = 0.922
BEAT_DISK_RADIUS_OF_SCREEN_FREQUENCY = 0.15
BRIGHTNESS_LEVEL_COUNT = 50
MINIMUM_LEVEL_GROUP_PIXEL_FRACTION = 0.10
REGRESSION_TILE_CORE_PIXELS = 1536
REGRESSION_TILE_MARGIN_PIXELS = 192
STRENGTH_TO_CUTOFF_STEP = 0.04  # shortcut: strength 5 = Sattva default; each step moves the cutoff 4 %


def radialCosineWindow(normalizedRadius, amplitudes):
    window = sum(amplitude * np.cos((2 * index + 1) * np.pi / 2 * normalizedRadius) for index, amplitude in enumerate(amplitudes))
    return np.where(normalizedRadius >= 1, 0.0, window).astype(np.float32)


def classifyScreensIntoRecords(screensPerChannel):
    """Maps detected fundamentals to cyan/magenta/black/yellow by their angle. Returns {name: (fx, fy)}."""
    records = {}
    for screens in screensPerChannel:
        for fx, fy, _ in screens:
            angle = np.degrees(np.arctan2(fy, fx)) % 90
            name = "yellow" if angle < 5 or angle > 85 else "cyan" if angle < 30 else "black" if angle < 60 else "magenta"
            records.setdefault(name, (fx, fy))
    return records


def stampDisks(shape, centres, radius, profile):
    height, width = shape
    mask = np.zeros(shape, np.float32)
    radiusBins = radius * max(height, width)
    reach = int(np.ceil(radiusBins)) + 1
    offsetX, offsetY = np.meshgrid(np.arange(-reach, reach + 1), np.arange(-reach, reach + 1))
    for centreX, centreY in centres:
        for sign in (1, -1):
            binX, binY = sign * centreX * width, sign * centreY * height
            roundedX, roundedY = int(round(binX)), int(round(binY))
            distance = np.hypot(roundedX + offsetX - binX, roundedY + offsetY - binY)
            rows, columns = (roundedY + offsetY) % height, (roundedX + offsetX) % width
            mask[rows, columns] = np.maximum(mask[rows, columns], profile(distance, radiusBins).astype(np.float32))
    return mask


def screenHarmonicCentres(fundamental, harmonics):
    fx, fy = fundamental
    centres = [(m * fx - n * fy, m * fy + n * fx) for m, n in harmonics]
    return [(x, y) for x, y in centres if np.hypot(x, y) < 0.5]


def descreenTile(tile, records, screenFrequency, cutoffScale):
    """tile: float32 (h, w, 3), already margin-padded. Returns the descreened tile (same size)."""
    height, width, _ = tile.shape
    spectra = [scipyFft.fft2(tile[..., channel]) for channel in range(3)]
    radius = np.hypot(np.fft.fftfreq(width)[None, :], np.fft.fftfreq(height)[:, None]).astype(np.float32)
    lowpass = radialCosineWindow(radius / (LOWPASS_CUTOFF_OF_SCREEN_FREQUENCY * cutoffScale * screenFrequency), LOWPASS_COSINE_AMPLITUDES)
    spectrumOfRecord = {"cyan": spectra[0], "magenta": spectra[1], "black": 0.5 * (spectra[0] + spectra[1]), "yellow": spectra[2]}
    extractionProfile = lambda distance, radiusBins: np.where(distance < radiusBins + 1, np.cos(np.pi / 2 * distance / (radiusBins + 1)), 0)
    components = {}
    for name, fundamental in records.items():
        spectrum = np.zeros((height, width), np.complex64)
        for index, harmonic in enumerate(LATTICE_HARMONICS):
            for centre in screenHarmonicCentres(fundamental, [harmonic]):
                weight = stampDisks((height, width), [centre], EXTRACTION_DISK_RADIUS_OF_SCREEN_FREQUENCY * screenFrequency, extractionProfile)
                spectrum += weight * spectrumOfRecord[name] * (1.0 if index < 2 else DIAGONAL_HARMONIC_AMPLITUDE)
        components[name] = scipyFft.ifft2(spectrum).real.astype(np.float32)
    predictorPairs = [(a, b) for a, b in (("magenta", "black"), ("cyan", "black"), ("cyan", "magenta"),
                                          ("cyan", "yellow"), ("magenta", "yellow"), ("black", "yellow"))
                      if a in components and b in components]
    predictors = [scipyFft.ifft2(scipyFft.fft2(components[a] * components[b]) * lowpass).real.astype(np.float32)
                  for a, b in predictorPairs]
    vectors = [(name, x, y) for name, fundamental in records.items() for x, y in screenHarmonicCentres(fundamental, LATTICE_HARMONICS)]
    beatCentres = [(ax + sign * bx, ay + sign * by) for i, (nameA, ax, ay) in enumerate(vectors)
                   for nameB, bx, by in vectors[i + 1:] if nameA != nameB for sign in (1, -1)]
    beatCentres = [c for c in beatCentres if np.hypot(*c) < 0.5]
    beatRadius = BEAT_DISK_RADIUS_OF_SCREEN_FREQUENCY * screenFrequency
    beatProfile = lambda distance, radiusBins: radialCosineWindow(np.clip(distance / radiusBins, 0, 1), BEAT_COSINE_AMPLITUDES) * (distance < radiusBins)
    beatMask = stampDisks((height, width), beatCentres, beatRadius, beatProfile) if beatCentres else 0
    lowFrequencyMask = stampDisks((height, width), [(0, 0)], 2 * beatRadius, beatProfile)
    referenceMask = 1 - np.maximum(beatMask, lowFrequencyMask)
    bases = [scipyFft.ifft2(spectra[c] * lowpass).real for c in range(3)]
    references = [scipyFft.ifft2(spectra[c] * lowpass * referenceMask).real for c in range(3)]
    if not predictors:
        return np.stack(bases, -1).astype(np.float32)
    levels = np.clip((0.5 * (references[0] + references[1]) * BRIGHTNESS_LEVEL_COUNT / 255).astype(int), 0, BRIGHTNESS_LEVEL_COUNT - 1)
    levelCounts = np.bincount(levels.ravel(), minlength=BRIGHTNESS_LEVEL_COUNT)
    groupOfLevel, group, accumulated = np.zeros(BRIGHTNESS_LEVEL_COUNT, int), 0, 0
    for level in range(BRIGHTNESS_LEVEL_COUNT):
        groupOfLevel[level] = group
        accumulated += levelCounts[level]
        if accumulated >= MINIMUM_LEVEL_GROUP_PIXEL_FRACTION * levels.size:
            group, accumulated = group + 1, 0
    if group > 0 and accumulated:
        groupOfLevel[groupOfLevel == group] = group - 1
    groups = groupOfLevel[levels]
    outputs = []
    for channel in range(3):
        residual = tile[..., channel].astype(np.float64) - references[channel]
        correction = np.zeros_like(residual)
        for groupIndex in range(groups.max() + 1):
            selection = groups == groupIndex
            if selection.sum() < 50:
                continue
            matrix = np.stack([p[selection] for p in predictors], 1).astype(np.float64)
            gram, cross = matrix.T @ matrix, matrix.T @ residual[selection]
            kept = list(range(len(predictors)))
            while kept:
                coefficients = np.linalg.lstsq(gram[np.ix_(kept, kept)], cross[kept], rcond=None)[0]
                negative = [k for k, c in zip(kept, coefficients) if c <= 0]
                if not negative:
                    break
                kept = [k for k in kept if k not in negative]
            if kept:
                full = np.zeros(len(predictors))
                full[kept] = coefficients
                correction[selection] = matrix @ full
        outputs.append(bases[channel] - correction)
    return np.stack(outputs, -1).astype(np.float32)


def descreenArray(rgbArray, screensPerChannel, strength):
    records = classifyScreensIntoRecords(screensPerChannel)
    regular = [records[name] for name in ("cyan", "magenta", "black") if name in records]
    if len(records) < 2 or not regular:
        return rgbArray
    screenFrequency = float(np.mean([np.hypot(fx, fy) for fx, fy in regular]))
    cutoffScale = 1 + STRENGTH_TO_CUTOFF_STEP * (5 - strength)
    height, width, _ = rgbArray.shape
    margin, core = REGRESSION_TILE_MARGIN_PIXELS, REGRESSION_TILE_CORE_PIXELS
    padded = np.pad(rgbArray, ((margin, margin + core), (margin, margin + core), (0, 0)), mode="reflect")
    output = np.empty_like(rgbArray)
    for top in range(0, height, core):
        for left in range(0, width, core):
            tile = padded[top:top + core + 2 * margin, left:left + core + 2 * margin]
            processed = descreenTile(tile, records, screenFrequency, cutoffScale)
            piece = processed[margin:margin + core, margin:margin + core]
            output[top:top + core, left:left + core] = piece[:height - top, :width - left]
    return output


def descreenWithScreens(image, screensPerChannel, strength, sharpenAmount):
    rgbArray = np.asarray(image.convert("RGB"), dtype=np.float32)
    rgbArray = descreenArray(rgbArray, screensPerChannel, strength)
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
