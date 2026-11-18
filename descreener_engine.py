"""Descreener engine: removes halftone moire by notching spectral peaks per color channel."""
import numpy as np
from PIL import Image, ImageFilter
from scipy import fft as scipyFft
from scipy.ndimage import maximum_filter, uniform_filter

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
