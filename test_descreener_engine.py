import numpy as np
from PIL import Image
from descreener_engine import descreenImage


def buildHalftonedImage():
    y, x = np.mgrid[0:768, 0:768].astype(np.float32)
    clean = 120 + 60 * np.sin(x / 90) + 40 * (np.hypot(x - 384, y - 384) < 150)
    screen = 35 * np.cos(2 * np.pi * (0.21 * x + 0.12 * y)) + 25 * np.cos(2 * np.pi * (0.12 * x - 0.21 * y))
    toImage = lambda a: Image.fromarray(np.clip(np.stack([a] * 3, axis=2), 0, 255).astype(np.uint8))
    return toImage(clean), toImage(clean + screen)


def test_descreen_reduces_error_against_clean_image():
    clean, screened = buildHalftonedImage()
    result = descreenImage(screened, sensitivity=6, sharpenAmount=0)
    errorBefore = np.abs(np.asarray(screened, float) - np.asarray(clean, float)).mean()
    errorAfter = np.abs(np.asarray(result, float) - np.asarray(clean, float)).mean()
    print(errorBefore, errorAfter)
    assert errorAfter < errorBefore / 3
