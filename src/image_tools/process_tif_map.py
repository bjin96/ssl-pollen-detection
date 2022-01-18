from pathlib import Path

from PIL import Image

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960
HORIZONTAL_TILES = range(6)
VERTICAL_TILES = range(24)
HORIZONTAL_TIF_LABELS = range(14, 8, -1)
VERTICAL_TIF_LABELS = range(23, -1, -1)


def crop_tif_map(
        tif_path: Path,
        output_directory: Path
):
    tif_map = Image.open(tif_path)
    output_directory.mkdir(exist_ok=True)
    for i, label_i in zip(HORIZONTAL_TILES, HORIZONTAL_TIF_LABELS):
        for j, label_j in zip(VERTICAL_TILES, VERTICAL_TIF_LABELS):
            crop = tif_map.crop((i * IMAGE_WIDTH, j * IMAGE_HEIGHT, (i + 1) * IMAGE_WIDTH, (j + 1) * IMAGE_HEIGHT))
            crop.save(output_directory / Path(f'{label_i}_{label_j}.png'))
