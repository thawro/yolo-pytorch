import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.utils.image import make_grid


@dataclass
class BaseResult:
    def set_preds(self):
        pass

    def plot(self) -> dict[str, np.ndarray]:
        raise NotImplementedError()


def plot_results(
    results: list[BaseResult], plot_name: str, filepath: str | None = None, ncols: int = -1
) -> np.ndarray:
    n_rows = min(20, len(results))
    grids = []
    for i in range(n_rows):
        result = results[i]
        result.set_preds()
        plots = result.plot()
        result_plot = plots[plot_name]
        grids.append(result_plot)
    n_grids = len(grids)
    if ncols < 1:
        nrows = n_grids
    else:
        nrows = math.ceil(n_grids / ncols)
    final_grid = make_grid(grids, nrows=nrows, pad=20, match_size=True)
    if filepath is not None:
        im = Image.fromarray(final_grid)
        im.save(filepath)
    return final_grid
