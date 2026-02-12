from pathlib import Path

import numpy as np
import soundfile as sf

from slm.controller import Controller


class FileController(Controller):
    blocksize: int = property(lambda self: self._blocksize)
    samplerate: int = property(lambda self: self._samplerate)
    sensitivity: float = property(lambda self: self._sensitivity)
    done: bool = property(lambda self: self._done)

    def __init__(self, filename: str | Path, blocksize: int, overlap: int = 0):
        super().__init__()
        self._blocksize = blocksize
        # self._overlap = overlap
        self._filename = filename
        self._sensitivity = 1
        info = sf.info(str(self._filename))
        self._samplerate = info.samplerate
        self._stream = sf.blocks(str(self._filename), blocksize=self._blocksize, overlap=overlap)
        self._done = False

    def new_file(self, filename: str | Path, blocksize: int, overlap: int = 0):
        if not self.done:
            raise RuntimeError("File has not been finished.")

        self._done = False
        self._blocksize = blocksize
        # self._overlap = overlap
        self._filename = filename
        self._stream = sf.blocks(str(self._filename), blocksize=self._blocksize, overlap=overlap)

    def read_block(self) -> tuple[np.ndarray, int]:
        try:
            return next(self._stream), next(self._counter)
        except StopIteration as e:
            self._done = True
            raise e

    def calibrate(self, target_spl=94.0):
        raise NotImplementedError("Calibration not implemented.")
