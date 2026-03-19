import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Slow-test opt-in
# ---------------------------------------------------------------------------
# Tests marked @pytest.mark.slow are skipped by default.
# Run them with:  pytest --slow

def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False,
        help="run slow tests (skipped by default)",
    )

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skipped unless --slow is passed)",
    )

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="slow test — pass --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

from util.xl2 import XL2_SLM_File
from slm.constants import REFERENCE_PRESSURE

DATA_DIR = Path("data/slm-test-01")


def parse_fs_db(wav_path: Path) -> float:
    """Parse the FS peak level in dB from an XL2 WAV filename.

    e.g. '..._Audio_FS128.1dB(PK)_00.wav' → 128.1
    """
    match = re.search(r"FS([\d.]+)dB\(PK\)", Path(wav_path).name)
    if not match:
        raise ValueError(f"No FS annotation found in filename: {Path(wav_path).name}")
    return float(match.group(1))


def sensitivity_from_fs(fs_db: float) -> float:
    """Return the sensitivity value for controller.set_sensitivity(..., "V").

    Derivation: for a normalised WAV sample s, the acoustic pressure is
        p = s * 10^(fs_db/20) * P_ref
    so that read_db = 10·log10(mean(s²) / (P_ref·sensitivity)²) gives correct SPL
    when sensitivity = 1 / (10^(fs_db/20) · P_ref).
    """
    return 1.0 / (10 ** (fs_db / 20) * REFERENCE_PRESSURE)


class XL2Measurement:
    """One XL2 dataset: WAV file + parsed log and report reference files."""

    def __init__(self, name: str, data_dir: Path = DATA_DIR):
        self.name = name

        wavs = list(data_dir.glob(f"{name}_Audio_*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No WAV file found for {name} in {data_dir}")
        self.wav_path = wavs[0]

        self.fs_db = parse_fs_db(self.wav_path)
        self.sensitivity = sensitivity_from_fs(self.fs_db)

        info = sf.info(str(self.wav_path))
        self.samplerate = info.samplerate
        self.n_frames = info.frames

        log_files = list(data_dir.glob(f"{name}_123_Log.txt"))
        report_files = list(data_dir.glob(f"{name}_123_Report.txt"))
        self.log = XL2_SLM_File(log_files[0]) if log_files else None
        self.report = XL2_SLM_File(report_files[0]) if report_files else None

        rta_log_files = list(data_dir.glob(f"{name}_RTA_*_Log.txt"))
        rta_report_files = list(data_dir.glob(f"{name}_RTA_*_Report.txt"))
        self.rta_log = XL2_SLM_File(rta_log_files[0]) if rta_log_files else None
        self.rta_report = XL2_SLM_File(rta_report_files[0]) if rta_report_files else None

    def report_value(self, col: str) -> float:
        """Scalar metric from the broadband Report file."""
        return float(self.report.sections["Broadband Results"].df[col].iloc[0])

    def log_series(self, col: str) -> np.ndarray:
        """Per-interval time series from the broadband Log (excludes summary row)."""
        df = self.log.sections["Broadband LOG Results"].df
        return df[col].astype(float).values


# --------------------------------------------------------------------------- #
# Session-scoped fixtures — one per XL2 measurement set                       #
# --------------------------------------------------------------------------- #

@pytest.fixture
def report() -> bool:
    return False


@pytest.fixture(scope="session")
def meas_000():
    """SLM_000: 10 s, 1 kHz calibrator tone at 94 dB."""
    return XL2Measurement("2026-02-06_SLM_000")


@pytest.fixture(scope="session")
def meas_001():
    """SLM_001: 30 s, repeating level ramp — tests Leq accumulation."""
    return XL2Measurement("2026-02-06_SLM_001")


@pytest.fixture(scope="session")
def meas_003():
    """SLM_003: 10 s, multi-frequency; LA=90.3, LC=92.1, LZ=94.0."""
    return XL2Measurement("2026-02-06_SLM_003")


@pytest.fixture(scope="session")
def meas_004():
    """SLM_004: 10 s, low-level signal (~36–40 dB range)."""
    return XL2Measurement("2026-02-06_SLM_004")


@pytest.fixture(scope="session")
def meas_005():
    """SLM_005: 10 s, background noise — octave RTA only, no broadband log."""
    return XL2Measurement("2026-02-06_SLM_005")
