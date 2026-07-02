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

    def _rta_results_section(self):
        """The RTA report's results section (tolerating a trailing space in the header)."""
        for key, section in self.rta_report.sections.items():
            if key.strip() == "RTA Results":
                return section
        raise KeyError("No 'RTA Results' section in RTA report")

    def rta_lzeq(self) -> np.ndarray:
        """Whole-measurement 1/1- or 1/3-octave LZeq spectrum (one value per band)."""
        return self._rta_results_section().df.loc["LZeq"].astype(float).values

    def rta_metric(self) -> str:
        """Metric name whose band set matches this recording's XL2 RTA spectrum.

        The XL2 RTA is 1/1-octave (12 bands, 8 Hz–16 kHz) or 1/3-octave (36 bands,
        6.3 Hz–20 kHz); pick the metric so the SLM produces the same bands in the
        same order (they can then be compared position-by-position).
        """
        n_bands = len(self.rta_lzeq())
        if n_bands <= 13:
            return "LZeq:bands:8-16000"          # 1/1-octave, 12 bands
        return "LZeq:bands:1/3:6.3-20000"        # 1/3-octave, 36 bands


# --------------------------------------------------------------------------- #
# Cross-dataset recording discovery (XL2 integration suite)                    #
# --------------------------------------------------------------------------- #

# Both reference sets: slm-test-01 is electrical (signal generator → XL2), slm-test-02
# is acoustic (microphone recordings). SLM_002 is excluded from the validation suite
# in both sets (electrical SLM_002 is an unused low-level chirp; acoustic SLM_002 is a
# quiet background dominated by sub-audio energy that flat Z legitimately integrates).
XL2_DATA_DIRS = (Path("data/slm-test-01"), Path("data/slm-test-02"))
XL2_EXCLUDE_KEYS = frozenset({"SLM_002"})


def discover_xl2_recordings() -> list[tuple[Path, str, str]]:
    """Find every ``SLM_NNN`` recording across both datasets (minus the exclusions).

    Returns ``(data_dir, name, key)`` tuples, e.g.
    ``(Path('data/slm-test-01'), '2026-02-06_SLM_000', 'SLM_000')``.
    """
    recordings: list[tuple[Path, str, str]] = []
    for data_dir in XL2_DATA_DIRS:
        for wav in sorted(data_dir.glob("*_Audio_*.wav")):
            match = re.search(r"(\d{4}-\d{2}-\d{2}_SLM_\d{3})", wav.name)
            if not match:
                continue
            name = match.group(1)
            key = name.split("_", 1)[1]        # 'SLM_000'
            if key in XL2_EXCLUDE_KEYS:
                continue
            recordings.append((data_dir, name, key))
    return recordings


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
