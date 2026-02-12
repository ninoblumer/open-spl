from util.xl2 import XL2_SLM_Measurement
from pathlib import Path

import soundfile as sf
import numpy as np

from slm.engine import Engine
from slm.file_controller import FileController


def main():
    projectname = "slm-test-01"
    filename = "2026-02-06_SLM_000"

    measurement = XL2_SLM_Measurement(projectname, filename)
    print(measurement.files["123_Log"].sections["Broadband LOG Results over whole log period"].df.loc[0, "LAeq_dt"])



    p = Path("data") / projectname
    files = list(p.glob(f"{filename}_Audio_*.wav"))
    filepath = files[0]

    data, fs = sf.read(str(filepath))

    controller = FileController(filename=filepath, blocksize=1024)
    engine = Engine(controller=controller)

    from slm.plugins.frequency_weighting import PluginZWeighting, PluginAWeighting
    bus_z = engine.add_bus('Z', PluginZWeighting)
    bus_a = engine.add_bus('A', PluginAWeighting)

    from slm.plugins.plugin import ReadMode
    from slm.plugins.time_weighting import PluginFastTimeWeighting, PluginTimeWeighting

    laf = bus_a.add_plugin(PluginFastTimeWeighting, input=bus_a.frequency_weighting)
    laf_max = engine.add_plugin(PluginFastTimeWeighting, bus='A', input=bus_a.frequency_weighting, read_mode=ReadMode("max", np.max))

    engine.run()













if __name__ == '__main__':
    main()
