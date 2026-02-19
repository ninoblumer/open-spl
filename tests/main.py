from slm.meter import Meter, MovingMeter, AccumulatingMeter
from util.xl2 import XL2_SLM_Measurement
from pathlib import Path

import soundfile as sf
import numpy as np

from slm.engine import Engine
from slm.file_controller import FileController

REFERENCE_PRESSURE = 20e-6

def main():
    projectname = "slm-test-01"
    filename = "2026-02-06_SLM_000"

    measurement = XL2_SLM_Measurement(projectname, filename)
    print(measurement.files["123_Log"].sections["Broadband LOG Results over whole log period"].df.loc[0, "LAFmax_dt"])



    p = Path("data") / projectname
    files = list(p.glob(f"{filename}_Audio_*.wav"))
    filepath = files[0]

    data, fs = sf.read(str(filepath))

    controller = FileController(filename=filepath, blocksize=1024)
    # controller.set_sensitivity(-26.0, "dB")
    # controller.set_sensitivity(50.1, "mV")
    sensitivity = 1 / (REFERENCE_PRESSURE * 10 ** (128.1 / 20))
    print(sensitivity)
    controller.set_sensitivity(sensitivity, unit="V")

    engine = Engine(controller=controller, dt=0.1)

    from slm.frequency_weighting import PluginZWeighting, PluginAWeighting
    bus_z = engine.add_bus('Z', PluginZWeighting)
    bus_a = engine.add_bus('A', PluginAWeighting)

    from slm.plugin import ReadMode
    from slm.time_weighting import PluginFastTimeWeighting, PluginSlowTimeWeighting

    laf = bus_a.add_plugin(PluginFastTimeWeighting, input=bus_a.frequency_weighting, zero_zi=True)
    laf_meter = bus_a.add_meter(name="=", plugin=laf, mtype=MovingMeter, block_fn=MovingMeter._block_fn_last, fifo_fn=MovingMeter._fifo_fn_last)
    laf_max = engine.add_plugin(PluginFastTimeWeighting, bus='A', input=bus_a.frequency_weighting, read_mode=ReadMode("max", np.max), zero_zi=True)

    las = bus_a.add_plugin(PluginSlowTimeWeighting, input=bus_a.frequency_weighting, zero_zi=True)
    las_max = engine.add_plugin(PluginSlowTimeWeighting, bus='A', input=bus_a.frequency_weighting,
                                read_mode=ReadMode("max", np.max), zero_zi=True)


    for bus in engine._busses.values():
        print(f"Bus {bus.name}")
        for meter in bus.meters:
            chain = meter.get_chain()
            print(" / ".join([str(element) for element in chain]))


    engine.run()

    engine.stop()


    # wav = sf.SoundFile(str(filepath))
    # data = wav.read()
    # peak = np.max(np.abs(data))
    # peak_db = 20 * np.log10(peak / (REFERENCE_PRESSURE * controller.sensitivity))
    # print(peak_db)















if __name__ == '__main__':
    main()
