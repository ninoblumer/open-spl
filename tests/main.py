from slm.meter import Meter, MovingMeter, AccumulatingMeter
from slm.plugin_meter import PluginMeter
from util.xl2 import XL2_SLM_Measurement
from pathlib import Path

import soundfile as sf
import numpy as np

from slm.engine import Engine
from slm.file_controller import FileController
from slm.octave_band import PluginOctaveBand

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

    la = bus_a.frequency_weighting
    laf = bus_a.add_plugin(PluginFastTimeWeighting(input=la, zero_zi=True))
    laf_meter = laf.add_meter(MovingMeter(name="last, last, dt", parent=laf, block_fn=MovingMeter._block_fn_last, fifo_fn=MovingMeter._fifo_fn_last))
    laf_max_accum_meter = laf.add_meter(AccumulatingMeter(name="max, max", parent=laf, block_fn=np.max, comp_fn=np.max))
    la_max_dt = la.add_meter(MovingMeter(name="max, max, dt", parent=la, block_fn=np.max, fifo_fn=np.max))

    la_oct = bus_a.add_plugin(PluginOctaveBand(input=la, zero_zi=True, limits=(50, 5000), bands_per_oct=1))
    la_oct_f = bus_a.add_plugin(PluginFastTimeWeighting(input=la_oct, width=la_oct.n_bands, zero_zi=True))
    la_oct_mean_1s_meter = la_oct_f.add_meter(MovingMeter(name="mean, mean, 1s", parent=la_oct_f, width=la_oct_f.width, block_fn=np.mean, fifo_fn=np.mean, t=1.0))
    la_oct_max_accum_meter = la_oct_f.add_meter(AccumulatingMeter(name="max, max, T", parent=la_oct_f, block_fn=np.max, comp_fn=np.max))

    for bus in engine._busses.values():
        print(f"Bus {bus.name}")
        for plugin in bus.plugins:
            print(f"\t{plugin}")
            if isinstance(plugin, PluginMeter):
                for meter in plugin.meters.values():
                    # chain = meter.get_chain()
                    # print("\t\t"+" / ".join([str(element) for element in chain]))
                    print(f"\t\t{meter}")

    engine.run()

    engine.stop()


    # wav = sf.SoundFile(str(filepath))
    # data = wav.read()
    # peak = np.max(np.abs(data))
    # peak_db = 20 * np.log10(peak / (REFERENCE_PRESSURE * controller.sensitivity))
    # print(peak_db)















if __name__ == '__main__':
    main()
