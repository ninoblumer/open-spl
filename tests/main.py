from slm.meter import LastMovingMeter, MaxMovingMeter, MaxAccumulator
from slm.plugin_meter import PluginMeter
from slm.reporter import Reporter
from util.xl2 import XL2_SLM_Measurement
from pathlib import Path

from slm.engine import Engine
from slm.file_controller import FileController
from slm.octave_band import PluginOctaveBand
from slm.constants import REFERENCE_PRESSURE

def main():
    projectname = "slm-test-01"
    filename = "2026-02-06_SLM_000"

    measurement = XL2_SLM_Measurement(projectname, filename)
    print(measurement.files["123_Log"].sections["Broadband LOG Results over whole log period"].df.loc[0, "LAFmax_dt"])



    p = Path("data") / projectname
    files = list(p.glob(f"{filename}_Audio_*.wav"))
    filepath = files[0]

    controller = FileController(filename=filepath, blocksize=1024)
    sensitivity = 1 / (REFERENCE_PRESSURE * 10 ** (128.1 / 20))
    print(sensitivity)
    controller.set_sensitivity(sensitivity, unit="V")

    engine = Engine(controller=controller, dt=0.1)

    from slm.frequency_weighting import PluginZWeighting, PluginAWeighting
    bus_z = engine.add_bus('Z', PluginZWeighting)
    bus_a = engine.add_bus('A', PluginAWeighting)

    from slm.time_weighting import PluginFastTimeWeighting

    la = bus_a.frequency_weighting
    laf = bus_a.add_plugin(PluginFastTimeWeighting(input=la, zero_zi=True))
    laf_meter = laf.add_meter(LastMovingMeter(name="LAF", parent=laf))
    laf_max_accum_meter = laf.add_meter(MaxAccumulator(name="LAFmax", parent=laf))
    la_max_dt = la.add_meter(MaxMovingMeter(name="LAmax_dt", parent=la))

    la_oct = bus_a.add_plugin(PluginOctaveBand(input=la, zero_zi=True, limits=(50, 5000), bands_per_oct=1))
    la_oct_f = bus_a.add_plugin(PluginFastTimeWeighting(input=la_oct, width=la_oct.n_bands, zero_zi=True))
    la_oct_max_1s_meter = la_oct_f.add_meter(MaxMovingMeter(name="LAFmax_1s", parent=la_oct_f, t=1.0))
    la_oct_max_accum_meter = la_oct_f.add_meter(MaxAccumulator(name="LAFmax", parent=la_oct_f))

    for bus in engine._busses.values():
        print(f"Bus {bus.name}")
        for plugin in bus.plugins:
            print(f"\t{plugin}")
            if isinstance(plugin, PluginMeter):
                for meter in plugin.meters.values():
                    print(f"\t\t{meter}")

    reporter = Reporter(print_to_console=True)
    reporter.add_column("LAF", laf, "LAF")
    reporter.add_column("LAFmax", laf, "LAFmax")
    reporter.add_column("LAmax_dt", la, "LAmax_dt")
    reporter.add_column("LAFmax_1s", la_oct_f, "LAFmax_1s",
                        center_frequencies=la_oct.center_frequencies)
    reporter.add_column("LAFmax", la_oct_f, "LAFmax",
                        center_frequencies=la_oct.center_frequencies)
    engine.reporter = reporter

    engine.run()

    engine.stop()

    reporter.write("output/slm-test-01")


if __name__ == '__main__':
    main()
