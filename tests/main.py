from util.xl2 import XL2_SLM_Measurement
from pathlib import Path

import soundfile as sf

def main():
    projectname = "slm-test-01"
    filename = "2026-02-06_SLM_000"

    # measurement = XL2_SLM_Measurement(projectname, filename)
    # print(measurement.files["123_Log"].sections["Broadband LOG Results over whole log period"].df.loc[0, "LAeq_dt"])

    p = Path("data") / projectname
    files = list(p.glob(f"{filename}_Audio_*.wav"))

    filepath = Path("data")/projectname/"test1.wav"
    if not filepath.exists():
        raise Exception

    data, fs = sf.read(str(filepath))

    print(fs)







if __name__ == '__main__':
    main()
