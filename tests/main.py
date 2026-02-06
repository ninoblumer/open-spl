from xl2 import XL2_SLM_Measurement

def main():
    projectname = "slm-test-01"
    filename = "2026-02-06_SLM_000"

    measurement = XL2_SLM_Measurement(projectname, filename)
    print(measurement.files["123_Log"].sections["Broadband LOG Results over whole log period"].df.loc[0, "LAeq_dt"])


if __name__ == '__main__':
    main()
