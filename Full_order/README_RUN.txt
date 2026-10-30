IEEE-39 dq0-EMT dynamic local safety-filter study

Run from a fresh Julia session:

    cd(raw"C:\Users\ajeye\OneDrive\Desktop\CBF_SafetyFilters_Project\Full_Order")
    include("ieee39_main_run.jl")

A single include automatically performs:
  1. FILTER OFF: original EMT plant with stock ActivePowerPI at every GFL.
  2. EMT sensitivity identification and local proximal-gain design.
  3. FILTER ON: the same EMT plant with ActivePowerPISafetyFilter at every GFL.

No environment variable and no manual ON/OFF toggle are used.

CSV folders:
  plots/csv/filter_off/
  plots/csv/filter_on/

Matched publication-comparison CSV folders:
  plots/python_style_cbf/csv/filter_off/
  plots/python_style_cbf/csv/filter_on/

CBF design:
  plots/python_style_cbf/csv/design/

Then run:
  plot_emt_safety_filter_publication.py

The Python script only replots Julia-exported CSVs; it does not recompute the EMT model.
