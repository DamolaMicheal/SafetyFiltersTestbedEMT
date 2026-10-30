DQ0-EMT EPSILON SAFETY-VIOLATION HEATMAP
=========================================

Files
-----
case39_matpower.jl
gfl_cbf_ieee39_tools.jl
gfl_cbf_outer_control.jl
ieee39_main_run.jl
ieee39_model_parameters.jl
plot_emt_safety_filter.py
plot_emt_epsilon_heatmap.py

What changed
------------
1. EMT safety-filter settings are synchronized to:
   epsilon nominal = 0.015 s
   RoCoF dirty-derivative tau_d = 0.015 s
   proximal step fraction = 0.55
   local diagonal Gamma scaling for bus-dependent k_i = enabled

2. The GFL model now accepts an epsilon_override for each sweep case.

3. The heatmap uses exactly 20 epsilon cases total:
   - 19 logarithmically spaced values from 0.001 s to 1.0 s
   - exact epsilon = 0.015 s inserted
   - the ordinary 0.015-s filter-ON EMT run is reused, so it is not rerun
   - therefore only 19 additional short EMT simulations are needed

4. The additional sweep simulations stop at 1.5 s because the heatmap only
   uses the early transient. The ordinary filter-OFF and nominal filter-ON
   validation runs remain at the original full TF_SIM = 6.0 s.

Run
---
From this folder:

    julia ieee39_main_run.jl

The compact heatmap data are written to:

    plots/python_style_cbf/csv/epsilon_sweep/emt_epsilon_time_violation_map.csv

Additional sweep files:

    plots/python_style_cbf/csv/epsilon_sweep/emt_epsilon_sweep_summary.csv
    plots/python_style_cbf/csv/epsilon_sweep/emt_epsilon_sweep_settings.csv
    plots/python_style_cbf/csv/epsilon_sweep/emt_epsilon_sweep_values.csv

Plot only the epsilon heatmap with:

    python plot_emt_epsilon_heatmap.py

or rebuild all publication plots, including the heatmap, with:

    python plot_emt_safety_filter.py

Heatmap definition
------------------
For the common active safety side, Julia exports

    violation(t, epsilon) = max(0, -min_i h_i(t; epsilon))

across all GFL buses. Therefore:

    violation = 0  -> safe
    violation > 0  -> at least one GFL bus violates the active frequency bound

The Python plot uses:
    white                  = safe region
    pink/red               = unsafe region, darker means larger violation
    solid dark contour     = h = 0 safety boundary
    vertical dotted line   = disturbance time
    x-axis                 = 0 to 1.5 s
    y-axis                 = epsilon on a logarithmic scale
