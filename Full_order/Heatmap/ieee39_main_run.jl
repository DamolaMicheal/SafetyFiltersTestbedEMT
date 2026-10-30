#!/usr/bin/env julia

include("ieee39_model_parameters.jl")
include("gfl_cbf_outer_control.jl")
include("gfl_cbf_ieee39_tools.jl")

# -----------------------------------------------------------------------------
# EMT epsilon-sensitivity sweep for the safety-violation heatmap
# -----------------------------------------------------------------------------
# Exactly 20 epsilon cases are used in the heatmap. Nineteen are logarithmically
# spaced from 1 ms to 1 s and epsilon = 0.015 s is inserted explicitly. The
# nominal 0.015-s ON case is reused from the regular validation run, so only
# 19 additional short EMT simulations are required for the sweep.
const RUN_EMT_EPSILON_HEATMAP_SWEEP = true
const EMT_EPSILON_SWEEP_VALUES_S = sort(unique(vcat(
    collect(exp.(range(log(0.005), log(1.0), length = 19))),
    [0.015],
)))
length(EMT_EPSILON_SWEEP_VALUES_S) == 20 ||
    error("The EMT epsilon sweep must contain exactly 20 values.")

const EMT_EPSILON_SWEEP_TFINAL_S = 1.5
const EMT_EPSILON_SWEEP_EXPORT_DT_S = 0.01
const EMT_EPSILON_SWEEP_VIOLATION_TOL_HZ = 1e-6

# Small epsilon makes the local filter state stiff because dz/dt scales as 1/epsilon.
# Keep the same EMT tolerances as the nominal validation, but force the IDA time
# step to resolve the selected filter time scale. If a run still terminates early,
# retry with progressively smaller step sizes.
const EMT_EPSILON_SWEEP_DTMAX_DIVISORS = (5.0, 10.0, 20.0)
const EMT_EPSILON_SWEEP_ENDTIME_TOL_S = 5e-4

function sg_machine_params()
    Dict{Symbol,Float64}(
        :Rs => 0.0025,
        :Xd => 1.80, :Xq => 1.70,
        :Xd_p => 0.30, :Xq_p => 0.55,
        :Xd_pp => 0.25, :Xl => 0.20,
        :Tdo_p => 8.00, :Tqo_p => 0.40,
        :Tdo_pp => 0.03, :Tqo_pp => 0.05,
    )
end

function sg_avr_params()
    Dict{Symbol,Float64}(
        :Ka => 20.0, :Ke => 1.0, :Kf => 0.063,
        :Ta => 0.2, :Te => 0.314, :Tf => 0.35, :Tr => 0.001,
    )
end

function generator_mbase_map()
    mbase = Dict{Int,Float64}(b => 1000.0 for b in 30:39)
    mbase[34] = 600.0
    mbase[39] = 10000.0
    return mbase
end

function build_initialized_emt_system(
    case_path::String,
    pfdir::String;
    write_pf::Bool = true,
    show_reference_summary::Bool = true,
    gfl_filter_enabled::Bool = false,
    calibration_k_override::Union{Nothing,Float64} = nothing,
    epsilon_override::Union{Nothing,Float64} = nothing,
)
    sys = System(case_path)
    set_units_base_system!(sys, "SYSTEM_BASE")
    PSY.set_units_base_system!(sys, "SYSTEM_BASE")

    enforce_load_bases!(sys; base_mva = SYS_BASE_MVA)
    enforce_min_line_r!(sys)

    pf_data = PowerFlowData(ACPowerFlow(), sys)
    solve_powerflow!(pf_data)
    if write_pf
        write_pf_results(
            pf_data,
            pfdir;
            basename = "powerflow_results",
            sys_base_mva = SYS_BASE_MVA,
        )
    end
    update_system_voltages!(sys, pf_data)

    ensure_perturbable_loads_all_buses!(sys; base_mva = SYS_BASE_MVA)
    loads_to_constant_impedance!(sys)
    add_dynamic_lines!(sys)

    sync_static_gens_to_pf!(sys, pf_data; buses = collect(30:39))
    mbase = generator_mbase_map()
    for b in 30:39
        PSY.set_base_power!(gen_at_bus(sys, b), mbase[b])
    end

    if show_reference_summary
        print_inverter_reference_summary(
            sys;
            gfm_buses = GFM_BUSES,
            gfl_buses = GFL_BUSES,
        )
    end

    for b in SG_BUSES
        attach_synchronous_generator!(
            sys,
            gen_at_bus(sys, b);
            H = getH(b),
            D = getDsg(b),
            droop_R = R_SG_PU,
            machine_params = sg_machine_params(),
            avr_params = sg_avr_params(),
        )
    end

    for b in GFM_BUSES
        attach_grid_forming_inverter!(
            sys,
            gen_at_bus(sys, b);
            droop_R = R_GFM_PU,
            Ta = getGfmTa(b),
            kd = getGfmKd(b),
            kω = getGfmKw(b),
            kpv = 1.0,
            kiv = 1000.0,
            kpc = 3.0,
            kic = 600.0,
            ωad = 2000.0,
        )
    end

    use_supervisor = gfl_filter_enabled || !isnothing(calibration_k_override)

    for b in GFL_BUSES
        attach_grid_following_inverter!(
            sys,
            gen_at_bus(sys, b);
            kω_droop = getGflKw(b),
            pll_kp = getGflPllKp(b),
            pll_ki = getGflPllKi(b),
            kp_q = 2.0,
            ki_q = 30.0,
            ωf_q = 500.0,
            kic = 300.0,
            use_safety_supervisor = use_supervisor,
            filter_enabled = gfl_filter_enabled,
            k_prox_override = calibration_k_override,
            epsilon_override = epsilon_override,
        )
    end

    if show_reference_summary
        print_dynamic_reference_summary(sys; buses = collect(30:39))
    end

    return sys, pf_data
end

function build_perturbations(sys::System, realized_steps::Vector{RealizedLoadStep})
    perturbations = PowerSimulationsDynamics.Perturbation[]

    if LOAD_CHANGE_ENABLE
        ensure_perturbable_loads_all_buses!(sys; base_mva = SYS_BASE_MVA)
        append!(perturbations, make_load_changes_from_event_schedule(sys, realized_steps))
    end

    if BUS_FAULT_ENABLE
        append!(perturbations, make_fault_perturbations(sys))
    end

    return perturbations
end

function export_base_csvs(
    results,
    sys::System,
    pf_data::PowerFlowData,
    realized_steps::Vector{RealizedLoadStep},
    csvdir::String,
    t_step_primary::Float64,
    load_event_times::Vector{Float64},
)
    EXPORT_SIM_CSVS || return nothing

    freqcsvdir = joinpath(csvdir, "states", "freq")
    pgcsvdir = joinpath(csvdir, "states", "pg")
    mkpath(freqcsvdir)
    mkpath(pgcsvdir)

    for b in 30:39
        save_frequency_csv_for_bus(results, sys, b, freqcsvdir; t_step = t_step_primary)
        save_rotor_angle_csv_for_bus(results, sys, b, csvdir; t_step = t_step_primary)
        save_voltage_csv_for_bus(results, sys, b, csvdir; t_step = t_step_primary)
        save_terminal_pq_csv_for_bus(results, sys, b, csvdir; event_times = load_event_times)
    end

    for b in SG_BUSES
        save_mechanical_power_csv_for_bus(results, sys, b, pgcsvdir; t_step = t_step_primary)
    end

    for b in GFM_BUSES
        save_gfm_power_csv_for_bus(results, sys, b, csvdir; t_step = t_step_primary)
    end

    for b in GFL_BUSES
        save_gfl_pref_eff_csv_for_bus(results, sys, b, csvdir; t_step = t_step_primary)
    end

    net_cache = nothing
    if SAVE_DIAGNOSTIC_NETWORK_PQ
        net_cache = network_bus_pq_dev(
            results,
            sys,
            pf_data,
            realized_steps;
            buses = NETWORK_PQ_ALL_BUSES,
        )
        for b in NETWORK_PQ_EXPORT_BUSES
            save_network_pq_csv(
                results,
                sys,
                pf_data,
                b,
                csvdir,
                realized_steps;
                net_cache = net_cache,
            )
        end
    end

    export_extra_load_bus_emt_csvs(
        results,
        sys,
        pf_data,
        csvdir,
        realized_steps;
        t_step = t_step_primary,
        net_cache = net_cache,
    )

    return nothing
end

function run_emt_case(
    case_path::String,
    plotsdir::String,
    realized_steps::Vector{RealizedLoadStep},
    load_event_times::Vector{Float64},
    t_step_primary::Float64;
    filter_enabled::Bool,
    show_system::Bool = false,
    epsilon_override::Union{Nothing,Float64} = nothing,
)
    tag = filter_enabled ? "filter_on" : "filter_off"
    casedir = joinpath(plotsdir, tag)
    pfdir = joinpath(casedir, "pf")
    csvdir = joinpath(plotsdir, "csv", tag)
    mkpath(pfdir)
    mkpath(csvdir)

    println("\n" * "="^90)
    println(filter_enabled ? "EMT VALIDATION: SAFETY FILTER ON" : "EMT BASELINE: SAFETY FILTER OFF")
    println("="^90)

    sys, pf_data = build_initialized_emt_system(
        case_path,
        pfdir;
        write_pf = true,
        show_reference_summary = true,
        gfl_filter_enabled = filter_enabled,
        calibration_k_override = nothing,
        epsilon_override = epsilon_override,
    )

    if show_system
        show(sys)
        println()
    end

    perturbations = build_perturbations(sys, realized_steps)
    @info "Starting EMT simulation" case=tag tf=TF_SIM n_perturbations=length(perturbations)

    sim = Simulation(
        ResidualModel,
        sys,
        mktempdir(),
        (0.0, TF_SIM),
        perturbations;
        console_level = Logging.Info,
    )

    try
        report_small_signal(sim)
    catch e
        @warn "Small-signal analysis skipped" case=tag err=string(e)
    end

    execute!(
        sim,
        IDA(linear_solver = :Dense, max_order = SOLVER_MAX_ORDER);
        abstol = SOLVER_ABSTOL,
        reltol = SOLVER_RELTOL,
        dtinit = SOLVER_INIT_DT,
        dtmax = SOLVER_DTMAX,
    )

    results = read_results(sim)

    export_base_csvs(
        results,
        sys,
        pf_data,
        realized_steps,
        csvdir,
        t_step_primary,
        load_event_times,
    )

    print_final_steady_state_freq(results, sys; buses = collect(30:39))
    @info "EMT case complete" case=tag csv=csvdir

    return (
        tag = tag,
        sys = sys,
        pf_data = pf_data,
        results = results,
        csvdir = csvdir,
    )
end


function _epsilon_sweep_time_grid(t_step_primary::Float64)
    tf = EMT_EPSILON_SWEEP_TFINAL_S
    dt_out = EMT_EPSILON_SWEEP_EXPORT_DT_S
    tg = collect(0.0:dt_out:tf)
    if isempty(tg) || tg[end] < tf - 1e-12
        push!(tg, tf)
    end
    if 0.0 <= t_step_primary <= tf && !any(abs.(tg .- t_step_primary) .<= 1e-12)
        push!(tg, t_step_primary)
        sort!(tg)
    end
    return Float64.(tg)
end

function _run_short_emt_epsilon_case(
    case_path::String,
    sweep_pfdir::String,
    realized_steps::Vector{RealizedLoadStep},
    epsilon_s::Float64,
)
    epsilon_s > 0.0 || error("epsilon_s must be strictly positive.")

    last_error = nothing
    last_t_end = NaN

    for (attempt, divisor) in enumerate(EMT_EPSILON_SWEEP_DTMAX_DIVISORS)
        # Resolve the dynamic-filter time scale. For example, epsilon=1 ms gives
        # dtmax=0.2 ms on the first attempt, 0.1 ms on the second, and 0.05 ms
        # on the third. Larger epsilon values remain capped by SOLVER_DTMAX.
        dtmax_case = min(
            Float64(SOLVER_DTMAX),
            Float64(epsilon_s) / Float64(divisor),
        )
        dtmax_case = max(dtmax_case, 1e-6)

        dtinit_case = min(
            Float64(SOLVER_INIT_DT),
            dtmax_case / 10.0,
        )
        dtinit_case = max(dtinit_case, 1e-8)

        @info(
            "Starting short EMT epsilon-sweep simulation",
            epsilon_s = epsilon_s,
            attempt = attempt,
            dtinit = dtinit_case,
            dtmax = dtmax_case,
            tf = EMT_EPSILON_SWEEP_TFINAL_S,
        )

        try
            # A fresh system is required for every retry.
            sys, _ = build_initialized_emt_system(
                case_path,
                sweep_pfdir;
                write_pf = false,
                show_reference_summary = false,
                gfl_filter_enabled = true,
                calibration_k_override = nothing,
                epsilon_override = epsilon_s,
            )

            perturbations = build_perturbations(sys, realized_steps)

            sim = Simulation(
                ResidualModel,
                sys,
                mktempdir(),
                (0.0, EMT_EPSILON_SWEEP_TFINAL_S),
                perturbations;
                console_level = Logging.Warn,
            )

            execute!(
                sim,
                IDA(linear_solver = :Dense, max_order = SOLVER_MAX_ORDER);
                abstol = SOLVER_ABSTOL,
                reltol = SOLVER_RELTOL,
                dtinit = dtinit_case,
                dtmax = dtmax_case,
            )

            results = read_results(sim)

            # Do not silently accept a partially integrated EMT case. Check the
            # actual recovered GFL signal horizon before it is used by the heatmap.
            sig = safety_filter_signals_at_bus(
                results,
                sys,
                first(GFL_BUSES);
                for_csv = false,
            )
            sig === nothing &&
                error("Could not recover GFL safety-filter signals after the EMT run.")

            tvals = Float64.(sig.t)
            finite_t = tvals[isfinite.(tvals)]
            isempty(finite_t) &&
                error("The EMT run produced no finite output times.")

            t_end = maximum(finite_t)
            last_t_end = t_end

            if t_end >= EMT_EPSILON_SWEEP_TFINAL_S - EMT_EPSILON_SWEEP_ENDTIME_TOL_S
                if attempt > 1
                    @info(
                        "EMT epsilon case succeeded after retry",
                        epsilon_s = epsilon_s,
                        attempt = attempt,
                        t_end = t_end,
                        dtmax = dtmax_case,
                    )
                end
                return (sys = sys, results = results)
            end

            @warn(
                "EMT epsilon case terminated before the requested heatmap horizon; retrying with a smaller solver step",
                epsilon_s = epsilon_s,
                attempt = attempt,
                t_end = t_end,
                requested_t_end = EMT_EPSILON_SWEEP_TFINAL_S,
                dtmax = dtmax_case,
            )

            last_error = ErrorException(
                "epsilon=$(epsilon_s) s terminated at t=$(t_end) s."
            )
        catch err
            last_error = err
            @warn(
                "EMT epsilon case failed; retrying with a smaller solver step",
                epsilon_s = epsilon_s,
                attempt = attempt,
                dtmax = dtmax_case,
                err = string(err),
            )
        end

        GC.gc()
    end

    error(
        "EMT epsilon sweep could not complete epsilon=$(epsilon_s) s to " *
        "t=$(EMT_EPSILON_SWEEP_TFINAL_S) s after " *
        "$(length(EMT_EPSILON_SWEEP_DTMAX_DIVISORS)) attempts. " *
        "Last recovered end time=$(last_t_end) s. Last error=$(last_error). " *
        "The heatmap will not extrapolate this case."
    )
end


function _epsilon_sweep_interp_finite(
    t_raw::AbstractVector,
    y_raw::AbstractVector,
    tg::Vector{Float64};
    label::AbstractString = "signal",
)
    length(t_raw) == length(y_raw) ||
        error("$label has mismatched time/value lengths.")

    keep = [
        isfinite(Float64(t_raw[i])) && isfinite(Float64(y_raw[i]))
        for i in eachindex(t_raw)
    ]
    count(keep) >= 2 ||
        error("$label has fewer than two finite samples.")

    tt = Float64.(t_raw[keep])
    yy = Float64.(y_raw[keep])

    ord = sortperm(tt)
    tt = tt[ord]
    yy = yy[ord]

    # Collapse duplicate time stamps deterministically by retaining the
    # last finite value at each time.
    tt_unique = Float64[]
    yy_unique = Float64[]
    for i in eachindex(tt)
        if !isempty(tt_unique) && abs(tt[i] - tt_unique[end]) <= 1e-12
            yy_unique[end] = yy[i]
        else
            push!(tt_unique, tt[i])
            push!(yy_unique, yy[i])
        end
    end

    length(tt_unique) >= 2 ||
        error("$label has fewer than two unique finite time samples.")

    # The DAE solver may terminate a few floating-point ulps before/after
    # the exact requested endpoint. Clamp only small endpoint mismatches.
    endpoint_tol = max(2.0 * EMT_EPSILON_SWEEP_EXPORT_DT_S, 5e-4)

    tg[1] >= tt_unique[1] - endpoint_tol ||
        error(
            "$label begins too late for the sweep grid: " *
            "source=$(tt_unique[1]) s, requested=$(tg[1]) s."
        )
    tg[end] <= tt_unique[end] + endpoint_tol ||
        error(
            "$label ends too early for the sweep grid: " *
            "source=$(tt_unique[end]) s, requested=$(tg[end]) s."
        )

    tq = clamp.(tg, tt_unique[1], tt_unique[end])
    out = interp1_linear(tt_unique, yy_unique, tq)

    all(isfinite, out) ||
        error("$label contains non-finite values after sweep resampling.")

    return out
end


function _epsilon_sweep_case_rows(
    results,
    sys::System,
    epsilon_s::Float64,
    side::String,
    tg::Vector{Float64},
)
    side in ("lower", "upper") || error("Sweep side must be lower or upper.")

    nF = length(GFL_BUSES)
    nt = length(tg)
    barrier_by_bus = Matrix{Float64}(undef, nF, nt)
    freq_by_bus = Matrix{Float64}(undef, nF, nt)

    for (j, bus) in enumerate(GFL_BUSES)
        sig = safety_filter_signals_at_bus(results, sys, bus; for_csv = false)
        sig === nothing && error("Could not recover safety-filter signals at GFL bus $bus.")

        barrier_raw = side == "lower" ? sig.h_lo_hz : sig.h_hi_hz
        barrier_by_bus[j, :] .= _epsilon_sweep_interp_finite(
            sig.t,
            barrier_raw,
            tg;
            label = "GFL bus $bus barrier at epsilon=$epsilon_s",
        )
        freq_by_bus[j, :] .= _epsilon_sweep_interp_finite(
            sig.t,
            sig.freq_hz,
            tg;
            label = "GFL bus $bus frequency at epsilon=$epsilon_s",
        )
    end

    worst_barrier = Vector{Float64}(undef, nt)
    worst_bus = Vector{Int}(undef, nt)
    worst_frequency = Vector{Float64}(undef, nt)
    violation = Vector{Float64}(undef, nt)
    safe = Vector{Bool}(undef, nt)

    for k in 1:nt
        j = argmin(@view barrier_by_bus[:, k])
        h = barrier_by_bus[j, k]
        worst_barrier[k] = h
        worst_bus[k] = GFL_BUSES[j]
        worst_frequency[k] = freq_by_bus[j, k]
        violation[k] = max(0.0, -h)
        safe[k] = violation[k] <= EMT_EPSILON_SWEEP_VIOLATION_TOL_HZ
    end

    rows = DataFrame(
        epsilon_s = fill(epsilon_s, nt),
        t_s = tg,
        active_safety_bound = fill(side, nt),
        worst_gfl_barrier_hz = worst_barrier,
        safety_violation_hz = violation,
        worst_gfl_frequency_hz = worst_frequency,
        worst_bus = worst_bus,
        safe = safe,
    )

    violation_duration_s = sum(.!safe) * EMT_EPSILON_SWEEP_EXPORT_DT_S
    summary = (
        epsilon_s = epsilon_s,
        min_barrier_hz = minimum(worst_barrier),
        max_violation_hz = maximum(violation),
        violation_duration_s = violation_duration_s,
        min_gfl_frequency_hz = minimum(freq_by_bus),
        max_gfl_frequency_hz = maximum(freq_by_bus),
        safe = minimum(worst_barrier) >= -EMT_EPSILON_SWEEP_VIOLATION_TOL_HZ,
    )

    return rows, summary
end

function run_emt_epsilon_heatmap_sweep(
    case_path::String,
    plotsdir::String,
    safetycsvdir::String,
    realized_steps::Vector{RealizedLoadStep},
    t_step_primary::Float64,
    nominal_on,
)
    validate_cbf_design!()

    sweepdir = joinpath(safetycsvdir, "epsilon_sweep")
    sweep_pfdir = joinpath(plotsdir, "epsilon_sweep", "pf")
    mkpath(sweepdir)
    mkpath(sweep_pfdir)

    tg = _epsilon_sweep_time_grid(t_step_primary)

    # Use one common safety side for the entire sweep. For the present
    # load-increase case, ACTIVE_SAFETY_BOUND="auto" resolves to "lower".
    side = active_safety_side(nominal_on.results, nominal_on.sys)

    all_rows = DataFrame(
        epsilon_s = Float64[],
        t_s = Float64[],
        active_safety_bound = String[],
        worst_gfl_barrier_hz = Float64[],
        safety_violation_hz = Float64[],
        worst_gfl_frequency_hz = Float64[],
        worst_bus = Int[],
        safe = Bool[],
    )

    summaries = DataFrame(
        epsilon_s = Float64[],
        min_barrier_hz = Float64[],
        max_violation_hz = Float64[],
        violation_duration_s = Float64[],
        min_gfl_frequency_hz = Float64[],
        max_gfl_frequency_hz = Float64[],
        safe = Bool[],
        reused_nominal_case = Bool[],
    )

    println("\n" * "="^94)
    println("dq0-EMT EPSILON SWEEP FOR SAFETY-VIOLATION HEATMAP")
    println("="^94)
    @printf("Number of epsilon cases: %d\n", length(EMT_EPSILON_SWEEP_VALUES_S))
    @printf("Heatmap time window   : [0, %.3f] s\n", EMT_EPSILON_SWEEP_TFINAL_S)
    @printf("CSV time resolution   : %.4f s\n", EMT_EPSILON_SWEEP_EXPORT_DT_S)
    println("Common safety side    : $side")
    println("The epsilon=0.015 s nominal ON case is reused; it is not simulated twice.")
    println("="^94)

    for (idx, epsilon_s_raw) in enumerate(EMT_EPSILON_SWEEP_VALUES_S)
        epsilon_s = Float64(epsilon_s_raw)
        reuse_nominal = isapprox(epsilon_s, GFL_CBF_EPS; atol = 1e-12, rtol = 0.0)

        @printf("[%2d/%2d] epsilon = %.9f s%s\n",
                idx, length(EMT_EPSILON_SWEEP_VALUES_S), epsilon_s,
                reuse_nominal ? "  [reuse nominal ON case]" : "")

        case_data = if reuse_nominal
            (sys = nominal_on.sys, results = nominal_on.results)
        else
            _run_short_emt_epsilon_case(
                case_path,
                sweep_pfdir,
                realized_steps,
                epsilon_s,
            )
        end

        rows, summary = _epsilon_sweep_case_rows(
            case_data.results,
            case_data.sys,
            epsilon_s,
            side,
            tg,
        )
        append!(all_rows, rows)
        push!(summaries, (
            summary.epsilon_s,
            summary.min_barrier_hz,
            summary.max_violation_hz,
            summary.violation_duration_s,
            summary.min_gfl_frequency_hz,
            summary.max_gfl_frequency_hz,
            summary.safe,
            reuse_nominal,
        ))

        @printf(
            "          min h = %+.6f Hz | max violation = %.6f Hz | T_viol = %.4f s | %s\n",
            summary.min_barrier_hz,
            summary.max_violation_hz,
            summary.violation_duration_s,
            summary.safe ? "SAFE" : "VIOLATION",
        )

        reuse_nominal || GC.gc()
    end

    # Verify the exported heatmap table is rectangular before writing.
    expected_rows_per_case = length(tg)
    for epsilon_s in EMT_EPSILON_SWEEP_VALUES_S
        mask = abs.(all_rows.epsilon_s .- Float64(epsilon_s)) .<= 1e-12
        nrows_case = count(mask)
        nrows_case == expected_rows_per_case ||
            error(
                "Non-rectangular epsilon sweep export at epsilon=$(epsilon_s): " *
                "expected $(expected_rows_per_case) rows, got $(nrows_case)."
            )
    end

    all(isfinite, all_rows.worst_gfl_barrier_hz) ||
        error("Non-finite barrier values remain in the epsilon sweep export.")
    all(isfinite, all_rows.safety_violation_hz) ||
        error("Non-finite violation values remain in the epsilon sweep export.")

    map_path = joinpath(sweepdir, "emt_epsilon_time_violation_map.csv")
    summary_path = joinpath(sweepdir, "emt_epsilon_sweep_summary.csv")
    CSV.write(map_path, all_rows)
    CSV.write(summary_path, summaries)

    settings = DataFrame(
        quantity = [
            "number_of_epsilon_cases",
            "epsilon_min_s",
            "epsilon_max_s",
            "sweep_tfinal_s",
            "sweep_export_dt_s",
            "violation_tolerance_hz",
            "frequency_lower_hz",
            "frequency_upper_hz",
            "alpha_lower",
            "alpha_upper",
            "rocof_tau_s",
            "prox_step_fraction",
            "active_safety_bound",
        ],
        value = string.([
            length(EMT_EPSILON_SWEEP_VALUES_S),
            minimum(EMT_EPSILON_SWEEP_VALUES_S),
            maximum(EMT_EPSILON_SWEEP_VALUES_S),
            EMT_EPSILON_SWEEP_TFINAL_S,
            EMT_EPSILON_SWEEP_EXPORT_DT_S,
            EMT_EPSILON_SWEEP_VIOLATION_TOL_HZ,
            GFL_CBF_FLOOR_HZ,
            GFL_CBF_CEIL_HZ,
            GFL_CBF_ALPHA_LO,
            GFL_CBF_ALPHA_HI,
            ROCOF_TAU_D,
            GFL_CBF_PROX_STEP_FRACTION,
            side,
        ]),
    )
    CSV.write(joinpath(sweepdir, "emt_epsilon_sweep_settings.csv"), settings)

    values_df = DataFrame(
        case_index = collect(1:length(EMT_EPSILON_SWEEP_VALUES_S)),
        epsilon_s = Float64.(EMT_EPSILON_SWEEP_VALUES_S),
    )
    CSV.write(joinpath(sweepdir, "emt_epsilon_sweep_values.csv"), values_df)

    println("\nEMT epsilon heatmap CSV : $map_path")
    println("EMT epsilon summary CSV : $summary_path")
    println("="^94 * "\n")

    return (
        map_path = map_path,
        summary_path = summary_path,
        side = side,
        n_cases = length(EMT_EPSILON_SWEEP_VALUES_S),
    )
end

function main(; show_system::Bool = false)
    global_logger(ConsoleLogger(stderr, Logging.Info))
    validate_partitions!(SG_BUSES, GFM_BUSES, GFL_BUSES)

    scriptdir = @__DIR__
    plotsdir = joinpath(scriptdir, "plots")
    safetycsvdir = joinpath(plotsdir, "python_style_cbf", "csv")
    design_dir = joinpath(safetycsvdir, "design")
    calibration_pfdir = joinpath(plotsdir, "cbf_calibration", "pf")

    mkpath(plotsdir)
    mkpath(safetycsvdir)
    mkpath(design_dir)
    mkpath(calibration_pfdir)

    realized_steps = LOAD_CHANGE_ENABLE ?
        build_realized_load_steps_from_events() : RealizedLoadStep[]
    load_event_times = event_times_from_realized_steps(realized_steps)
    t_step_primary = isempty(load_event_times) ? 0.25 : minimum(load_event_times)

    if LOAD_CHANGE_ENABLE
        print_load_change_schedule(realized_steps)
    end
    if BUS_FAULT_ENABLE
        print_fault_disturbance_schedule()
    end

    save_safety_disturbance_schedule(realized_steps, safetycsvdir)

    case_path = joinpath(scriptdir, "modified_ieee39.m")
    write_modified_ieee39_case(case_path)
    @info "Wrote MATPOWER case file" file=case_path

    # 1) Original EMT model: stock GFL ActivePowerPI, no CBF states.
    off = run_emt_case(
        case_path,
        plotsdir,
        realized_steps,
        load_event_times,
        t_step_primary;
        filter_enabled = false,
        show_system = show_system,
    )

    # 2) Identify the EMT local sensitivity and proximal gains for this model.
    println("\n" * "="^90)
    println("CBF DESIGN: EMT SENSITIVITY IDENTIFICATION")
    println("="^90)
    estimate_emt_gfl_sensitivity!(case_path, calibration_pfdir, design_dir)
    validate_cbf_design!()

    # Export the OFF case after Gamma is available so its diagnostic intervals
    # use the same k_i as the ON case, while z_i remains identically zero.
    off_safety_dir = save_safety_filter_csvs(
        off.results,
        off.sys,
        safetycsvdir;
        t_step = t_step_primary,
    )
    print_safety_filter_summary(off.results, off.sys)

    # 3) Same EMT model with the dynamic local safety supervisor added at GFLs.
    on = run_emt_case(
        case_path,
        plotsdir,
        realized_steps,
        load_event_times,
        t_step_primary;
        filter_enabled = true,
        show_system = false,
    )

    validate_gfl_cbf_internal_consistency!(on.results, on.sys)
    on_safety_dir = save_safety_filter_csvs(
        on.results,
        on.sys,
        safetycsvdir;
        t_step = t_step_primary,
    )
    print_safety_filter_summary(on.results, on.sys)

    epsilon_sweep = nothing
    if RUN_EMT_EPSILON_HEATMAP_SWEEP
        epsilon_sweep = run_emt_epsilon_heatmap_sweep(
            case_path,
            plotsdir,
            safetycsvdir,
            realized_steps,
            t_step_primary,
            on,
        )
    end

    println("\n" * "="^90)
    println("BOTH EMT CASES COMPLETED")
    println("="^90)
    println("Original EMT / filter OFF CSVs : $(off.csvdir)")
    println("Supervised EMT / filter ON CSVs: $(on.csvdir)")
    println("Comparison OFF CSVs            : $(off_safety_dir)")
    println("Comparison ON CSVs             : $(on_safety_dir)")
    println("CBF design CSVs                : $(design_dir)")
    if epsilon_sweep !== nothing
        println("Epsilon heatmap CSV            : $(epsilon_sweep.map_path)")
    end
    println("Run plot_emt_safety_filter.py to rebuild the paper figures, including the epsilon heatmap.")

    return (off = off, on = on, epsilon_sweep = epsilon_sweep)
end

main(show_system = false)
