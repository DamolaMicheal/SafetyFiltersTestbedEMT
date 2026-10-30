#!/usr/bin/env julia

include("ieee39_model_parameters.jl")
include("gfl_cbf_outer_control.jl")
include("gfl_cbf_ieee39_tools.jl")

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

    println("\n" * "="^90)
    println("BOTH EMT CASES COMPLETED")
    println("="^90)
    println("Original EMT / filter OFF CSVs : $(off.csvdir)")
    println("Supervised EMT / filter ON CSVs: $(on.csvdir)")
    println("Comparison OFF CSVs            : $(off_safety_dir)")
    println("Comparison ON CSVs             : $(on_safety_dir)")
    println("CBF design CSVs                : $(design_dir)")
    println("Run plot_emt_safety_filter_publication.py to rebuild the paper figures.")

    return (off = off, on = on)
end

main(show_system = false)
