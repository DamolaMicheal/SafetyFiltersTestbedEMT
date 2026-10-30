# GFL dynamic local frequency-safety supervisor.

const GFL_USE_SAFETY_SUPERVISOR = true

# Same safety-filter settings as the analytical model.
# Same safety-filter settings as the analytical model.
const GFL_CBF_FLOOR_HZ = -0.070
const GFL_CBF_CEIL_HZ  = +0.070

const GFL_CBF_ALPHA_LO = 5.0
const GFL_CBF_ALPHA_HI = 5.0

const GFL_CBF_EPS = 0.01
const ROCOF_TAU_D = 0.002
# Bus-dependent proximal gains k_i = rho/Gamma_ii.
const GFL_CBF_PROX_STEP_FRACTION = 0.8
const GFL_CBF_USE_LOCAL_GAMMA_SENSITIVITY_FOR_K = true

# EMT-only sensitivity identificatsion.
const GFL_CBF_SENSITIVITY_SOURCE = :emt_offline

# Automatic cache policy:
# false -> use Gamma_EMT_raw.csv if it exists; otherwise identify it automatically.
# true  -> ignore any cached matrix and force a fresh EMT identification.
const GFL_CBF_FORCE_RECOMPUTE_SENSITIVITY = false

const GFL_CBF_DESIGN_BUSES = copy(GFL_BUSES)

const GFL_CBF_SENSITIVITY_PERTURBATION_SYSTEM_PU = 0.01
const GFL_CBF_SENSITIVITY_EVENT_TIME_S = 0.050
const GFL_CBF_SENSITIVITY_WINDOW_START_S = 0.020
const GFL_CBF_SENSITIVITY_WINDOW_END_S   = 0.050
const GFL_CBF_SENSITIVITY_SIM_END_S      = 0.120
const GFL_CBF_SENSITIVITY_EXPORT_DT      = 0.0005

const GFL_CBF_SENSITIVITY_ABSTOL = 1e-6
const GFL_CBF_SENSITIVITY_RELTOL = 1e-6
const GFL_CBF_SENSITIVITY_DTINIT = 1e-5
const GFL_CBF_SENSITIVITY_DTMAX  = 5e-4

const GFL_CBF_MAX_RELATIVE_ASYMMETRY_WARN = 0.35
const GFL_CBF_MIN_POSITIVE_EIGENVALUE = 1e-10

const GFL_CBF_Z_MIN_SYS_PU = -Inf
const GFL_CBF_Z_MAX_SYS_PU = +Inf
const GFL_I_RATED = 1.0
const GFL_P_MIN = 0.0
const ENFORCE_HEADROOM = false
const GFL_CBF_DORMANCY_TOL_SYS_PU = 1e-7

# Julia only exports CSVs; publication plots are rebuilt in Python.
const ACTIVE_SAFETY_BOUND = "auto"
const MAKE_JULIA_CBF_DIAGNOSTIC_PLOTS = false
const MAKE_PYTHON_STYLE_EMT_PLOTS = false
const PYTHON_STYLE_DPI = 600

const CBF_PY_COLORS = Dict{Int,String}(
    30 => "#0072B2",
    31 => "#009E73",
    32 => "#E69F00",
    33 => "#3A923A",
    34 => "#56B4E9",
    35 => "#CC6677",
    36 => "#D55E00",
    37 => "#AA4499",
    38 => "#7E57C2",
    39 => "#008B8B",
)
cbf_bus_color(bus::Int) = get(CBF_PY_COLORS, bus, "#252525")


function gfl_cbf_case_enabled(sys::System)
    for bus in GFL_BUSES
        dyn = dyn_inverter_at_bus(sys, bus)
        outer = PSY.get_outer_control(dyn)
        active = PSY.get_active_power_control(outer)
        if active isa ActivePowerPISafetyFilter
            return active.filter_enabled
        end
    end
    return false
end

# Runtime CBF design data. The EMT sensitivity is identified before the final
# nonlinear validation system is constructed, then frozen for that run.
const GFL_CBF_GAMMA_RAW_REF = Ref{Matrix{Float64}}(
    zeros(length(GFL_CBF_DESIGN_BUSES), length(GFL_CBF_DESIGN_BUSES))
)
const GFL_CBF_GAMMA_REF = Ref{Matrix{Float64}}(
    zeros(length(GFL_CBF_DESIGN_BUSES), length(GFL_CBF_DESIGN_BUSES))
)
const GFL_CBF_K_MAP = Dict{Int,Float64}()
const GFL_CBF_DESIGN_READY = Ref(false)
const GFL_CBF_DESIGN_SOURCE_REF = Ref("uninitialized")

function gfl_cbf_gamma_raw()
    GFL_CBF_DESIGN_READY[] ||
        error("The GFL CBF sensitivity has not been prepared.")
    return GFL_CBF_GAMMA_RAW_REF[]
end

function gfl_cbf_gamma()
    GFL_CBF_DESIGN_READY[] ||
        error("The GFL CBF sensitivity has not been prepared.")
    return GFL_CBF_GAMMA_REF[]
end

function gfl_cbf_k(bus::Int)
    GFL_CBF_DESIGN_READY[] ||
        error("The GFL CBF proximal gains have not been prepared.")
    haskey(GFL_CBF_K_MAP, bus) ||
        error("No proximal gain k_i is available for GFL bus $bus.")
    return GFL_CBF_K_MAP[bus]
end

function set_gfl_cbf_design!(Γraw::AbstractMatrix{<:Real}; source::AbstractString)
    nF = length(GFL_CBF_DESIGN_BUSES)
    size(Γraw) == (nF, nF) ||
        error("Expected a $(nF)x$(nF) GFL sensitivity matrix, got $(size(Γraw)).")

    Γr = Matrix{Float64}(Γraw)
    all(isfinite, Γr) || error("The EMT sensitivity matrix contains non-finite values.")

    d_raw = diag(Γr)
    any(d_raw .<= 0.0) &&
        error(
            "At least one local EMT RoCoF sensitivity is non-positive: $(d_raw). " *
            "Check the z_i sign, perturbation size, PLL signal, and calibration window."
        )

    Γs = 0.5 .* (Γr .+ transpose(Γr))
    asym = opnorm(Γr .- transpose(Γr), Inf) /
           max(opnorm(Γr, Inf), eps(Float64))
    if asym > GFL_CBF_MAX_RELATIVE_ASYMMETRY_WARN
        @warn "Offline EMT sensitivity is strongly asymmetric" relative_asymmetry=asym
    end

    λ = eigvals(Symmetric(Γs))
    λmin = minimum(λ)
    λmax = maximum(λ)
    λmin > GFL_CBF_MIN_POSITIVE_EIGENVALUE ||
        error(
            "The symmetric EMT sensitivity is not positive definite. " *
            "lambda_min=$(λmin). Adjust the calibration perturbation/window or inspect the raw matrix."
        )

    if GFL_CBF_USE_LOCAL_GAMMA_SENSITIVITY_FOR_K
        d = diag(Γs)
        Dinvhalf = Diagonal(1.0 ./ sqrt.(d))
        Γn = Symmetric(Dinvhalf * Γs * Dinvhalf)
        λnorm = maximum(eigvals(Γn))
        ρ = GFL_CBF_PROX_STEP_FRACTION / λnorm
        kvals = ρ ./ d
    else
        kvals = fill(
            GFL_CBF_PROX_STEP_FRACTION / λmax,
            nF,
        )
    end

    empty!(GFL_CBF_K_MAP)
    for (i, bus) in enumerate(GFL_CBF_DESIGN_BUSES)
        GFL_CBF_K_MAP[bus] = Float64(kvals[i])
    end

    GFL_CBF_GAMMA_RAW_REF[] = Γr
    GFL_CBF_GAMMA_REF[] = Γs
    GFL_CBF_DESIGN_SOURCE_REF[] = String(source)
    GFL_CBF_DESIGN_READY[] = true

    Khalf = Diagonal(sqrt.([GFL_CBF_K_MAP[b] for b in GFL_CBF_DESIGN_BUSES]))
    λweighted = maximum(eigvals(Symmetric(Khalf * Γs * Khalf)))
    λweighted < 2.0 ||
        error(
            "Computed proximal gains violate " *
            "lambda_max(K^(1/2) Gamma K^(1/2)) < 2; value=$(λweighted)."
        )

    return (
        gamma_raw = Γr,
        gamma = Γs,
        eigenvalues = λ,
        relative_asymmetry = asym,
        k = [GFL_CBF_K_MAP[b] for b in GFL_CBF_DESIGN_BUSES],
        weighted_lambda_max = λweighted,
    )
end

function validate_cbf_design!()
    GFL_CBF_DESIGN_READY[] ||
        error("The GFL CBF design has not been prepared.")
    Set(GFL_BUSES) == Set(GFL_CBF_DESIGN_BUSES) ||
        error(
            "CBF design buses $(GFL_CBF_DESIGN_BUSES) do not match GFL_BUSES=$(GFL_BUSES)."
        )
    return true
end


# -----------------------------------------------------------------------------
# Offline EMT sensitivity identification
# -----------------------------------------------------------------------------

function calibration_time_grid()
    tg = collect(
        0.0:GFL_CBF_SENSITIVITY_EXPORT_DT:GFL_CBF_SENSITIVITY_SIM_END_S
    )
    tev = GFL_CBF_SENSITIVITY_EVENT_TIME_S
    if !any(abs.(tg .- tev) .<= 1e-12)
        push!(tg, tev)
        sort!(tg)
    end
    return tg
end

function calibration_window_indices(tg::Vector{Float64})
    t0 = GFL_CBF_SENSITIVITY_EVENT_TIME_S +
         GFL_CBF_SENSITIVITY_WINDOW_START_S
    t1 = GFL_CBF_SENSITIVITY_EVENT_TIME_S +
         GFL_CBF_SENSITIVITY_WINDOW_END_S
    t1 > t0 || error("The EMT sensitivity averaging window must have positive length.")
    t1 <= GFL_CBF_SENSITIVITY_SIM_END_S ||
        error("The EMT sensitivity averaging window extends beyond the calibration run.")
    idx = findall(t -> t0 <= t <= t1, tg)
    length(idx) >= 2 ||
        error("The EMT sensitivity averaging window contains too few samples.")
    return idx
end

function run_gfl_sensitivity_probe(
    case_path::String,
    pfdir::String,
    input_bus::Int,
    signed_delta_z_sys::Float64,
)
    # k_i is irrelevant during calibration because the safety-filter state is
    # disabled. A positive placeholder avoids requiring the final design first.
    sys, _ = build_initialized_emt_system(
        case_path,
        pfdir;
        write_pf = false,
        show_reference_summary = false,
        gfl_filter_enabled = false,
        calibration_k_override = 1.0,
    )

    dyn = dyn_inverter_at_bus(sys, input_bus)
    outer = PSY.get_outer_control(dyn)
    active = PSY.get_active_power_control(outer)
    p_ref0 = Float64(PSY.get_P_ref(active))

    mbase = Float64(PSY.get_base_power(gen_at_bus(sys, input_bus)))
    delta_z_dev = signed_delta_z_sys * SYS_BASE_MVA / mbase
    p_ref_probe = p_ref0 + delta_z_dev

    probe = ControlReferenceChange(
        GFL_CBF_SENSITIVITY_EVENT_TIME_S,
        dyn,
        :P_ref,
        p_ref_probe,
    )

    sim = Simulation(
        ResidualModel,
        sys,
        mktempdir(),
        (0.0, GFL_CBF_SENSITIVITY_SIM_END_S),
        PowerSimulationsDynamics.Perturbation[probe];
        console_level = Logging.Warn,
    )

    execute!(
        sim,
        IDA(linear_solver = :Dense, max_order = SOLVER_MAX_ORDER);
        abstol = GFL_CBF_SENSITIVITY_ABSTOL,
        reltol = GFL_CBF_SENSITIVITY_RELTOL,
        dtinit = GFL_CBF_SENSITIVITY_DTINIT,
        dtmax = GFL_CBF_SENSITIVITY_DTMAX,
    )

    results = read_results(sim)
    tg = calibration_time_grid()
    nF = length(GFL_CBF_DESIGN_BUSES)
    rocof = zeros(length(tg), nF)
    freq = zeros(length(tg), nF)

    for (i, output_bus) in enumerate(GFL_CBF_DESIGN_BUSES)
        sig = safety_filter_signals_at_bus(results, sys, output_bus)
        sig === nothing &&
            error("Could not extract GFL safety-filter signals at bus $output_bus during calibration.")
        rocof[:, i] .= interp1_linear(
            Float64.(sig.t),
            Float64.(sig.rocof_hz_s ./ F_BASE),
            tg,
        )
        freq[:, i] .= interp1_linear(
            Float64.(sig.t),
            Float64.(sig.freq_hz ./ F_BASE),
            tg,
        )
    end

    return (
        t = tg,
        rocof_pu_s = rocof,
        freq_pu = freq,
        p_ref0_dev = p_ref0,
        delta_z_sys = signed_delta_z_sys,
        delta_z_dev = delta_z_dev,
        mbase = mbase,
    )
end

function save_sensitivity_probe_pair(
    design_dir::String,
    input_bus::Int,
    plus,
    minus,
)
    mkpath(design_dir)
    df = DataFrame(t_s = plus.t)
    for (i, output_bus) in enumerate(GFL_CBF_DESIGN_BUSES)
        df[!, Symbol("rocof_plus_bus$(output_bus)_pu_per_s")] =
            plus.rocof_pu_s[:, i]
        df[!, Symbol("rocof_minus_bus$(output_bus)_pu_per_s")] =
            minus.rocof_pu_s[:, i]
        df[!, Symbol("freq_plus_bus$(output_bus)_pu")] =
            plus.freq_pu[:, i]
        df[!, Symbol("freq_minus_bus$(output_bus)_pu")] =
            minus.freq_pu[:, i]
    end
    path = joinpath(
        design_dir,
        @sprintf("sensitivity_probe_input_bus%02d.csv", input_bus),
    )
    CSV.write(path, df)
    return path
end

function save_emt_sensitivity_design(
    design_dir::String,
    Γraw::Matrix{Float64},
    design_info,
    pair_rows::DataFrame,
)
    mkpath(design_dir)

    raw_df = DataFrame(
        Γraw,
        [Symbol("bus_$(b)") for b in GFL_CBF_DESIGN_BUSES],
    )
    insertcols!(raw_df, 1, :bus => GFL_CBF_DESIGN_BUSES)
    CSV.write(joinpath(design_dir, "Gamma_EMT_raw.csv"), raw_df)

    Γ = design_info.gamma
    sym_df = DataFrame(
        Γ,
        [Symbol("bus_$(b)") for b in GFL_CBF_DESIGN_BUSES],
    )
    insertcols!(sym_df, 1, :bus => GFL_CBF_DESIGN_BUSES)
    CSV.write(joinpath(design_dir, "Gamma_EMT_symmetric.csv"), sym_df)

    gain_df = DataFrame(
        bus = GFL_CBF_DESIGN_BUSES,
        gamma_ii_per_s = diag(Γ),
        k_i_s = design_info.k,
        proximal_step_fraction =
            fill(GFL_CBF_PROX_STEP_FRACTION, length(GFL_CBF_DESIGN_BUSES)),
        local_gamma_scaling =
            fill(
                GFL_CBF_USE_LOCAL_GAMMA_SENSITIVITY_FOR_K,
                length(GFL_CBF_DESIGN_BUSES),
            ),
    )
    CSV.write(joinpath(design_dir, "cbf_proximal_gains.csv"), gain_df)
    if nrow(pair_rows) > 0
        CSV.write(
            joinpath(design_dir, "sensitivity_pairwise_summary.csv"),
            pair_rows,
        )
    end

    settings = DataFrame(
        quantity = [
            "sensitivity_source",
            "delta_z_system_pu",
            "delta_z_MW_on_100MVA_base",
            "probe_event_time_s",
            "window_start_after_event_s",
            "window_end_after_event_s",
            "simulation_end_s",
            "export_dt_s",
            "relative_asymmetry",
            "lambda_min_symmetric",
            "lambda_max_symmetric",
            "weighted_lambda_max",
        ],
        value = string.([
            GFL_CBF_SENSITIVITY_SOURCE,
            GFL_CBF_SENSITIVITY_PERTURBATION_SYSTEM_PU,
            GFL_CBF_SENSITIVITY_PERTURBATION_SYSTEM_PU * SYS_BASE_MVA,
            GFL_CBF_SENSITIVITY_EVENT_TIME_S,
            GFL_CBF_SENSITIVITY_WINDOW_START_S,
            GFL_CBF_SENSITIVITY_WINDOW_END_S,
            GFL_CBF_SENSITIVITY_SIM_END_S,
            GFL_CBF_SENSITIVITY_EXPORT_DT,
            design_info.relative_asymmetry,
            minimum(design_info.eigenvalues),
            maximum(design_info.eigenvalues),
            design_info.weighted_lambda_max,
        ]),
    )
    CSV.write(joinpath(design_dir, "sensitivity_settings.csv"), settings)

    return nothing
end

function load_cached_emt_sensitivity(design_dir::String)
    path = joinpath(design_dir, "Gamma_EMT_raw.csv")
    isfile(path) ||
        error("Cached EMT sensitivity not found at $path.")

    df = CSV.read(path, DataFrame)
    expected = GFL_CBF_DESIGN_BUSES
    (:bus in propertynames(df)) ||
        error("Cached EMT sensitivity CSV does not contain a bus column.")

    buses = Int.(df.bus)
    buses == expected ||
        error(
            "Cached EMT sensitivity bus order $buses does not match " *
            "GFL_CBF_DESIGN_BUSES=$expected."
        )

    Γ = zeros(length(expected), length(expected))
    for (j, bus) in enumerate(expected)
        col = Symbol("bus_$(bus)")
        col in propertynames(df) ||
            error("Cached EMT sensitivity is missing column $col.")
        Γ[:, j] .= Float64.(df[!, col])
    end
    return Γ
end

function estimate_emt_gfl_sensitivity!(
    case_path::String,
    pfdir::String,
    design_dir::String,
)
    GFL_CBF_SENSITIVITY_SOURCE == :emt_offline ||
        error(
            "Unsupported GFL_CBF_SENSITIVITY_SOURCE=$(GFL_CBF_SENSITIVITY_SOURCE). " *
            "Use :emt_offline."
        )

    cache_path = joinpath(design_dir, "Gamma_EMT_raw.csv")

    if isfile(cache_path) && !GFL_CBF_FORCE_RECOMPUTE_SENSITIVITY
        @info "Using cached EMT GFL sensitivity" file=cache_path
        Γraw = load_cached_emt_sensitivity(design_dir)
        info = set_gfl_cbf_design!(Γraw; source = "cached EMT sensitivity")
        save_emt_sensitivity_design(
            design_dir,
            Γraw,
            info,
            DataFrame(
                input_bus = Int[],
                output_bus = Int[],
                input_mbase_mva = Float64[],
                delta_z_system_pu = Float64[],
                delta_z_device_pu = Float64[],
                gamma_mean_per_s = Float64[],
                gamma_std_per_s = Float64[],
            ),
        )
        return info
    end

    if GFL_CBF_FORCE_RECOMPUTE_SENSITIVITY
        @info "Forcing fresh EMT GFL sensitivity identification"
    else
        @info "No cached EMT GFL sensitivity found; identifying it automatically" file=cache_path
    end

    nF = length(GFL_CBF_DESIGN_BUSES)
    Γraw = zeros(nF, nF)
    Γstd = zeros(nF, nF)
    pair_rows = DataFrame(
        input_bus = Int[],
        output_bus = Int[],
        input_mbase_mva = Float64[],
        delta_z_system_pu = Float64[],
        delta_z_device_pu = Float64[],
        gamma_mean_per_s = Float64[],
        gamma_std_per_s = Float64[],
    )
    idx = calibration_window_indices(calibration_time_grid())
    δ = GFL_CBF_SENSITIVITY_PERTURBATION_SYSTEM_PU

    println("\n" * "="^92)
    println("OFFLINE EMT GFL RoCoF SENSITIVITY IDENTIFICATION")
    println("GFL order: $(GFL_CBF_DESIGN_BUSES)")
    @printf(
        "Central probe: +/- %.6f pu system base = +/- %.3f MW on %.1f MVA base\n",
        δ,
        δ * SYS_BASE_MVA,
        SYS_BASE_MVA,
    )
    @printf(
        "Probe at %.4f s; averaging window %.4f--%.4f s after the probe\n",
        GFL_CBF_SENSITIVITY_EVENT_TIME_S,
        GFL_CBF_SENSITIVITY_WINDOW_START_S,
        GFL_CBF_SENSITIVITY_WINDOW_END_S,
    )
    println("="^92)

    for (j, input_bus) in enumerate(GFL_CBF_DESIGN_BUSES)
        @info "Running positive EMT sensitivity probe" input_bus=input_bus delta_z_system_pu=δ
        plus = run_gfl_sensitivity_probe(
            case_path,
            pfdir,
            input_bus,
            +δ,
        )

        @info "Running negative EMT sensitivity probe" input_bus=input_bus delta_z_system_pu=-δ
        minus = run_gfl_sensitivity_probe(
            case_path,
            pfdir,
            input_bus,
            -δ,
        )

        plus.t == minus.t ||
            error("Positive/negative EMT sensitivity probe time grids do not match.")

        save_sensitivity_probe_pair(
            design_dir,
            input_bus,
            plus,
            minus,
        )

        central = (
            plus.rocof_pu_s .- minus.rocof_pu_s
        ) ./ (2.0 * δ)

        for i in 1:nF
            samples = central[idx, i]
            Γraw[i, j] = mean(samples)
            Γstd[i, j] = std(samples)
            push!(
                pair_rows,
                (
                    input_bus,
                    GFL_CBF_DESIGN_BUSES[i],
                    plus.mbase,
                    δ,
                    plus.delta_z_dev,
                    Γraw[i, j],
                    Γstd[i, j],
                ),
            )
        end
    end

    # Save the empirical matrix before the design checks so the calibration
    # remains inspectable even if the symmetric matrix fails the PD test.
    raw_df = DataFrame(
        Γraw,
        [Symbol("bus_$(b)") for b in GFL_CBF_DESIGN_BUSES],
    )
    insertcols!(raw_df, 1, :bus => GFL_CBF_DESIGN_BUSES)
    CSV.write(joinpath(design_dir, "Gamma_EMT_raw.csv"), raw_df)
    CSV.write(
        joinpath(design_dir, "sensitivity_pairwise_summary.csv"),
        pair_rows,
    )

    info = set_gfl_cbf_design!(
        Γraw;
        source = "offline EMT central-difference RoCoF sensitivity",
    )
    save_emt_sensitivity_design(
        design_dir,
        Γraw,
        info,
        pair_rows,
    )

    println("\nRaw effective EMT sensitivity Gamma_raw [1/s]:")
    display(Γraw)
    println("\nSymmetric design sensitivity Gamma [1/s]:")
    display(info.gamma)
    @printf(
        "\nrelative asymmetry = %.6e\nlambda_min = %.6e\nlambda_max = %.6e\n",
        info.relative_asymmetry,
        minimum(info.eigenvalues),
        maximum(info.eigenvalues),
    )
    println("Proximal gains:")
    for bus in GFL_CBF_DESIGN_BUSES
        @printf("  Bus %d: k_i = %.9g s\n", bus, gfl_cbf_k(bus))
    end
    @printf(
        "lambda_max(K^(1/2) Gamma K^(1/2)) = %.6f\n",
        info.weighted_lambda_max,
    )
    println("="^92 * "\n")

    return info
end


# -----------------------------------------------------------------------------
# Safety-filter signals, CSVs, and Python-style plots
# -----------------------------------------------------------------------------

function safety_filter_control_at_bus(sys::System, bus::Int)
    dyn = dyn_inverter_at_bus(sys, bus)
    outer = PSY.get_outer_control(dyn)
    return PSY.get_active_power_control(outer)
end

function _dirty_rocof_trace(t::Vector{Float64}, omega_dev::Vector{Float64}, tau::Float64)
    n = length(t)
    chi = zeros(n)
    rocof = zeros(n)
    n == 0 && return chi, rocof
    chi[1] = omega_dev[1]
    for k in 2:n
        h = max(t[k] - t[k - 1], 0.0)
        a = exp(-h / tau)
        chi[k] = a * chi[k - 1] + (1.0 - a) * omega_dev[k - 1]
        rocof[k] = (omega_dev[k] - chi[k]) / tau
    end
    return chi, rocof
end

function safety_filter_signals_at_bus(results, sys::System, bus::Int;
                                      t_step::Float64 = 0.25,
                                      for_csv::Bool = false,
                                      for_plots::Bool = false)
    dyn = dyn_inverter_at_bus(sys, bus)
    name = PSY.get_name(dyn)
    active = safety_filter_control_at_bus(sys, bus)

    kp_pll, ki_pll = gfl_pll_gains_at_bus(sys, bus)
    pll = get_gfl_pll_speed_pu(results, name; kp_pll = kp_pll, ki_pll = ki_pll)
    pll === nothing && return nothing

    t, omega_pll, _ = pll
    t = Float64.(t)
    omega_pll = Float64.(omega_pll)
    omega_dev = omega_pll .- 1.0

    is_supervised = active isa ActivePowerPISafetyFilter
    alpha_lo = is_supervised ? active.α_lo : GFL_CBF_ALPHA_LO
    alpha_hi = is_supervised ? active.α_hi : GFL_CBF_ALPHA_HI
    omega_min_dev = is_supervised ? active.ω_min_dev : GFL_CBF_FLOOR_HZ / F_BASE
    omega_max_dev = is_supervised ? active.ω_max_dev : GFL_CBF_CEIL_HZ / F_BASE

    if is_supervised
        tchi, chi_raw = get_state_series(results, (name, :chi_rocof))
        tz, z_raw = get_state_series(results, (name, :z_cbf))
        chi = interp1_linear(Float64.(tchi), Float64.(chi_raw), t)
        z_sys = interp1_linear(Float64.(tz), Float64.(z_raw), t)
        rocof_hat = (omega_dev .- chi) ./ active.τd
        k_prox = active.k_prox
        a = z_sys .+ k_prox .* (
            .-alpha_lo .* (omega_dev .- omega_min_dev) .- rocof_hat
        )
        b = z_sys .+ k_prox .* (
            alpha_hi .* (omega_max_dev .- omega_dev) .- rocof_hat
        )
        target = [project_zero_to_interval(ai, bi) for (ai, bi) in zip(a, b)]
        z_cmd_sys = active.filter_enabled ?
            clamp.(z_sys, active.z_min_sys, active.z_max_sys) :
            zeros(length(z_sys))
    else
        chi, rocof_hat = _dirty_rocof_trace(t, omega_dev, ROCOF_TAU_D)
        z_sys = zeros(length(t))
        z_cmd_sys = zeros(length(t))
        k_prox = GFL_CBF_DESIGN_READY[] ? gfl_cbf_k(bus) : NaN
        if isfinite(k_prox)
            a = k_prox .* (
                .-alpha_lo .* (omega_dev .- omega_min_dev) .- rocof_hat
            )
            b = k_prox .* (
                alpha_hi .* (omega_max_dev .- omega_dev) .- rocof_hat
            )
            target = [project_zero_to_interval(ai, bi) for (ai, bi) in zip(a, b)]
        else
            a = fill(NaN, length(t))
            b = fill(NaN, length(t))
            target = zeros(length(t))
        end
    end

    mbase = Float64(PSY.get_base_power(gen_at_bus(sys, bus)))
    dev_to_sys = mbase / SYS_BASE_MVA
    sys_to_dev = SYS_BASE_MVA / mbase
    Komega = try get(PSY.get_ext(active), "Kω", getGflKw(bus)) catch; getGflKw(bus) end
    z_cmd_dev = z_cmd_sys .* sys_to_dev

    if is_supervised && active.enforce_head
        hr = gfl_headroom_at_bus(results, sys, bus)
        if hr !== nothing && !all(isnan, hr.P_head)
            P_head = interp1_linear(hr.t, hr.P_head, t)
            P_shed = interp1_linear(hr.t, hr.P_now .- active.P_min, t)
            z_cmd_dev = clamp.(z_cmd_dev, .-max.(P_shed, 0.0), max.(P_head, 0.0))
            z_cmd_sys = z_cmd_dev .* dev_to_sys
        end
    end

    delta_p_ffr_sys = (-Komega .* omega_dev) .* dev_to_sys
    delta_p_sys = delta_p_ffr_sys .+ z_cmd_sys

    freq_hz = omega_dev .* F_BASE
    rocof_hz_s = rocof_hat .* F_BASE
    h_lo_hz = freq_hz .- GFL_CBF_FLOOR_HZ
    h_hi_hz = GFL_CBF_CEIL_HZ .- freq_hz
    lower_bound_hz_s = .-alpha_lo .* h_lo_hz
    upper_bound_hz_s = alpha_hi .* h_hi_hz
    lower_margin_hz_s = rocof_hz_s .- lower_bound_hz_s
    upper_margin_hz_s = upper_bound_hz_s .- rocof_hz_s

    if (for_csv && RESAMPLE_FOR_CSV) || (for_plots && RESAMPLE_FOR_PLOTS)
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        keys = (
            :freq_hz, :rocof_hz_s, :z_sys, :z_cmd_sys, :z_cmd_dev,
            :target, :a, :b, :delta_p_ffr_sys, :delta_p_sys, :h_lo_hz, :h_hi_hz,
            :lower_bound_hz_s, :upper_bound_hz_s,
            :lower_margin_hz_s, :upper_margin_hz_s,
        )
        vals = (
            freq_hz, rocof_hz_s, z_sys, z_cmd_sys, z_cmd_dev,
            target, a, b, delta_p_ffr_sys, delta_p_sys, h_lo_hz, h_hi_hz,
            lower_bound_hz_s, upper_bound_hz_s,
            lower_margin_hz_s, upper_margin_hz_s,
        )
        data = Dict{Symbol,Vector{Float64}}()
        for (key, value) in zip(keys, vals)
            data[key] = interp1_linear(t, value, tg)
        end
        t = tg
        freq_hz = data[:freq_hz]
        rocof_hz_s = data[:rocof_hz_s]
        z_sys = data[:z_sys]
        z_cmd_sys = data[:z_cmd_sys]
        z_cmd_dev = data[:z_cmd_dev]
        target = data[:target]
        a = data[:a]
        b = data[:b]
        delta_p_ffr_sys = data[:delta_p_ffr_sys]
        delta_p_sys = data[:delta_p_sys]
        h_lo_hz = data[:h_lo_hz]
        h_hi_hz = data[:h_hi_hz]
        lower_bound_hz_s = data[:lower_bound_hz_s]
        upper_bound_hz_s = data[:upper_bound_hz_s]
        lower_margin_hz_s = data[:lower_margin_hz_s]
        upper_margin_hz_s = data[:upper_margin_hz_s]
    end

    return (
        t = t,
        freq_hz = freq_hz,
        rocof_hat_hz_s = rocof_hz_s,
        rocof_hz_s = rocof_hz_s,
        z_sys = z_sys,
        z_cmd_sys = z_cmd_sys,
        z_cmd_dev = z_cmd_dev,
        target_sys = target,
        a_sys = a,
        b_sys = b,
        Δp_ffr_sys = delta_p_ffr_sys,
        Δp_sys = delta_p_sys,
        h_lo_hz = h_lo_hz,
        h_hi_hz = h_hi_hz,
        lower_bound_hz_s = lower_bound_hz_s,
        upper_bound_hz_s = upper_bound_hz_s,
        lower_margin_hz_s = lower_margin_hz_s,
        upper_margin_hz_s = upper_margin_hz_s,
        k_prox = k_prox,
        Kω = Komega,
        mbase = mbase,
        supervised = is_supervised,
    )
end


_first_cbf_state(results, name, syms) = get_first_state_series(results, name, syms)

function gfl_headroom_at_bus(results, sys::System, bus::Int)
    dyn = dyn_inverter_at_bus(sys, bus)
    name = PSY.get_name(dyn)

    ir = _first_cbf_state(
        results, name,
        Symbol[:ir_cnv, :Ir_cnv, :i_cnv_r, :ir_filter, :Ir_filter],
    )
    ii = _first_cbf_state(
        results, name,
        Symbol[:ii_cnv, :Ii_cnv, :i_cnv_i, :ii_filter, :Ii_filter],
    )
    vr = _first_cbf_state(
        results, name,
        Symbol[:vr_filter, :Vr_filter, :vr_cap, :vr_out],
    )
    vi = _first_cbf_state(
        results, name,
        Symbol[:vi_filter, :Vi_filter, :vi_cap, :vi_out],
    )

    (ir === nothing || ii === nothing) && return nothing

    t = Float64.(ir[1])
    Ir = Float64.(ir[2])
    Ii = interp1_linear(ii[1], Float64.(ii[2]), t)
    Imag = sqrt.(Ir .^ 2 .+ Ii .^ 2)

    if vr === nothing || vi === nothing
        nanv = fill(NaN, length(t))
        return (
            t = t,
            name = name,
            I_rated = GFL_I_RATED,
            Imag = Imag,
            I_head = GFL_I_RATED .- Imag,
            P_now = nanv,
            P_max = nanv,
            P_head = nanv,
            Vmag = nanv,
        )
    end

    Vr = interp1_linear(vr[1], Float64.(vr[2]), t)
    Vi = interp1_linear(vi[1], Float64.(vi[2]), t)
    Vmag = sqrt.(Vr .^ 2 .+ Vi .^ 2)
    P_now = Ir .* Vr .+ Ii .* Vi

    # Active-power capability at the measured voltage for the configured
    # current limit. This is used as a supervisory diagnostic by default.
    i_active = P_now ./ max.(Vmag, 1e-6)
    i_reactive_sq = max.(Imag .^ 2 .- i_active .^ 2, 0.0)
    P_max = Vmag .* sqrt.(max.(GFL_I_RATED^2 .- i_reactive_sq, 0.0))

    return (
        t = t,
        name = name,
        I_rated = GFL_I_RATED,
        Imag = Imag,
        I_head = GFL_I_RATED .- Imag,
        P_now = P_now,
        P_max = P_max,
        P_head = P_max .- P_now,
        Vmag = Vmag,
    )
end

function save_gfl_headroom_csvs(
    results,
    sys::System,
    outdir::String;
    t_step::Float64 = 0.25,
)
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    dir = joinpath(outdir, tag, "headroom")
    mkpath(dir)

    for bus in GFL_BUSES
        hr = gfl_headroom_at_bus(results, sys, bus)
        cb = safety_filter_signals_at_bus(
            results, sys, bus;
            t_step = t_step,
            for_csv = true,
        )
        (hr === nothing || cb === nothing) && continue

        tg = cb.t
        Imag = interp1_linear(hr.t, hr.Imag, tg)
        Ihead = interp1_linear(hr.t, hr.I_head, tg)
        Pnow = interp1_linear(hr.t, hr.P_now, tg)
        Pmax = interp1_linear(hr.t, hr.P_max, tg)
        Phead = interp1_linear(hr.t, hr.P_head, tg)

        df = DataFrame(
            t_s = tg,
            current_magnitude_pu = Imag,
            current_headroom_pu = Ihead,
            terminal_active_power_device_pu = Pnow,
            active_power_limit_device_pu = Pmax,
            injection_headroom_device_pu = Phead,
            z_state_system_pu = cb.z_sys,
            z_command_system_pu = cb.z_cmd_sys,
            z_command_device_pu = cb.z_cmd_dev,
        )
        CSV.write(joinpath(dir, @sprintf("gfl_headroom_bus%02d.csv", bus)), df)
    end

    return dir
end

function active_safety_side(results, sys::System)
    mode = lowercase(strip(ACTIVE_SAFETY_BOUND))
    mode in ("lower", "upper") && return mode
    mode == "auto" || error("ACTIVE_SAFETY_BOUND must be auto, lower, or upper.")

    lo_ratio = 0.0
    hi_ratio = 0.0
    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(results, sys, bus)
        s === nothing && continue
        lo_ratio = max(lo_ratio, max(0.0, -minimum(s.freq_hz)) / abs(GFL_CBF_FLOOR_HZ))
        hi_ratio = max(hi_ratio, max(0.0, maximum(s.freq_hz)) / abs(GFL_CBF_CEIL_HZ))
    end
    return lo_ratio >= hi_ratio ? "lower" : "upper"
end

function save_safety_disturbance_schedule(
    realized_steps::Vector{RealizedLoadStep},
    outdir::String,
)
    mkpath(outdir)
    rows = DataFrame(
        bus = Int[],
        event_time_s = Float64[],
        delta_p_mw = Float64[],
        delta_q_mvar = Float64[],
        delta_p_system_pu = Float64[],
        delta_q_system_pu = Float64[],
        background_load_model = String[],
        disturbance_load_model = String[],
    )

    for e in realized_steps
        push!(rows, (
            e.bus,
            e.t_step,
            e.ΔP_MW,
            e.ΔQ_MVAr,
            e.deltaP_pu,
            e.deltaQ_pu,
            "FixedAdmittance",
            "constant-PQ PerturbLoad (Q may be zero)",
        ))
    end

    path = joinpath(outdir, "disturbance_schedule.csv")
    CSV.write(path, rows)
    return path
end

function save_safety_filter_csvs(results, sys::System, outdir::String;
                                 t_step::Float64 = 0.25)
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    csvdir = joinpath(outdir, tag)
    mkpath(csvdir)
    side = active_safety_side(results, sys)

    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(
            results, sys, bus; t_step = t_step, for_csv = true)
        s === nothing && continue

        active_limit = side == "lower" ?
            fill(GFL_CBF_FLOOR_HZ, length(s.t)) :
            fill(GFL_CBF_CEIL_HZ, length(s.t))
        active_bound = side == "lower" ? s.lower_bound_hz_s : s.upper_bound_hz_s
        active_barrier = side == "lower" ? s.h_lo_hz : s.h_hi_hz
        active_margin = side == "lower" ? s.lower_margin_hz_s : s.upper_margin_hz_s

        df = DataFrame(
            t_s = s.t,
            frequency_deviation_hz = s.freq_hz,
            active_frequency_limit_hz = active_limit,
            rocof_hat_hz_per_s = s.rocof_hz_s,
            active_cbf_bound_hz_per_s = active_bound,
            active_barrier_hz = active_barrier,
            active_cbf_margin_hat_hz_per_s = active_margin,
            z_state_system_pu = s.z_sys,
            z_command_system_pu = s.z_cmd_sys,
            z_command_device_pu = s.z_cmd_dev,
            proximal_target_system_pu = s.target_sys,
            proximal_lower_a_system_pu = s.a_sys,
            proximal_upper_b_system_pu = s.b_sys,
            lower_cbf_margin_hz_per_s = s.lower_margin_hz_s,
            upper_cbf_margin_hz_per_s = s.upper_margin_hz_s,
            delta_pF_ffr_system_pu = s.Δp_ffr_sys,
            delta_pF_system_pu = s.Δp_sys,
            filter_enabled = fill(gfl_cbf_case_enabled(sys), length(s.t)),
        )
        CSV.write(joinpath(csvdir, @sprintf("gfl_bus%02d.csv", bus)), df)
    end

    if GFL_CBF_DESIGN_READY[]
        design = DataFrame(
            bus = GFL_BUSES,
            gamma_ii_per_s = [
                gfl_cbf_gamma()[i, i] for i in eachindex(GFL_CBF_DESIGN_BUSES)
            ],
            k_i_s = [gfl_cbf_k(b) for b in GFL_BUSES],
            Komega_device_pu = [getGflKw(b) for b in GFL_BUSES],
            proximal_step_fraction = fill(GFL_CBF_PROX_STEP_FRACTION, length(GFL_BUSES)),
            sensitivity_source = fill(GFL_CBF_DESIGN_SOURCE_REF[], length(GFL_BUSES)),
        )
        CSV.write(joinpath(csvdir, "cbf_design_parameters.csv"), design)

        gamma_df = DataFrame(
            gfl_cbf_gamma(),
            [Symbol("bus_$(b)") for b in GFL_CBF_DESIGN_BUSES],
        )
        insertcols!(gamma_df, 1, :bus => GFL_CBF_DESIGN_BUSES)
        CSV.write(joinpath(csvdir, "Gamma.csv"), gamma_df)

        gamma_raw_df = DataFrame(
            gfl_cbf_gamma_raw(),
            [Symbol("bus_$(b)") for b in GFL_CBF_DESIGN_BUSES],
        )
        insertcols!(gamma_raw_df, 1, :bus => GFL_CBF_DESIGN_BUSES)
        CSV.write(joinpath(csvdir, "Gamma_raw.csv"), gamma_raw_df)
    end

    for bus in SG_BUSES
        fs = frequency_deviation_hz_at_bus(results, sys, bus)
        fs === nothing && continue
        tf, df_hz, _, _ = fs
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        freq_i = interp1_linear(Float64.(tf), Float64.(df_hz), tg)

        pm = get_mechanical_power_dev_pu_at_bus(
            results, sys, bus; t_step = t_step, for_csv = true)

        if pm === nothing
            CSV.write(
                joinpath(csvdir, @sprintf("sg_bus%02d.csv", bus)),
                DataFrame(t_s = tg, frequency_deviation_hz = freq_i),
            )
        else
            tpm, dpm, _, _, _, _ = pm
            pm_i = interp1_linear(Float64.(tpm), Float64.(dpm), tg)
            CSV.write(
                joinpath(csvdir, @sprintf("sg_bus%02d.csv", bus)),
                DataFrame(
                    t_s = tg,
                    frequency_deviation_hz = freq_i,
                    delta_p_m_system_pu = pm_i,
                ),
            )
        end
    end

    metadata = DataFrame(
        quantity = [
            "filter_enabled", "frequency_lower_hz", "frequency_upper_hz",
            "alpha_lower", "alpha_upper", "filter_epsilon_s",
            "rocof_tau_s", "prox_step_fraction",
        ],
        value = string.([
            gfl_cbf_case_enabled(sys), GFL_CBF_FLOOR_HZ, GFL_CBF_CEIL_HZ,
            GFL_CBF_ALPHA_LO, GFL_CBF_ALPHA_HI, GFL_CBF_EPS,
            ROCOF_TAU_D, GFL_CBF_PROX_STEP_FRACTION,
        ]),
    )
    CSV.write(joinpath(csvdir, "case_metadata.csv"), metadata)

    return csvdir
end

function _cbf_event_lines!(p, event_times)
    for t_event in event_times
        vline!(p, [t_event];
            color = "#7D8590",
            linestyle = :dot,
            linewidth = 0.8,
            alpha = 0.9,
            label = "")
    end
    return p
end

function _cbf_panel(; ylabel::String = "", legend = :topright)
    return plot(
        xlabel = "",
        ylabel = ylabel,
        legend = legend,
        grid = :y,
        gridalpha = 0.25,
        framestyle = :box,
        xlims = (0.0, TF_SIM),
        widen = false,
        foreground_color_axis = "#343434",
        foreground_color_text = "#252525",
        background_color = :white,
        fontfamily = "Computer Modern",
        guidefontsize = 12,
        tickfontsize = 9,
        legendfontsize = 9,
    )
end

function _cbf_save(p, dir::String, name::String)
    mkpath(dir)
    savefig(p, joinpath(dir, name * ".png"))
    savefig(p, joinpath(dir, name * ".pdf"))
    return p
end

function make_python_style_safety_plots(results, sys::System, plot_root::String,
                                        event_times::Vector{Float64};
                                        t_step::Float64 = 0.25)
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    outdir = joinpath(plot_root, tag)
    mkpath(outdir)
    side = active_safety_side(results, sys)

    signals = Dict{Int,Any}()
    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(
            results, sys, bus; t_step = t_step, for_plots = true)
        s === nothing || (signals[bus] = s)
    end
    isempty(signals) && return nothing

    # GFL frequency response with one active safety boundary.
    p_freq = _cbf_panel(ylabel = "Δf [Hz]", legend = :top)
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        plot!(p_freq, s.t, s.freq_hz;
            color = cbf_bus_color(bus), linewidth = 1.6,
            label = "Bus $bus")
    end
    limit = side == "lower" ? GFL_CBF_FLOOR_HZ : GFL_CBF_CEIL_HZ
    hline!(p_freq, [limit];
        color = "#CC6677", linestyle = :dash, linewidth = 1.0,
        label = side == "lower" ? "Δf_min" : "Δf_max")
    _cbf_event_lines!(p_freq, event_times)
    xlabel!(p_freq, "t [s]")
    _cbf_save(p_freq, outdir, "gfl_frequency_$(side)_bound_emt")

    # Local RoCoF and active CBF bound, one panel per GFL bus.
    rocof_panels = Any[]
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        bound = side == "lower" ? s.lower_bound_hz_s : s.upper_bound_hz_s
        p = _cbf_panel(ylabel = "RoCoF_hat [Hz/s]", legend = :topright)
        plot!(p, s.t, s.rocof_hat_hz_s;
            color = cbf_bus_color(bus), linewidth = 1.55,
            label = "RoCoF_hat")
        plot!(p, s.t, bound;
            color = "#E69F00", linestyle = :dash, linewidth = 1.15,
            label = side == "lower" ? "-α h_min" : "+α h_max")
        _cbf_event_lines!(p, event_times)
        annotate!(p, (0.985 * TF_SIM, maximum(vcat(s.rocof_hz_s, bound)),
            text("Bus $bus", 9, :right)))
        push!(rocof_panels, p)
    end
    if !isempty(rocof_panels)
        p = plot(rocof_panels...;
            layout = (length(rocof_panels), 1),
            size = (1050, 230 * length(rocof_panels)),
            link = :x,
            margin = 4Plots.mm)
        xlabel!(p.subplots[end], "t [s]")
        _cbf_save(p, outdir, "gfl_cbf_rocof_$(side)_constraint_emt")
    end

    # Controller-side CBF residual. Negative values are precisely where
    # the nominal local CBF inequality is not satisfied by the measured
    # PLL/dirty-RoCoF signal.
    residual_panels = Any[]
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        residual =
            side == "lower" ?
            s.lower_margin_hz_s :
            s.upper_margin_hz_s

        p = _cbf_panel(
            ylabel = side == "lower" ?
                "r_lo [Hz/s]" :
                "r_up [Hz/s]",
            legend = false,
        )
        plot!(p, s.t, residual;
            color = cbf_bus_color(bus),
            linewidth = 1.55,
            label = "",
        )
        hline!(p, [0.0];
            color = "#CC6677",
            linestyle = :dash,
            linewidth = 1.0,
            label = "",
        )
        _cbf_event_lines!(p, event_times)
        annotate!(
            p,
            (
                0.985 * TF_SIM,
                maximum(vcat(residual, [0.0])),
                text("Bus $bus", 9, :right),
            ),
        )
        push!(residual_panels, p)
    end

    if !isempty(residual_panels)
        p = plot(
            residual_panels...;
            layout = (length(residual_panels), 1),
            size = (1050, 220 * length(residual_panels)),
            link = :x,
            margin = 4Plots.mm,
        )
        xlabel!(p.subplots[end], "t [s]")
        _cbf_save(
            p,
            outdir,
            "gfl_cbf_$(side)_residual_hat_emt",
        )
    end

    # Active barrier functions.
    barrier_panels = Any[]
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        h = side == "lower" ? s.h_lo_hz : s.h_hi_hz
        p = _cbf_panel(
            ylabel = side == "lower" ? "h_min [Hz]" : "h_max [Hz]",
            legend = false)
        plot!(p, s.t, h;
            color = cbf_bus_color(bus), linewidth = 1.55, label = "")
        hline!(p, [0.0];
            color = "#CC6677", linestyle = :dash, linewidth = 1.0, label = "")
        _cbf_event_lines!(p, event_times)
        annotate!(p, (0.985 * TF_SIM, maximum(h), text("Bus $bus", 9, :right)))
        push!(barrier_panels, p)
    end
    if !isempty(barrier_panels)
        p = plot(barrier_panels...;
            layout = (length(barrier_panels), 1),
            size = (1050, 220 * length(barrier_panels)),
            link = :x,
            margin = 4Plots.mm)
        xlabel!(p.subplots[end], "t [s]")
        _cbf_save(p, outdir, "gfl_$(side)_barrier_functions_emt")
    end

    # Dynamic filter tracking: z_i versus the local proximal target T_i(z).
    tracking_panels = Any[]
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        p = _cbf_panel(ylabel = "z_i [p.u.]", legend = :topright)
        plot!(p, s.t, s.z_sys;
            color = cbf_bus_color(bus),
            linewidth = 1.55,
            label = "z_i",
        )
        plot!(p, s.t, s.target_sys;
            color = "#7E57C2",
            linestyle = :dash,
            linewidth = 1.10,
            label = "T_i(z)",
        )
        hline!(p, [0.0];
            color = "#98A1AA",
            linewidth = 0.7,
            label = "",
        )
        _cbf_event_lines!(p, event_times)
        annotate!(
            p,
            (
                0.985 * TF_SIM,
                maximum(vcat(s.z_sys, s.target_sys, [0.0])),
                text("Bus $bus", 9, :right),
            ),
        )
        push!(tracking_panels, p)
    end

    if !isempty(tracking_panels)
        p = plot(
            tracking_panels...;
            layout = (length(tracking_panels), 1),
            size = (1050, 230 * length(tracking_panels)),
            link = :x,
            margin = 4Plots.mm,
        )
        xlabel!(p.subplots[end], "t [s]")
        _cbf_save(p, outdir, "gfl_filter_tracking_emt")
    end

    # Filter state z_i.
    z_panels = Any[]
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        p = _cbf_panel(ylabel = "z_i [p.u.]", legend = false)
        plot!(p, s.t, s.z_sys;
            color = cbf_bus_color(bus), linewidth = 1.55, label = "")
        hline!(p, [0.0]; color = "#98A1AA", linewidth = 0.7, label = "")
        _cbf_event_lines!(p, event_times)
        annotate!(p, (0.985 * TF_SIM, maximum(vcat(s.z_sys, [0.0])),
            text("Bus $bus", 9, :right)))
        push!(z_panels, p)
    end
    if !isempty(z_panels)
        p = plot(z_panels...;
            layout = (length(z_panels), 1),
            size = (1050, 220 * length(z_panels)),
            link = :x,
            margin = 4Plots.mm)
        xlabel!(p.subplots[end], "t [s]")
        _cbf_save(p, outdir, "gfl_filter_state_emt")
    end

    # GFL active-power decomposition.
    p_panels = Any[]
    for bus in GFL_BUSES
        haskey(signals, bus) || continue
        s = signals[bus]
        p = _cbf_panel(ylabel = "Δp_F [p.u.]", legend = :topright)
        plot!(p, s.t, s.Δp_ffr_sys;
            color = "#7D8590", linestyle = :dash, linewidth = 1.2,
            label = "Δp_F,i^FFR")
        plot!(p, s.t, s.Δp_sys;
            color = cbf_bus_color(bus), linewidth = 1.6,
            label = "Δp_F,i")
        if gfl_cbf_case_enabled(sys)
            plot!(p, s.t, s.z_cmd_sys;
                color = cbf_bus_color(bus), linestyle = :dot,
                linewidth = 1.1, label = "z_i")
        end
        _cbf_event_lines!(p, event_times)
        annotate!(p, (0.985 * TF_SIM, maximum(vcat(s.Δp_sys, s.Δp_ffr_sys)),
            text("Bus $bus", 9, :right)))
        push!(p_panels, p)
    end
    if !isempty(p_panels)
        p = plot(p_panels...;
            layout = (length(p_panels), 1),
            size = (1050, 230 * length(p_panels)),
            link = :x,
            margin = 4Plots.mm)
        xlabel!(p.subplots[end], "t [s]")
        _cbf_save(p, outdir, "gfl_power_decomposition_emt")
    end

    make_python_style_filter_overlay(
        results, sys, outdir, event_times;
        t_step = t_step,
    )
    make_python_style_gfl_aggregate_plot(
        results, sys, outdir, event_times;
        t_step = t_step,
    )
    make_python_style_sg_plots(
        results, sys, outdir, event_times;
        t_step = t_step,
    )

    return outdir
end

function make_python_style_filter_overlay(
    results,
    sys::System,
    outdir::String,
    event_times::Vector{Float64};
    t_step::Float64 = 0.25,
)
    p = _cbf_panel(ylabel = "z_i [p.u.]", legend = :top)
    n = 0

    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(
            results, sys, bus;
            t_step = t_step,
            for_plots = true,
        )
        s === nothing && continue
        plot!(p, s.t, s.z_sys;
            color = cbf_bus_color(bus),
            linewidth = 1.55,
            label = "Bus $bus",
        )
        n += 1
    end

    n == 0 && return nothing
    hline!(p, [0.0]; color = "#98A1AA", linewidth = 0.7, label = "")
    _cbf_event_lines!(p, event_times)
    xlabel!(p, "t [s]")
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    return _cbf_save(p, outdir, "gfl_filter_state_overlay_$(tag)_emt")
end

function make_python_style_gfl_aggregate_plot(
    results,
    sys::System,
    outdir::String,
    event_times::Vector{Float64};
    t_step::Float64 = 0.25,
)
    t = nothing
    sum_ffr = nothing
    sum_z = nothing
    sum_p = nothing

    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(
            results, sys, bus;
            t_step = t_step,
            for_plots = true,
        )
        s === nothing && continue

        if t === nothing
            t = s.t
            sum_ffr = zeros(length(t))
            sum_z = zeros(length(t))
            sum_p = zeros(length(t))
        end

        sum_ffr .+= s.Δp_ffr_sys
        sum_z .+= s.z_cmd_sys
        sum_p .+= s.Δp_sys
    end

    t === nothing && return nothing

    p1 = _cbf_panel(ylabel = "ΔP_F [p.u.]", legend = :topright)
    plot!(p1, t, sum_ffr;
        color = "#0072B2",
        linestyle = :dash,
        linewidth = 1.40,
        label = "ΔP_F^FFR",
    )
    plot!(p1, t, sum_p;
        color = "#009E73",
        linewidth = 1.70,
        label = "ΔP_F",
    )
    if gfl_cbf_case_enabled(sys)
        plot!(p1, t, sum_p;
            linealpha = 0.0,
            fillrange = sum_ffr,
            fillcolor = "#F3CD85",
            fillalpha = 0.50,
            label = "sum z_i",
        )
    end
    _cbf_event_lines!(p1, event_times)

    p2 = _cbf_panel(ylabel = "ΔP_F [p.u.]", legend = :topright)
    plot!(p2, t, sum_ffr;
        color = "#56B4E9",
        linewidth = 1.60,
        label = "ΔP_F^FFR",
    )
    plot!(p2, t, sum_z;
        color = "#D55E00",
        linewidth = 1.60,
        label = "sum z_i",
    )
    hline!(p2, [0.0]; color = "#98A1AA", linewidth = 0.7, label = "")
    _cbf_event_lines!(p2, event_times)
    xlabel!(p2, "t [s]")

    panel = plot(
        p1, p2;
        layout = (2, 1),
        size = (1050, 560),
        link = :x,
    )
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    return _cbf_save(panel, outdir, "gfl_power_aggregate_$(tag)_emt")
end

function make_python_style_sg_plots(
    results,
    sys::System,
    outdir::String,
    event_times::Vector{Float64};
    t_step::Float64 = 0.25,
)
    freq_panels = Any[]
    pm_panels = Any[]

    for bus in SG_BUSES
        fs = frequency_deviation_hz_at_bus(results, sys, bus)
        if fs !== nothing
            tf, df_hz, _, _ = fs
            tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
            df_i = interp1_linear(Float64.(tf), Float64.(df_hz), tg)

            p = _cbf_panel(ylabel = "Δf [Hz]", legend = false)
            plot!(p, tg, df_i;
                color = cbf_bus_color(bus),
                linewidth = 1.55,
                label = "",
            )
            _cbf_event_lines!(p, event_times)
            annotate!(p, (
                0.985 * TF_SIM,
                maximum(vcat(df_i, [0.0])),
                text("Bus $bus", 9, :right),
            ))
            push!(freq_panels, p)
        end

        pm = get_mechanical_power_dev_pu_at_bus(
            results, sys, bus;
            t_step = t_step,
            for_plots = true,
        )
        if pm !== nothing
            tpm, dpm, _, _, _, _ = pm
            p = _cbf_panel(ylabel = "Δp_m [p.u.]", legend = false)
            plot!(p, tpm, dpm;
                color = cbf_bus_color(bus),
                linewidth = 1.55,
                label = "",
            )
            hline!(p, [0.0]; color = "#98A1AA", linewidth = 0.7, label = "")
            _cbf_event_lines!(p, event_times)
            annotate!(p, (
                0.985 * TF_SIM,
                maximum(vcat(dpm, [0.0])),
                text("Bus $bus", 9, :right),
            ))
            push!(pm_panels, p)
        end
    end

    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"

    if !isempty(freq_panels)
        pf = plot(
            freq_panels...;
            layout = (length(freq_panels), 1),
            size = (1050, 220 * length(freq_panels)),
            link = :x,
        )
        xlabel!(pf.subplots[end], "t [s]")
        _cbf_save(pf, outdir, "sg_frequency_$(tag)_emt")
    end

    if !isempty(pm_panels)
        pp = plot(
            pm_panels...;
            layout = (length(pm_panels), 1),
            size = (1050, 220 * length(pm_panels)),
            link = :x,
        )
        xlabel!(pp.subplots[end], "t [s]")
        _cbf_save(pp, outdir, "sg_mechanical_power_$(tag)_emt")
    end

    return nothing
end


function _julia_cbf_event_lines!(p, event_times::Vector{Float64})
    for tev in event_times
        vline!(
            p, [tev];
            lw = 1.0,
            ls = :dash,
            c = :black,
            alpha = 0.35,
            label = "",
        )
    end
    return p
end

function _save_cbf_png_pdf(p, dir::String, base::String)
    mkpath(dir)
    savefig(p, joinpath(dir, base * ".png"))
    savefig(p, joinpath(dir, base * ".pdf"))
    return p
end

function plot_gfl_cbf_headroom(
    results,
    sys::System,
    outdir::String,
    event_times::Vector{Float64};
    t_step::Float64 = 0.25,
)
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    dir = joinpath(outdir, tag, "headroom")
    mkpath(dir)

    for bus in GFL_BUSES
        hr = gfl_headroom_at_bus(results, sys, bus)
        cb = safety_filter_signals_at_bus(
            results, sys, bus;
            t_step = t_step,
            for_plots = true,
        )
        (hr === nothing || cb === nothing) && continue
        all(isnan, hr.P_head) && continue

        t = cb.t
        Pnow = interp1_linear(hr.t, hr.P_now, t)
        Pmax = interp1_linear(hr.t, hr.P_max, t)
        Pmin = fill(GFL_P_MIN, length(t))
        zdev = cb.z_cmd_dev
        Prequest = Pnow .+ zdev

        p = plot(
            xlabel = "Time [s]",
            ylabel = "Power [p.u., device base]",
            legend = :outertopright,
            grid = false,
            framestyle = :box,
            xlims = (0.0, TF_SIM),
            widen = false,
            size = (1450, 650),
            title = @sprintf("GFL active-power headroom — Bus %d", bus),
        )

        plot!(
            p, t, Pmax;
            lw = 0.0,
            fillrange = Pnow,
            fillalpha = 0.12,
            fillcolor = bus_plot_color(bus),
            label = "Injection headroom",
        )
        plot!(
            p, t, Pnow;
            lw = 0.0,
            fillrange = Pmin,
            fillalpha = 0.10,
            fillcolor = :steelblue,
            label = "Shed headroom",
        )
        plot!(
            p, t, Prequest;
            lw = 0.0,
            fillrange = Pnow,
            fillalpha = 0.28,
            fillcolor = :orange,
            label = "Requested z_i",
        )
        plot!(p, t, Pnow; lw = 2.3, color = bus_plot_color(bus), label = "P")
        plot!(p, t, Prequest; lw = 1.8, ls = :dot, color = :orange, label = "P + z_i")
        plot!(p, t, Pmax; lw = 1.4, ls = :dash, color = :red, label = "P_max")
        plot!(p, t, Pmin; lw = 1.4, ls = :dash, color = :steelblue, label = "P_min")
        _julia_cbf_event_lines!(p, event_times)
        _save_cbf_png_pdf(p, dir, @sprintf("gfl_headroom_bus%02d", bus))
    end
    return nothing
end

function plot_gfl_cbf_diagnostics(
    results,
    sys::System,
    outdir::String,
    event_times::Vector{Float64};
    t_step::Float64 = 0.25,
)
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    dir = joinpath(outdir, tag)
    mkpath(dir)

    # Two-sided safety-band plot retained as an engineering diagnostic.
    p_freq = plot(
        xlabel = "Time [s]",
        ylabel = "Δf [Hz]",
        legend = :outertopright,
        grid = false,
        framestyle = :box,
        xlims = (0.0, TF_SIM),
        widen = false,
        size = (1450, 650),
        title = "GFL frequency and two-sided safety band",
    )
    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(
            results, sys, bus;
            t_step = t_step,
            for_plots = true,
        )
        s === nothing && continue
        plot!(p_freq, s.t, s.freq_hz; lw = 2.2, color = bus_plot_color(bus), label = "Bus $bus")
    end
    hline!(p_freq, [GFL_CBF_FLOOR_HZ]; lw = 1.5, ls = :dash, c = :red, label = "lower limit")
    hline!(p_freq, [GFL_CBF_CEIL_HZ]; lw = 1.5, ls = :dash, c = :blue, label = "upper limit")
    _julia_cbf_event_lines!(p_freq, event_times)
    _save_cbf_png_pdf(p_freq, dir, "gfl_frequency_two_sided")

    # Local RoCoF estimate.
    p_rocof = plot(
        xlabel = "Time [s]",
        ylabel = "RoCoF_hat [Hz/s]",
        legend = :outertopright,
        grid = false,
        framestyle = :box,
        xlims = (0.0, TF_SIM),
        widen = false,
        size = (1450, 650),
        title = "GFL local PLL dirty-derivative RoCoF",
    )
    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(results, sys, bus)
        s === nothing && continue
        plot!(p_rocof, s.t, s.rocof_hz_s; lw = 2.2, color = bus_plot_color(bus), label = "Bus $bus")
    end
    hline!(p_rocof, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    _julia_cbf_event_lines!(p_rocof, event_times)
    _save_cbf_png_pdf(p_rocof, dir, "gfl_local_rocof")

    plot_gfl_cbf_headroom(
        results, sys, outdir, event_times;
        t_step = t_step,
    )
    save_gfl_headroom_csvs(
        results, sys, joinpath(outdir, "csv");
        t_step = t_step,
    )

    return dir
end

function write_gfl_cbf_report(
    results,
    sys::System,
    outdir::String,
    event_summary::String,
)
    tag = gfl_cbf_case_enabled(sys) ? "filter_on" : "filter_off"
    dir = joinpath(outdir, tag)
    mkpath(dir)
    path = joinpath(dir, "gfl_cbf_validation_report.txt")

    open(path, "w") do io
        println(io, "="^104)
        println(io, "GFL DYNAMIC LOCAL FREQUENCY-SAFETY FILTER — IEEE-39 EMT")
        println(io, "Generated: $(Dates.now())")
        println(io, "Disturbance: $event_summary")
        @printf(
            io,
            "Band=[%+.4f,%+.4f] Hz | alpha=(%.3f, %.3f) | epsilon=%.3f s | tau_d=%.3f s | filter=%s\n",
            GFL_CBF_FLOOR_HZ,
            GFL_CBF_CEIL_HZ,
            GFL_CBF_ALPHA_LO,
            GFL_CBF_ALPHA_HI,
            GFL_CBF_EPS,
            ROCOF_TAU_D,
            gfl_cbf_case_enabled(sys) ? "ON" : "OFF",
        )
        println(io, "Sensitivity source: $(GFL_CBF_DESIGN_SOURCE_REF[])")
        println(io, "Headroom enforcement: $(ENFORCE_HEADROOM)")
        println(io, "="^104)

        for bus in GFL_BUSES
            s = safety_filter_signals_at_bus(results, sys, bus)
            s === nothing && continue
            @printf(io, "Bus %d: k_i=%.9g s\n", bus, s.k_prox)
            @printf(io, "  nadir=%+.6f Hz | zenith=%+.6f Hz\n",
                    minimum(s.freq_hz), maximum(s.freq_hz))
            @printf(io, "  peak |z_i state|=%.8e pu system\n",
                    maximum(abs.(s.z_sys)))
            @printf(io, "  peak |z_i command|=%.8e pu system\n",
                    maximum(abs.(s.z_cmd_sys)))
            @printf(io, "  peak |T_i(z)|=%.8e pu system\n",
                    maximum(abs.(s.target_sys)))
            @printf(io, "  min lower CBF margin (PLL dirty RoCoF)=%+.8e Hz/s\n",
                    minimum(s.lower_margin_hz_s))
            @printf(io, "  min upper CBF margin (PLL dirty RoCoF)=%+.8e Hz/s\n",
                    minimum(s.upper_margin_hz_s))

            idx = findfirst(abs.(s.target_sys) .> GFL_CBF_DORMANCY_TOL_SYS_PU)
            if idx === nothing
                println(io, "  activation: dormant; zero remained in [a_i,b_i].")
            else
                branch =
                    s.a_sys[idx] > GFL_CBF_DORMANCY_TOL_SYS_PU ? "lower" :
                    (s.b_sys[idx] < -GFL_CBF_DORMANCY_TOL_SYS_PU ?
                     "upper" : "dynamic-memory")
                @printf(
                    io,
                    "  first activation: t=%.6f s | branch=%s | a=%+.8e | b=%+.8e | r_lo=%+.8e Hz/s | r_up=%+.8e Hz/s\n",
                    s.t[idx],
                    branch,
                    s.a_sys[idx],
                    s.b_sys[idx],
                    s.lower_margin_hz_s[idx],
                    s.upper_margin_hz_s[idx],
                )
            end

            hr = gfl_headroom_at_bus(results, sys, bus)
            if hr !== nothing
                @printf(io, "  peak |I|=%.6f pu, I_rated=%.6f pu\n",
                        maximum(hr.Imag), hr.I_rated)
                if !all(isnan, hr.P_head)
                    @printf(io, "  minimum injection headroom=%.6f pu device\n",
                            minimum(hr.P_head))
                    @printf(io, "  minimum shed headroom=%.6f pu device\n",
                            minimum(hr.P_now .- GFL_P_MIN))
                end
            end
            println(io)
        end
        println(io, "="^104)
    end

    return path
end

function validate_gfl_cbf_internal_consistency!(results, sys::System)
    if !gfl_cbf_case_enabled(sys)
        println("\nGFL safety-filter internal consistency: skipped (stock GFL plant, CBF OFF).")
        return nothing
    end
    println("\nGFL safety-filter internal consistency")
    println(
        "Bus | max |z_i| state | max |z_i| command | max |T_i(z)| | " *
        "min r_lo [Hz/s] | min r_up [Hz/s]"
    )
    println("-"^116)

    tol_z = GFL_CBF_DORMANCY_TOL_SYS_PU
    tol_endpoint = 5e-8
    tol_target = 5e-8
    tol_margin_hz_s = 1e-8

    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(results, sys, bus)
        s === nothing && continue

        zpeak = maximum(abs.(s.z_sys))
        zcmdpeak = maximum(abs.(s.z_cmd_sys))
        tpeak = maximum(abs.(s.target_sys))
        min_lo = minimum(s.lower_margin_hz_s)
        min_up = minimum(s.upper_margin_hz_s)

        # From the measurement-based local realization:
        #   a_i = z_i - k_i r_i^lo
        #   b_i = z_i + k_i r_i^up,
        # where r_i^lo and r_i^up use the same PLL dirty-RoCoF signal
        # that drives the online filter.
        a_expected =
            s.z_sys .- s.k_prox .* (s.lower_margin_hz_s ./ F_BASE)
        b_expected =
            s.z_sys .+ s.k_prox .* (s.upper_margin_hz_s ./ F_BASE)

        endpoint_error = max(
            maximum(abs.(s.a_sys .- a_expected)),
            maximum(abs.(s.b_sys .- b_expected)),
        )
        endpoint_error <= tol_endpoint ||
            error(
                "GFL bus $bus failed the local CBF endpoint identity: " *
                "max error=$(endpoint_error)."
            )

        target_expected = [
            project_zero_to_interval(ai, bi)
            for (ai, bi) in zip(s.a_sys, s.b_sys)
        ]
        target_error = maximum(abs.(s.target_sys .- target_expected))
        target_error <= tol_target ||
            error(
                "GFL bus $bus failed the local projection identity: " *
                "max error=$(target_error)."
            )

        @printf(
            "%3d | %15.7e | %17.7e | %13.7e | %+.7e | %+.7e\n",
            bus, zpeak, zcmdpeak, tpeak, min_lo, min_up,
        )

        if !gfl_cbf_case_enabled(sys) && zcmdpeak > 10.0 * tol_z
            error(
                "Safety filter is OFF but z_i command is nonzero at GFL bus $bus."
            )
        end

        if gfl_cbf_case_enabled(sys) &&
           tpeak <= tol_z &&
           zpeak > 10.0 * tol_z
            error(
                "T_i(z) is dormant but z_i is not zero at GFL bus $bus."
            )
        end

        idx = findfirst(abs.(s.target_sys) .> tol_z)
        if idx === nothing
            println(
                "      dormant: zero remained in the local proximal interval."
            )
            continue
        end

        branch =
            s.a_sys[idx] > tol_z ? "lower" :
            (s.b_sys[idx] < -tol_z ? "upper" : "dynamic-memory")

        @printf(
            "      first T_i(z) != 0: t=%.6f s, branch=%s, z=%+.7e, a=%+.7e, b=%+.7e, r_lo=%+.7e Hz/s, r_up=%+.7e Hz/s, RoCoF_hat=%+.7e Hz/s\n",
            s.t[idx],
            branch,
            s.z_sys[idx],
            s.a_sys[idx],
            s.b_sys[idx],
            s.lower_margin_hz_s[idx],
            s.upper_margin_hz_s[idx],
            s.rocof_hat_hz_s[idx],
        )

        # At the first activation z_i should still be approximately zero.
        # Therefore the lower branch can activate only if r_lo < 0, and
        # the upper branch only if r_up < 0.
        if abs(s.z_sys[idx]) <= 10.0 * tol_z
            if s.a_sys[idx] > tol_z &&
               s.lower_margin_hz_s[idx] >= -tol_margin_hz_s
                error(
                    "GFL bus $bus activated the lower branch although the " *
                    "controller-side lower CBF residual is nonnegative."
                )
            end
            if s.b_sys[idx] < -tol_z &&
               s.upper_margin_hz_s[idx] >= -tol_margin_hz_s
                error(
                    "GFL bus $bus activated the upper branch although the " *
                    "controller-side upper CBF residual is nonnegative."
                )
            end
        end
    end

    return true
end

function print_safety_filter_summary(results, sys::System)
    side = active_safety_side(results, sys)
    println("\n" * "="^94)
    println("DYNAMIC LOCAL FREQUENCY-SAFETY FILTER")
    println("="^94)
    @printf("Mode: %s | active safety side: %s | band=[%+.4f,%+.4f] Hz\n",
            gfl_cbf_case_enabled(sys) ? "ON" : "OFF", side,
            GFL_CBF_FLOOR_HZ, GFL_CBF_CEIL_HZ)
    @printf("alpha_lo=%.3f | alpha_hi=%.3f | epsilon=%.3f s | tau_d=%.3f s\n",
            GFL_CBF_ALPHA_LO, GFL_CBF_ALPHA_HI, GFL_CBF_EPS, ROCOF_TAU_D)
    println(
        "Bus | nadir [Hz] | zenith [Hz] | peak |z_i| [pu] | " *
        "peak |T_i(z)| [pu] | min r_lo [Hz/s] | min r_up [Hz/s]"
    )
    println("-"^126)

    for bus in GFL_BUSES
        s = safety_filter_signals_at_bus(results, sys, bus)
        s === nothing && continue
        z_peak = maximum(abs.(s.z_sys))
        target_peak = maximum(abs.(s.target_sys))
        @printf(
            "%3d | %+.6f | %+.6f | %.8e | %.8e | %+.8e | %+.8e\n",
            bus, minimum(s.freq_hz), maximum(s.freq_hz),
            z_peak, target_peak,
            minimum(s.lower_margin_hz_s),
            minimum(s.upper_margin_hz_s),
        )
        if gfl_cbf_case_enabled(sys) && target_peak <= 1e-10 && z_peak > 1e-8
            @warn "CBF target is dormant but z_i is not numerically zero" bus=bus z_peak=z_peak
        end
    end
    println("="^94 * "\n")
end


function run_gfl_cbf_outputs!(
    results,
    sys::System,
    plotsdir::String,
    csvdir::String,
    event_times::Vector{Float64},
    event_summary::String;
    t_step::Float64 = 0.25,
)
    validate_gfl_cbf_internal_consistency!(results, sys)

    save_safety_filter_csvs(
        results, sys, joinpath(plotsdir, "python_style_cbf", "csv");
        t_step = t_step,
    )

    print_safety_filter_summary(results, sys)

    if MAKE_JULIA_CBF_DIAGNOSTIC_PLOTS
        diagnostics_root = joinpath(plotsdir, "gfl_cbf")
        plot_gfl_cbf_diagnostics(
            results, sys, diagnostics_root, event_times;
            t_step = t_step,
        )
        write_gfl_cbf_report(
            results, sys, diagnostics_root, event_summary,
        )
    end

    if MAKE_PYTHON_STYLE_EMT_PLOTS
        make_python_style_safety_plots(
            results,
            sys,
            joinpath(plotsdir, "python_style_cbf"),
            event_times;
            t_step = t_step,
        )
    end

    return nothing
end
