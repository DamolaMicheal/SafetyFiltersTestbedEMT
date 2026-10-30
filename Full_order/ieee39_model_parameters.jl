# IEEE 39-bus model parameters 
using PowerSystems
using PowerSimulationsDynamics
using PowerFlows
using Sundials
using Plots
using LinearAlgebra
using Statistics
using Dates
using Printf
using Logging
using CSV
using DataFrames
using SparseArrays
const PSY  = PowerSystems
const PSID = PowerSimulationsDynamics

#Case data
include("case39_matpower.jl")

const SYS_BASE_MVA = 100.0
const F_BASE       = 60.0
const Ω_BASE_RAD_S = 2π * F_BASE

const TF_SIM              = 6.0
const EXPORT_DT           = 0.01
const RESAMPLE_FOR_CSV    = true
const RESAMPLE_FOR_PLOTS  = true
const GFM_TA_DEFAULT      = 2.0 * 4.2
const TG_EPS              = 0.001
const TG_TS               = 0.5
const PSS_KS              = 0.0

#Solver settings
const SOLVER_ABSTOL    = 1e-2
const SOLVER_RELTOL    = 1e-2
const SOLVER_MAX_ORDER = 5
const SOLVER_INIT_DT   = 1e-3
const SOLVER_DTMAX     = 1e-3

#GFL PLL gains
# Tune ONLY the values in these two maps to change the PLL gains at each GFL bus.
# All other model parameters and logic are unchanged.
const GFL_PLL_KP_map_ext = Dict{Int,Float64}(
    30 => 0.03,
    34 => 0.054,
    36 => 0.054,
    31 => 0.054,
    38 => 0.015,
)

const GFL_PLL_KI_map_ext = Dict{Int,Float64}(
    30 => 0.3,
    34 => 1.69,
    36 => 1.69,
    31 => 1.69,
    38 => 2.69,
)


function getGflPllKp(bus::Int)
    haskey(GFL_PLL_KP_map_ext, bus) ||
        error("No GFL PLL Kp is defined for bus $bus. Add it to GFL_PLL_KP_map_ext.")
    return GFL_PLL_KP_map_ext[bus]
end

function getGflPllKi(bus::Int)
    haskey(GFL_PLL_KI_map_ext, bus) ||
        error("No GFL PLL Ki is defined for bus $bus. Add it to GFL_PLL_KI_map_ext.")
    return GFL_PLL_KI_map_ext[bus]
end

#GFL active-power / frequency-droop gain
#Bus 30 is the only GFL bus in this case; its Kω is fixed at 75.0.
const GFL_KW_DEFAULT = 20.0
const GFL_KW_map_ext = Dict{Int,Float64}(
    30 => 20.0,
)
getGflKw(bus::Int) = get(GFL_KW_map_ext, bus, GFL_KW_DEFAULT)

# Droops
const R_SG_PU  = 0.05   # 0.043
const R_GFM_PU = 0.08

#Bus partition
const SG_BUSES  = [32, 33, 35, 37, 39]
const GFM_BUSES = Int[]
const GFL_BUSES = [30, 34, 36, 31, 38]

#Plot selections
#Buses on the combined frequency plot 
const FREQ_OVERLAY_BUSES  = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
#Buses on the combined angle plots
const ANGLE_OVERLAY_BUSES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
# SG buses on the combined mechanical-power plots
const SG_PM_OVERLAY_BUSES = SG_BUSES

#Buses used by theterminal P/Q panels
const PLOT_BUSES_EXT = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

#Voltage plot settings
const ABS_VOLTAGE_YLIM   = (0.94, 1.08)
const ABS_VOLTAGE_YTICKS = 0.94:0.02:1.08

const BUS_PLOT_COLORS_EXT = Dict{Int,String}(
    30 => "#1f77b4",
    31 => "#9467bd",
    32 => "#ff7f0e",
    33 => "#8c564b",
    34 => "#d62728",
    35 => "#e377c2",
    36 => "#7f7f7f",
    37 => "#bcbd22",
    38 => "#17becf",
    39 => "#2ca02c",
)

bus_plot_color(bus::Int) = get(BUS_PLOT_COLORS_EXT, bus, "#333333")

#GFM parameter maps
const GFM_TA_map_ext = Dict{Int,Float64}(
    31 =>  2 * 3.030,
    32 =>  2 * 3.580,
    35 =>  2*3.580,
    38 =>  2 * 3.450,
)
getGfmTa(bus::Int) = get(GFM_TA_map_ext, bus, GFM_TA_DEFAULT)

const GFM_KW_DEFAULT = 12.5
const GFM_KW_map_ext = Dict{Int,Float64}(
    31 => GFM_KW_DEFAULT,
    32 => GFM_KW_DEFAULT,
    35 => GFM_KW_DEFAULT,
    38 => GFM_KW_DEFAULT,
)
getGfmKw(bus::Int) = get(GFM_KW_map_ext, bus, GFM_KW_DEFAULT)

const GFM_KD_DEFAULT = 0.0
const GFM_KD_map_ext = Dict{Int,Float64}(
    31 => GFM_KD_DEFAULT,
    32 => GFM_KD_DEFAULT,
    35 => GFM_KD_DEFAULT,
    38 => GFM_KD_DEFAULT,
)
getGfmKd(bus::Int) = get(GFM_KD_map_ext, bus, GFM_KD_DEFAULT)

#Disturbance settings
#Active-power load step with the fault disabled.
const BUS_FAULT_ENABLE   = false   #fault OFF
const LOAD_CHANGE_ENABLE = true    # load step ON
const EXPORT_SIM_CSVS    = true    #write per-bus simulation CSVs

#:line_fault_line_trip  -> 3φ-G fault on line, clear by tripping same line
#:bus_fault_line_trip   -> 3φ-G fault at bus, clear by tripping selected line
const FAULT_SCENARIO = :bus_fault_line_trip

#line-fault scenario settings 
const FAULT_LINE_FROM       = 15
const FAULT_LINE_TO         = 16
const FAULT_LOCATION_FROM   = 0.5
const FAULT_APPLY_TIME_S    = 0.2
const FAULT_CLEAR_TIME_S    = 0.3

#Bus-fault scenario settings
const BUS_FAULT_BUS             = 21
const BUS_FAULT_TRIP_LINE_FROM  = 21
const BUS_FAULT_TRIP_LINE_TO    = 22
const BUS_FAULT_DURATION_S      = 0.1
const BUS_FAULT_CLEAR_TIME_S    = FAULT_APPLY_TIME_S + BUS_FAULT_DURATION_S

const FAULT_R_PU            = 0.007
const FAULT_X_PU            = 0.0

# network P/Q settings
const SAVE_DIAGNOSTIC_NETWORK_PQ = true
const NETWORK_PQ_ALL_BUSES    = collect(1:39)
const NETWORK_PQ_EXPORT_BUSES = collect(30:39)

# Extra load-bus exports
const IEEE39_LOAD_BUSES_ALL = [1, 3, 4, 7, 8, 9, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 35]
const IEEE39_EXTRA_LOAD_BUSES = [b for b in IEEE39_LOAD_BUSES_ALL if !(b in collect(30:39))]
const EXPORT_EXTRA_LOAD_BUS_CSVS = true

# Load-change events
#Each :pu_map entry is a perturbation-load setpoint on the system base.
#250 MW active-power step at bus 35 at t = 0.25 s.
const LOAD_CHANGE_EVENTS = [
    Dict(
        :buses        => [35],
        :event_time_s => 0.25,
        :pu_map       => Dict(
            35 => 300.0 / SYS_BASE_MVA,
        ),
    ),
    Dict(
        :buses        => [2],
        :event_time_s => 0.25,
        :pu_map       => Dict(
            2 => 150.0 / SYS_BASE_MVA,
        ),
    ),
    Dict(
        :buses        => [29],
        :event_time_s => 0.25,
        :pu_map       => Dict(
            29 => 75.0 / SYS_BASE_MVA,
        ),
    ),
]

#Inertia and damping
const H_ALL_map_ext = Dict{Int,Float64}(
    30 => 4.200,
    31 => 3.030,
    32 => 3.580/2,
    33 => 2.860,
    34 => 4.333,   # 2.533
    35 => 3.580/2,
    36 => 3.480,
    37 => 2.430,
    38 => 3.450,
    39 => 5.000/2,
)





getH(bus::Int) = haskey(H_ALL_map_ext, bus) ? H_ALL_map_ext[bus] : error("No inertia H found for bus $bus")

const D_SCALE = 1.0
const D_SG_DEFAULT = 40.0 / D_SCALE
const D_SG_map_ext = Dict{Int,Float64}(
    33 => 40.0 / D_SCALE,
    34 => 40.0 / D_SCALE,
    36 => 40.0 / D_SCALE,
    37 => 40.0 / D_SCALE,
    39 => 48.0 / D_SCALE,
)
function getDsg(bus::Int)
    haskey(D_SG_map_ext, bus) && return D_SG_map_ext[bus]
    @warn "No SG damping D found for bus; using default D = 40.0/D_SCALE" bus = bus
    return 40.0 / D_SCALE
end

#Asset helpers
is_sg_bus(bus::Int)  = bus in SG_BUSES
is_gfm_bus(bus::Int) = bus in GFM_BUSES
is_gfl_bus(bus::Int) = bus in GFL_BUSES

function model_label(bus::Int)
    if is_sg_bus(bus)
        return "SG"
    elseif is_gfm_bus(bus)
        return "GFM"
    elseif is_gfl_bus(bus)
        return "GFL"
    else
        return "Unknown"
    end
end

# Validation
function validate_partitions!(sg::Vector{Int}, gfm::Vector{Int}, gfl::Vector{Int})
    sgset  = Set(sg)
    gfmset = Set(gfm)
    gflset = Set(gfl)

    if 39 ∉ sgset
        error("Bus 39 must be in SG_BUSES.")
    end
    if (39 ∈ gfmset) || (39 ∈ gflset)
        error("Bus 39 must NOT be in GFM_BUSES or GFL_BUSES.")
    end

    ov1 = intersect(sgset, gfmset)
    ov2 = intersect(sgset, gflset)
    ov3 = intersect(gfmset, gflset)
    if !isempty(ov1) || !isempty(ov2) || !isempty(ov3)
        error("Partitions overlap. SG∩GFM=$(collect(ov1)), SG∩GFL=$(collect(ov2)), GFM∩GFL=$(collect(ov3))")
    end

    allowed = Set(30:39)
    if !issubset(sgset, allowed) || !issubset(gfmset, allowed) || !issubset(gflset, allowed)
        error("SG_BUSES/GFM_BUSES/GFL_BUSES must be within 30..39.")
    end

    allowed_gfl = Set(30:38)
    if !issubset(gflset, allowed_gfl)
        bad = collect(setdiff(gflset, allowed_gfl))
        error("GFL_BUSES must be within 30..38 only. Bad=$(bad)")
    end

    cover = union(union(sgset, gfmset), gflset)
    if cover != allowed
        missing = collect(setdiff(allowed, cover))
        extra   = collect(setdiff(cover, allowed))
        error("Partitions must cover ALL buses 30..39 exactly. Missing=$(missing), Extra=$(extra)")
    end

    return true
end

# Load-event helpers
struct LoadEventSpec
    buses::Vector{Int}
    event_time_s::Float64
    p_pu_map::Dict{Int,Float64}
    q_pu_map::Dict{Int,Float64}
end

struct RealizedLoadStep
    bus::Int
    t_step::Float64
    ΔP_MW::Float64
    ΔQ_MVAr::Float64
    refP_MW::Float64
    refQ_MVAr::Float64
    refP_pu::Float64
    refQ_pu::Float64
    deltaP_pu::Float64
    deltaQ_pu::Float64
end

function _event_p_map(ev)
    if haskey(ev, :p_pu_map)
        return ev[:p_pu_map]
    elseif haskey(ev, :pu_map)
        return ev[:pu_map]
    end
    return Dict{Int,Float64}()
end

function _event_q_map(ev)
    haskey(ev, :q_pu_map) && return ev[:q_pu_map]
    return Dict{Int,Float64}()
end

function _to_int_float_dict(raw_map)
    out = Dict{Int,Float64}()
    for (k, v) in raw_map
        out[Int(k)] = Float64(v)
    end
    return out
end

function load_event_specs_from_config()
    specs = LoadEventSpec[]
    for ev in LOAD_CHANGE_EVENTS
        buses = Vector{Int}(ev[:buses])
        t     = Float64(ev[:event_time_s])
        p_map = _to_int_float_dict(_event_p_map(ev))
        q_map = _to_int_float_dict(_event_q_map(ev))
        push!(specs, LoadEventSpec(buses, t, p_map, q_map))
    end
    sort!(specs, by = x -> x.event_time_s)
    return specs
end

function build_realized_load_steps_from_events()
    specs = load_event_specs_from_config()
    prevP_MW   = Dict{Int,Float64}()
    prevQ_MVAr = Dict{Int,Float64}()
    realized_steps = RealizedLoadStep[]
    for ev in specs
        for b in ev.buses
            refP_pu = get(ev.p_pu_map, b, 0.0)
            refQ_pu = get(ev.q_pu_map, b, 0.0)
            refP_MW   = refP_pu * SYS_BASE_MVA
            refQ_MVAr = refQ_pu * SYS_BASE_MVA
            oldP_MW   = get(prevP_MW, b, 0.0)
            oldQ_MVAr = get(prevQ_MVAr, b, 0.0)
            ΔP_MW   = refP_MW - oldP_MW
            ΔQ_MVAr = refQ_MVAr - oldQ_MVAr
            prevP_MW[b]   = refP_MW
            prevQ_MVAr[b] = refQ_MVAr
            push!(realized_steps,
                RealizedLoadStep(b, ev.event_time_s, ΔP_MW, ΔQ_MVAr, refP_MW, refQ_MVAr,
                                 refP_pu, refQ_pu, ΔP_MW / SYS_BASE_MVA, ΔQ_MVAr / SYS_BASE_MVA))
        end
    end
    sort!(realized_steps, by = x -> x.t_step)
    return realized_steps
end

function print_load_change_schedule(realized_steps::Vector{RealizedLoadStep})
    println("\n" * "="^118)
    println("ACTIVE/REACTIVE LOAD-CHANGE EVENT SCHEDULE (realized EMT sequence)")
    println("="^118)
    @printf("%-8s %-12s %-25s %-25s %-25s %-25s\n", "Bus", "t [s]", "P* reference", "ΔP*", "Q* reference", "ΔQ*")
    println("-"^118)
    for s in realized_steps
        psign = s.ΔP_MW >= 0.0 ? "+" : ""
        qsign = s.ΔQ_MVAr >= 0.0 ? "+" : ""
        @printf("%-8d %-12.3f %-25s %-25s %-25s %-25s\n",
            s.bus, s.t_step,
            @sprintf("%.4f pu (%.1f MW)", s.refP_pu, s.refP_MW),
            @sprintf("%s%.4f pu (%s%.1f MW)", psign, s.deltaP_pu, psign, s.ΔP_MW),
            @sprintf("%.4f pu (%.1f MVAr)", s.refQ_pu, s.refQ_MVAr),
            @sprintf("%s%.4f pu (%s%.1f MVAr)", qsign, s.deltaQ_pu, qsign, s.ΔQ_MVAr))
    end
    println("="^118 * "\n")
end

event_times_from_realized_steps(realized_steps::Vector{RealizedLoadStep}) =
    unique(sort([s.t_step for s in realized_steps]))

function event_summary_string(realized_steps::Vector{RealizedLoadStep})
    isempty(realized_steps) && return "No load-change events"
    parts = String[]
    for s in realized_steps
        push!(parts, @sprintf("Bus %d, t=%.2fs: ΔP=%+.1f MW, ΔQ=%+.1f MVAr", s.bus, s.t_step, s.ΔP_MW, s.ΔQ_MVAr))
    end
    return join(parts, "  |  ")
end

# Dynamic line setup
function add_dynamic_lines!(sys::System)
    n_added = 0
    for line in get_components(Line, sys)
        try
            dyn_branch = DynamicBranch(line)
            add_component!(sys, dyn_branch)
            n_added += 1
        catch e
            @warn "Skipping line -> DynamicBranch conversion (constructor/add failed)" line = PSY.get_name(line) err = string(e)
        end
    end
    println("Converted AC Lines to DynamicBranches (differential line models). Added = $n_added")
    return n_added
end

function gen_at_bus(sys::System, busnum::Int)
    for g in get_components(Generator, sys)
        if PSY.get_number(PSY.get_bus(g)) == busnum
            return g
        end
    end
    error("No Generator found at bus $busnum")
end

function update_system_voltages!(sys::System, pf_data::PowerFlowData)
    for bus in get_components(Bus, sys)
        bus_num = PSY.get_number(bus)
        if haskey(pf_data.bus_lookup, bus_num)
            idx = pf_data.bus_lookup[bus_num]
            PSY.set_magnitude!(bus, pf_data.bus_magnitude[idx])
            PSY.set_angle!(bus, pf_data.bus_angles[idx])
        end
    end
end

function enforce_load_bases!(sys::System; base_mva::Float64 = SYS_BASE_MVA)
    for ld in get_components(ElectricLoad, sys)
        try
            PSY.set_base_power!(ld, base_mva)
        catch
        end
    end
end

# Damp zero-resistance dynamic lines
function enforce_min_line_r!(sys::System; rmin::Float64 = 0.002, xr_cap::Float64 = 6.0)
    n = 0
    for l in get_components(Line, sys)
        r = try PSY.get_r(l) catch; 0.0 end
        x = try PSY.get_x(l) catch; 0.0 end
        target = max(r, rmin, x / xr_cap)
        if target > r + 1e-12
            try PSY.set_r!(l, target); n += 1 catch end
        end
    end
    println("Line resistance floored on $n lines (rmin=$rmin, X/R<=$xr_cap).")
    return n
end


function loads_to_constant_impedance!(sys::System)
    converted = 0
    for ld in collect(get_components(PowerLoad, sys))
        nm = PSY.get_name(ld)
        startswith(nm, "PerturbLoad") && continue
        b = PSY.get_bus(ld)
        V = max(try PSY.get_magnitude(b) catch; 1.0 end, 0.5)
        P = try PSY.get_active_power(ld)   catch; 0.0 end
        Q = try PSY.get_reactive_power(ld) catch; 0.0 end
        (abs(P) < 1e-9 && abs(Q) < 1e-9) && continue
        Y = complex(P, -Q) / V^2
        adm = nothing
        try
            adm = FixedAdmittance(name = "Zload_" * nm, available = true, bus = b, Y = Y)
        catch
            adm = FixedAdmittance("Zload_" * nm, true, b, Y)
        end
        remove_component!(sys, ld)
        add_component!(sys, adm)
        converted += 1
    end
    println("Converted $converted constant-power loads to constant impedance (FixedAdmittance).")
    return converted
end

function sync_static_gens_to_pf!(sys::System, pf_data::PowerFlowData; buses::Vector{Int})
    @info "Syncing static generator P/Q setpoints from PF (SYSTEM_BASE)..." buses = buses
    for b in buses
        if !haskey(pf_data.bus_lookup, b)
            @warn "Bus not found in PF bus_lookup; skipping" bus = b
            continue
        end
        idx = pf_data.bus_lookup[b]
        Pinj = try pf_data.bus_activepower_injection[idx] catch; 0.0 end
        Qinj = try pf_data.bus_reactivepower_injection[idx] catch; 0.0 end
        Pwd  = try pf_data.bus_activepower_withdrawals[idx] catch; 0.0 end
        Qwd  = try pf_data.bus_reactivepower_withdrawals[idx] catch; 0.0 end
        Pnet_pu = Pinj - Pwd
        Qnet_pu = Qinj - Qwd
        g = gen_at_bus(sys, b)
        PSY.set_active_power!(g, Pnet_pu)
        try
            PSY.set_reactive_power!(g, Qnet_pu)
        catch
        end
    end
end

# Controller reference helpers
function get_generator_pref_qref(g::Generator)
    p_ref = try
        PSY.get_active_power(g)
    catch
        0.0
    end
    q_ref = try
        PSY.get_reactive_power(g)
    catch
        0.0
    end

    #Convert controller references to device base
    mbase = try PSY.get_base_power(g) catch; SYS_BASE_MVA end
    scale = SYS_BASE_MVA / mbase
    return p_ref * scale, q_ref * scale
end

function get_bus_vref(sys::System, g::Generator)
    bus = PSY.get_bus(g)
    v_ref = try
        PSY.get_magnitude(bus)
    catch
        1.0
    end
    return v_ref
end

function print_inverter_reference_summary(sys::System; gfm_buses::Vector{Int}, gfl_buses::Vector{Int})
    println("\n" * "="^100)
    println("INVERTER OUTER-CONTROL REFERENCE VALUES (from initialized PF operating point)")
    println("="^100)
    if !isempty(gfm_buses)
        println("GFM buses:")
        println(rpad("Bus", 8), rpad("P_ref [pu]", 18), rpad("Q_ref [pu]", 18), rpad("V_ref [pu]", 18))
        println("-"^100)
        for b in gfm_buses
            g = gen_at_bus(sys, b)
            p_ref0, q_ref0 = get_generator_pref_qref(g)
            v_ref0 = get_bus_vref(sys, g)
            println(rpad(string(b), 8), rpad(@sprintf("%.6f", p_ref0), 18),
                    rpad(@sprintf("%.6f", q_ref0), 18), rpad(@sprintf("%.6f", v_ref0), 18))
        end
        println("-"^100)
    end
    if !isempty(gfl_buses)
        println("GFL buses:")
        println(rpad("Bus", 8), rpad("P_ref [pu]", 18), rpad("Q_ref [pu]", 18))
        println("-"^100)
        for b in gfl_buses
            g = gen_at_bus(sys, b)
            p_ref0, q_ref0 = get_generator_pref_qref(g)
            println(rpad(string(b), 8), rpad(@sprintf("%.6f", p_ref0), 18), rpad(@sprintf("%.6f", q_ref0), 18))
        end
        println("-"^100)
    end
    println("="^100 * "\n")
end

function print_dynamic_reference_summary(sys::System; buses::Vector{Int})
    println("\n" * "="^112)
    println("DYNAMIC MODEL REFERENCE SUMMARY AFTER PF SYNC AND mBase SETTING")
    println("="^112)
    @printf("%-8s %-10s %-16s %-16s %-16s %-14s %-14s\n",
            "Bus", "Model", "P_ref[pu]", "Q_ref[pu]", "V_ref[pu]", "mBase[MVA]", "P_ref[MW]")
    println("-"^112)
    for b in buses
        g = gen_at_bus(sys, b)
        p_ref0, q_ref0 = get_generator_pref_qref(g)
        v_ref0 = get_bus_vref(sys, g)
        mbase = try PSY.get_base_power(g) catch; NaN end
        @printf("%-8d %-10s %-16.6f %-16.6f %-16.6f %-14.2f %-14.3f\n",
                b, model_label(b), p_ref0, q_ref0, v_ref0, mbase, p_ref0 * mbase)
    end
    println("="^112 * "\n")
end

#Power-flow export
function _pf_angle_to_deg(ang::Real)
    if abs(ang) <= 2π + 0.5
        return ang * (180.0 / π)
    else
        return ang
    end
end

function build_pf_bus_dataframe(pf_data::PowerFlowData; sys_base_mva::Float64 = SYS_BASE_MVA)
    busnums = sort(collect(keys(pf_data.bus_lookup)))
    rows = Vector{NamedTuple}(undef, length(busnums))
    for (k, b) in enumerate(busnums)
        idx = pf_data.bus_lookup[b]
        Vm = try pf_data.bus_magnitude[idx] catch; NaN end
        Va = try pf_data.bus_angles[idx]    catch; NaN end
        Pinj = try pf_data.bus_activepower_injection[idx]   catch; 0.0 end
        Qinj = try pf_data.bus_reactivepower_injection[idx] catch; 0.0 end
        Pwd  = try pf_data.bus_activepower_withdrawals[idx] catch; 0.0 end
        Qwd  = try pf_data.bus_reactivepower_withdrawals[idx] catch; 0.0 end
        Pnet = Pinj - Pwd
        Qnet = Qinj - Qwd
        rows[k] = (
            bus = b, Vm_pu = Vm, Va_deg = _pf_angle_to_deg(Va),
            Pinj_pu = Pinj, Qinj_pu = Qinj, Pwd_pu = Pwd, Qwd_pu = Qwd,
            Pnet_pu = Pnet, Qnet_pu = Qnet,
            Pinj_MW = Pinj * sys_base_mva, Qinj_MVAr = Qinj * sys_base_mva,
            Pwd_MW = Pwd * sys_base_mva, Qwd_MVAr = Qwd * sys_base_mva,
            Pnet_MW = Pnet * sys_base_mva, Qnet_MVAr = Qnet * sys_base_mva
        )
    end
    return DataFrame(rows)
end

function write_pf_results(pf_data::PowerFlowData, outdir::String;
                          basename::String = "powerflow_results", sys_base_mva::Float64 = SYS_BASE_MVA)
    mkpath(outdir)
    df = build_pf_bus_dataframe(pf_data; sys_base_mva = sys_base_mva)
    csv_path = joinpath(outdir, basename * ".csv")
    txt_path = joinpath(outdir, basename * ".txt")
    CSV.write(csv_path, df)
    open(txt_path, "w") do io
        println(io, "POWER FLOW RESULTS (SYSTEM_BASE)")
        println(io, @sprintf("System base MVA: %.3f", sys_base_mva))
        println(io, "Columns include bus voltage magnitude/angle and bus injections/withdrawals (P,Q) in pu and MW/MVAr.")
        println(io, "-"^120)
        @printf(io, "%5s  %9s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                "Bus", "Vm[pu]", "Va[deg]", "Pinj[MW]", "Qinj[MVAr]", "Pwd[MW]", "Qwd[MVAr]", "Pnet[MW]", "Qnet[MVAr]")
        println(io, "-"^120)
        for r in eachrow(df)
            @printf(io, "%5d  %9.5f  %10.5f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f  %10.3f\n",
                    r.bus, r.Vm_pu, r.Va_deg, r.Pinj_MW, r.Qinj_MVAr, r.Pwd_MW, r.Qwd_MVAr, r.Pnet_MW, r.Qnet_MVAr)
        end
        println(io, "-"^120)
        println(io, "Note: Pinj/Qinj and Pwd/Qwd are bus-level totals from PowerFlowData; Pnet/Qnet = injections - withdrawals.")
    end
    @info "Saved PF results (CSV + TXT)" csv = csv_path txt = txt_path nbus = nrow(df)
    return (csv_path = csv_path, txt_path = txt_path)
end

function bus_component(sys::System, busnum::Int)
    for b in get_components(Bus, sys)
        if PSY.get_number(b) == busnum
            return b
        end
    end
    error("Bus $busnum not found in system")
end

function ensure_perturbable_load_on_bus!(sys::System, busnum::Int; base_mva::Float64 = SYS_BASE_MVA)
    name = @sprintf("PerturbLoad_bus%02d", busnum)
    for ld in get_components(ElectricLoad, sys)
        if PSY.get_name(ld) == name
            try PSY.set_base_power!(ld, base_mva) catch end
            return ld
        end
    end
    b = bus_component(sys, busnum)
    ld = nothing
    try
        ld = PowerLoad(name = name, available = true, bus = b, active_power = 0.0,
                       reactive_power = 0.0, base_power = base_mva,
                       max_active_power = 10.0, max_reactive_power = 10.0)
    catch
        try
            ld = PowerLoad(name, true, b, 0.0, 0.0, base_mva, 10.0, 10.0)
        catch
            ld = PowerLoad(name, true, b, 0.0, 0.0, base_mva)
        end
    end
    add_component!(sys, ld)
    try PSY.set_base_power!(ld, base_mva) catch end
    return ld
end

function ensure_perturbable_loads_all_buses!(sys::System; base_mva::Float64 = SYS_BASE_MVA)
    for busnum in 1:39
        ensure_perturbable_load_on_bus!(sys, busnum; base_mva = base_mva)
    end
end

function perturb_load_at_bus(sys::System, busnum::Int)
    target = @sprintf("PerturbLoad_bus%02d", busnum)
    for ld in get_components(ElectricLoad, sys)
        if PSY.get_name(ld) == target
            if PSY.get_number(PSY.get_bus(ld)) != busnum
                error("Found $target but attached to bus $(PSY.get_number(PSY.get_bus(ld))) not $busnum")
            end
            return ld
        end
    end
    error("Perturb load $target not found")
end

function print_loads_on_bus(sys::System, busnum::Int)
    println("\nLoads on bus $busnum:")
    found = false
    for ld in get_components(ElectricLoad, sys)
        if PSY.get_number(PSY.get_bus(ld)) == busnum
            found = true
            basep = try PSY.get_base_power(ld) catch; NaN end
            ppu   = try PSY.get_active_power(ld) catch; NaN end
            qpu   = try PSY.get_reactive_power(ld) catch; NaN end
            println("  - $(PSY.get_name(ld)) :: $(typeof(ld)) | base=$(basep) | P=$(ppu) pu | Q=$(qpu) pu")
        end
    end
    !found && println("  (none)")
end

function make_load_change_constantP(sys::System; bus::Int, t_step::Float64, ΔP_MW::Float64)
    ld = perturb_load_at_bus(sys, bus)
    ΔP_pu = ΔP_MW / SYS_BASE_MVA
    @info "Applying CONSTANT-P load step (PerturbLoad)" bus = bus load = PSY.get_name(ld) t_step = t_step ΔP_MW = ΔP_MW ΔP_pu = ΔP_pu typeof = string(typeof(ld))
    for field in (:active_power, :P, :p, :P_ref_power, :P_ref)
        try
            return LoadChange(t_step, ld, field, ΔP_pu)
        catch
        end
    end
    error("Could not apply active-power load step: no recognized active-power field found on $(typeof(ld)).")
end

function make_load_change_constantQ(sys::System; bus::Int, t_step::Float64, ΔQ_MVAr::Float64)
    ld = perturb_load_at_bus(sys, bus)
    ΔQ_pu = ΔQ_MVAr / SYS_BASE_MVA
    @info "Applying CONSTANT-Q load step (PerturbLoad)" bus = bus load = PSY.get_name(ld) t_step = t_step ΔQ_MVAr = ΔQ_MVAr ΔQ_pu = ΔQ_pu typeof = string(typeof(ld))
    for field in (:reactive_power, :Q, :q, :Q_ref_power, :Q_ref)
        try
            return LoadChange(t_step, ld, field, ΔQ_pu)
        catch
        end
    end
    error("Could not apply reactive-power load step: no recognized reactive-power field found on $(typeof(ld)).")
end

function make_load_changes_from_event_schedule(sys::System, realized_steps::Vector{RealizedLoadStep})
    perts = PowerSimulationsDynamics.Perturbation[]
    for s in realized_steps
        if abs(s.ΔP_MW) > 1e-12
            push!(perts, make_load_change_constantP(sys; bus = s.bus, t_step = s.t_step, ΔP_MW = s.ΔP_MW))
        end
        if abs(s.ΔQ_MVAr) > 1e-12
            push!(perts, make_load_change_constantQ(sys; bus = s.bus, t_step = s.t_step, ΔQ_MVAr = s.ΔQ_MVAr))
        end
    end
    return perts
end

# Network-switch fault helpers
function print_bus_fault_line_trip_schedule()
    println("\n" * "="^118)
    println("LINE-FAULT + LINE-TRIP DISTURBANCE SCHEDULE")
    println("="^118)
    @printf("Fault-on   : t = %.3f s, 3φ-G fault on line %d--%d at %.1f%% from bus %d, Zf = %.6g %+.6gim pu\n",
            FAULT_APPLY_TIME_S, FAULT_LINE_FROM, FAULT_LINE_TO,
            100.0 * FAULT_LOCATION_FROM, FAULT_LINE_FROM, FAULT_R_PU, FAULT_X_PU)
    @printf("Fault-clear: t = %.3f s, remove fault and trip/open line %d--%d\n",
            FAULT_CLEAR_TIME_S, FAULT_LINE_FROM, FAULT_LINE_TO)
    @printf("Fault duration = %.3f s = %.1f ms\n",
            FAULT_CLEAR_TIME_S - FAULT_APPLY_TIME_S, 1000.0 * (FAULT_CLEAR_TIME_S - FAULT_APPLY_TIME_S))
    println("="^118 * "\n")
end

function selected_fault_clear_time()
    if FAULT_SCENARIO == :line_fault_line_trip
        return FAULT_CLEAR_TIME_S
    elseif FAULT_SCENARIO == :bus_fault_line_trip
        return BUS_FAULT_CLEAR_TIME_S
    else
        error("Unsupported FAULT_SCENARIO=$(FAULT_SCENARIO). Use :line_fault_line_trip or :bus_fault_line_trip.")
    end
end

function fault_event_times()
    BUS_FAULT_ENABLE || return Float64[]
    return unique(sort([FAULT_APPLY_TIME_S, selected_fault_clear_time()]))
end

function fault_event_summary_string()
    BUS_FAULT_ENABLE || return "No fault disturbance"
    if FAULT_SCENARIO == :line_fault_line_trip
        return @sprintf("3φ-G fault on line %d--%d at %.1f%% from bus %d, cleared at %.3fs by tripping line %d--%d",
                        FAULT_LINE_FROM, FAULT_LINE_TO, 100.0 * FAULT_LOCATION_FROM, FAULT_LINE_FROM,
                        FAULT_CLEAR_TIME_S, FAULT_LINE_FROM, FAULT_LINE_TO)
    elseif FAULT_SCENARIO == :bus_fault_line_trip
        return @sprintf("3φ-G fault at bus %d, cleared after %.3fs at t=%.3fs by tripping line %d--%d",
                        BUS_FAULT_BUS, BUS_FAULT_DURATION_S, BUS_FAULT_CLEAR_TIME_S,
                        BUS_FAULT_TRIP_LINE_FROM, BUS_FAULT_TRIP_LINE_TO)
    else
        error("Unsupported FAULT_SCENARIO=$(FAULT_SCENARIO). Use :line_fault_line_trip or :bus_fault_line_trip.")
    end
end

function print_bus_3ph_fault_line_trip_schedule()
    println("\n" * "="^118)
    println("BUS-FAULT + LINE-TRIP DISTURBANCE SCHEDULE")
    println("="^118)
    @printf("Fault-on   : t = %.3f s, 3φ-G fault at bus %d, Zf = %.6g %+.6gim pu\n",
            FAULT_APPLY_TIME_S, BUS_FAULT_BUS, FAULT_R_PU, FAULT_X_PU)
    @printf("Fault-clear: t = %.3f s, remove bus fault and trip/open line %d--%d\n",
            BUS_FAULT_CLEAR_TIME_S, BUS_FAULT_TRIP_LINE_FROM, BUS_FAULT_TRIP_LINE_TO)
    @printf("Fault duration = %.3f s = %.1f ms\n",
            BUS_FAULT_CLEAR_TIME_S - FAULT_APPLY_TIME_S,
            1000.0 * (BUS_FAULT_CLEAR_TIME_S - FAULT_APPLY_TIME_S))
    println("="^118 * "\n")
end

function print_fault_disturbance_schedule()
    if FAULT_SCENARIO == :line_fault_line_trip
        print_bus_fault_line_trip_schedule()
    elseif FAULT_SCENARIO == :bus_fault_line_trip
        print_bus_3ph_fault_line_trip_schedule()
    else
        error("Unsupported FAULT_SCENARIO=$(FAULT_SCENARIO). Use :line_fault_line_trip or :bus_fault_line_trip.")
    end
end

bus_index_map_for_network_switch(; buses::Vector{Int} = collect(1:39)) =
    Dict{Int,Int}(b => i for (i, b) in enumerate(buses))

function remove_line_from_ybus!(Y::AbstractMatrix{ComplexF64}, fbus::Int, tbus::Int;
                                buses::Vector{Int} = collect(1:39), r::Float64, x::Float64, b::Float64 = 0.0)
    bus_to_idx = bus_index_map_for_network_switch(; buses = buses)
    if !(haskey(bus_to_idx, fbus) && haskey(bus_to_idx, tbus))
        error("Cannot remove line $(fbus)--$(tbus): one or both buses are missing from Ybus bus ordering.")
    end
    z = complex(r, x)
    if abs(z) < 1e-12
        error("Cannot remove line $(fbus)--$(tbus): near-zero impedance r+jx = $(z).")
    end
    y = inv(z)
    ysh = 1im * b / 2.0
    i = bus_to_idx[fbus]
    j = bus_to_idx[tbus]
    Y[i, i] -= y + ysh
    Y[j, j] -= y + ysh
    Y[i, j] += y
    Y[j, i] += y
    return Y
end

const IEEE39_BRANCH_PARAM_FALLBACK = Dict{Tuple{Int,Int},Tuple{Float64,Float64,Float64}}(
    (1, 2) => (0.004, 0.041, 0.698), (1, 39) => (0.001, 0.025, 0.750),
    (2, 3) => (0.001, 0.015, 0.258), (2, 25) => (0.007, 0.009, 0.146),
    (3, 4) => (0.001, 0.021, 0.222), (3, 18) => (0.001, 0.013, 0.214),
    (4, 5) => (0.001, 0.013, 0.134), (4, 14) => (0.001, 0.013, 0.138),
    (5, 6) => (0.000, 0.003, 0.044), (5, 8) => (0.001, 0.011, 0.148),
    (6, 7) => (0.001, 0.009, 0.114), (6, 11) => (0.001, 0.008, 0.138),
    (7, 8) => (0.000, 0.005, 0.078), (8, 9) => (0.002, 0.036, 0.380),
    (9, 39) => (0.001, 0.025, 1.200), (10, 11) => (0.000, 0.004, 0.072),
    (10, 13) => (0.000, 0.004, 0.072), (13, 14) => (0.001, 0.010, 0.172),
    (14, 15) => (0.002, 0.022, 0.366), (15, 16) => (0.001, 0.009, 0.172),
    (16, 17) => (0.001, 0.009, 0.134), (16, 19) => (0.002, 0.020, 0.304),
    (16, 21) => (0.001, 0.014, 0.254), (16, 24) => (0.000, 0.006, 0.068),
    (17, 18) => (0.001, 0.008, 0.132), (17, 27) => (0.001, 0.017, 0.322),
    (21, 22) => (0.001, 0.014, 0.256), (22, 23) => (0.001, 0.010, 0.184),
    (23, 24) => (0.002, 0.035, 0.362), (25, 26) => (0.003, 0.032, 0.512),
    (26, 27) => (0.001, 0.015, 0.240), (26, 28) => (0.004, 0.047, 0.780),
    (26, 29) => (0.006, 0.063, 1.028), (28, 29) => (0.001, 0.015, 0.250),
    (2, 30) => (0.000, 0.018, 0.000), (6, 31) => (0.000, 0.025, 0.000),
    (10, 32) => (0.000, 0.020, 0.000), (12, 11) => (0.002, 0.043, 0.000),
    (12, 13) => (0.002, 0.043, 0.000), (19, 20) => (0.001, 0.014, 0.000),
    (19, 33) => (0.001, 0.014, 0.000), (20, 34) => (0.001, 0.018, 0.000),
    (22, 35) => (0.000, 0.014, 0.000), (23, 36) => (0.000, 0.027, 0.000),
    (25, 37) => (0.001, 0.023, 0.000), (29, 38) => (0.001, 0.016, 0.000),
)

const IEEE39_BRANCH_YBUS_FALLBACK = Tuple{Int,Int,Float64,Float64,Float64,Float64}[
    (1, 2, 0.004, 0.041, 0.698, 0.0), (1, 39, 0.001, 0.025, 0.750, 0.0),
    (2, 3, 0.001, 0.015, 0.258, 0.0), (2, 25, 0.007, 0.009, 0.146, 0.0),
    (3, 4, 0.001, 0.021, 0.222, 0.0), (3, 18, 0.001, 0.013, 0.214, 0.0),
    (4, 5, 0.001, 0.013, 0.134, 0.0), (4, 14, 0.001, 0.013, 0.138, 0.0),
    (5, 6, 0.000, 0.003, 0.044, 0.0), (5, 8, 0.001, 0.011, 0.148, 0.0),
    (6, 7, 0.001, 0.009, 0.114, 0.0), (6, 11, 0.001, 0.008, 0.138, 0.0),
    (7, 8, 0.000, 0.005, 0.078, 0.0), (8, 9, 0.002, 0.036, 0.380, 0.0),
    (9, 39, 0.001, 0.025, 1.200, 0.0), (10, 11, 0.000, 0.004, 0.072, 0.0),
    (10, 13, 0.000, 0.004, 0.072, 0.0), (13, 14, 0.001, 0.010, 0.172, 0.0),
    (14, 15, 0.002, 0.022, 0.366, 0.0), (15, 16, 0.001, 0.009, 0.172, 0.0),
    (16, 17, 0.001, 0.009, 0.134, 0.0), (16, 19, 0.002, 0.020, 0.304, 0.0),
    (16, 21, 0.001, 0.014, 0.254, 0.0), (16, 24, 0.000, 0.006, 0.068, 0.0),
    (17, 18, 0.001, 0.008, 0.132, 0.0), (17, 27, 0.001, 0.017, 0.322, 0.0),
    (21, 22, 0.001, 0.014, 0.256, 0.0), (22, 23, 0.001, 0.010, 0.184, 0.0),
    (23, 24, 0.002, 0.035, 0.362, 0.0), (25, 26, 0.003, 0.032, 0.512, 0.0),
    (26, 27, 0.001, 0.015, 0.240, 0.0), (26, 28, 0.004, 0.047, 0.780, 0.0),
    (26, 29, 0.006, 0.063, 1.028, 0.0), (28, 29, 0.001, 0.015, 0.250, 0.0),
    (2, 30, 0.000, 0.018, 0.000, 1.025), (6, 31, 0.000, 0.025, 0.000, 1.070),
    (10, 32, 0.000, 0.020, 0.000, 1.070), (12, 11, 0.002, 0.043, 0.000, 1.006),
    (12, 13, 0.002, 0.043, 0.000, 1.006), (19, 20, 0.001, 0.014, 0.000, 1.060),
    (19, 33, 0.001, 0.014, 0.000, 1.070), (20, 34, 0.001, 0.018, 0.000, 1.009),
    (22, 35, 0.000, 0.014, 0.000, 1.025), (23, 36, 0.000, 0.027, 0.000, 1.000),
    (25, 37, 0.001, 0.023, 0.000, 1.025), (29, 38, 0.001, 0.016, 0.000, 1.025),
]

function stamp_branch_ybus!(Y::AbstractMatrix{ComplexF64}, fbus::Int, tbus::Int,
                            r::Float64, x::Float64, b::Float64, tap_raw::Float64; buses::Vector{Int})
    bus_to_idx = Dict{Int,Int}(b => i for (i, b) in enumerate(buses))
    if !(haskey(bus_to_idx, fbus) && haskey(bus_to_idx, tbus))
        return Y
    end
    z = complex(r, x)
    if abs(z) < 1e-12
        @warn "Skipping near-zero impedance branch while building Ybus" fbus=fbus tbus=tbus r=r x=x
        return Y
    end
    tap = abs(tap_raw) < 1e-12 ? 1.0 : tap_raw
    y = inv(z)
    ysh = 1im * b / 2.0
    i = bus_to_idx[fbus]
    j = bus_to_idx[tbus]
    Y[i, i] += (y + ysh) / (tap * tap)
    Y[i, j] -= y / tap
    Y[j, i] -= y / tap
    Y[j, j] += y + ysh
    return Y
end

function build_ybus_dense_from_ieee39_fallback(; buses::Vector{Int} = NETWORK_PQ_ALL_BUSES)
    Y = zeros(ComplexF64, length(buses), length(buses))
    for (f, t, r, x, b, tap) in IEEE39_BRANCH_YBUS_FALLBACK
        stamp_branch_ybus!(Y, f, t, r, x, b, tap; buses = buses)
    end
    return Y
end

_branch_key(fbus::Int, tbus::Int) = fbus < tbus ? (fbus, tbus) : (tbus, fbus)

function find_line_parameters(sys::System, fbus::Int, tbus::Int)
    target = _branch_key(fbus, tbus)
    for line in get_components(Line, sys)
        try
            f, t = line_pair(line)
            if _branch_key(f, t) == target
                r = _line_parameter_or_default(line, :get_r, NaN)
                x = _line_parameter_or_default(line, :get_x, NaN)
                b = _line_parameter_or_default(line, :get_b, 0.0)
                if !isfinite(r) || !isfinite(x)
                    error("Found line $(fbus)--$(tbus), but could not read finite r/x values.")
                end
                return (r = r, x = x, b = b, name = PSY.get_name(line))
            end
        catch
        end
    end
    if haskey(IEEE39_BRANCH_PARAM_FALLBACK, target)
        r, x, b = IEEE39_BRANCH_PARAM_FALLBACK[target]
        @warn "Line component not found in sys; using IEEE-39 MATPOWER fallback branch parameters" fbus=fbus tbus=tbus r=r x=x b=b
        return (r = r, x = x, b = b, name = "MATPOWER_branch_$(fbus)_$(tbus)")
    end
    error("Could not find branch parameters for branch $(fbus)--$(tbus).")
end

function stamp_series_branch_by_indices!(Y::AbstractMatrix{ComplexF64}, i::Int, j::Int;
                                         r::Float64, x::Float64, b::Float64 = 0.0)
    z = complex(r, x)
    if abs(z) < 1e-12
        error("Cannot stamp branch: near-zero impedance r+jx = $(z).")
    end
    y = inv(z)
    ysh = 1im * b / 2.0
    Y[i, i] += y + ysh
    Y[j, j] += y + ysh
    Y[i, j] -= y
    Y[j, i] -= y
    return Y
end

function build_line_fault_ybus_kron_reduced(Ybus_prefault::AbstractMatrix{ComplexF64},
                                            line_from::Int, line_to::Int;
                                            fault_location_from::Float64 = FAULT_LOCATION_FROM,
                                            fault_r::Float64 = FAULT_R_PU, fault_x::Float64 = FAULT_X_PU,
                                            buses::Vector{Int} = collect(1:39),
                                            r::Float64, x::Float64, b::Float64 = 0.0)
    if !(0.0 < fault_location_from < 1.0)
        error("FAULT_LOCATION_FROM must be strictly between 0 and 1. Use 0.5 for a midpoint line fault.")
    end
    zf = complex(fault_r, fault_x)
    if abs(zf) < 1e-12
        error("Fault impedance is too close to zero. Use a small nonzero value.")
    end
    yf = inv(zf)
    bus_to_idx = bus_index_map_for_network_switch(; buses = buses)
    if !(haskey(bus_to_idx, line_from) && haskey(bus_to_idx, line_to))
        error("Line-fault buses $(line_from)--$(line_to) are not present in the NetworkSwitch bus ordering.")
    end
    Ybase = copy(Ybus_prefault)
    remove_line_from_ybus!(Ybase, line_from, line_to; buses = buses, r = r, x = x, b = b)
    n = size(Ybase, 1)
    m = n + 1
    Yaug = zeros(ComplexF64, n + 1, n + 1)
    Yaug[1:n, 1:n] .= Ybase
    i = bus_to_idx[line_from]
    j = bus_to_idx[line_to]
    α = fault_location_from
    r1 = α * r; x1 = α * x; b1 = α * b
    r2 = (1.0 - α) * r; x2 = (1.0 - α) * x; b2 = (1.0 - α) * b
    stamp_series_branch_by_indices!(Yaug, i, m; r = r1, x = x1, b = b1)
    stamp_series_branch_by_indices!(Yaug, m, j; r = r2, x = x2, b = b2)
    Yaug[m, m] += yf
    if abs(Yaug[m, m]) < 1e-12
        error("Cannot Kron-reduce line fault point because Ymm is near zero.")
    end
    Yred = Yaug[1:n, 1:n] .- (Yaug[1:n, m] * transpose(Yaug[m, 1:n])) ./ Yaug[m, m]
    return ComplexF64.(Yred)
end

# NetworkSwitch Ybus must include the constant-impedance loads/shunts.
function build_switch_ybus_with_loads(sys::System; buses::Vector{Int} = collect(1:39))
    Y = ComplexF64.(build_ybus_dense_from_system(sys; buses = buses))
    bus_to_idx = Dict{Int,Int}(b => i for (i, b) in enumerate(buses))
    n = 0
    for fa in get_components(FixedAdmittance, sys)
        bnum = PSY.get_number(PSY.get_bus(fa))
        haskey(bus_to_idx, bnum) || continue
        Yval = try ComplexF64(PSY.get_Y(fa)) catch; continue end
        Y[bus_to_idx[bnum], bus_to_idx[bnum]] += Yval
        n += 1
    end
    @info "NetworkSwitch Ybus includes constant-impedance loads/shunts" fixed_admittances_stamped = n
    return Y
end

function make_bus_fault_cleared_by_line_trip_perturbations(sys::System;
                                                           line_from::Int = FAULT_LINE_FROM,
                                                           line_to::Int = FAULT_LINE_TO,
                                                           fault_location_from::Float64 = FAULT_LOCATION_FROM,
                                                           t_fault::Float64 = FAULT_APPLY_TIME_S,
                                                           t_clear::Float64 = FAULT_CLEAR_TIME_S,
                                                           fault_r::Float64 = FAULT_R_PU,
                                                           fault_x::Float64 = FAULT_X_PU,
                                                           buses::Vector{Int} = collect(1:39))
    if t_clear <= t_fault
        error("FAULT_CLEAR_TIME_S must be greater than FAULT_APPLY_TIME_S.")
    end
    Ybus_prefault = build_switch_ybus_with_loads(sys; buses = buses)
    lp = find_line_parameters(sys, line_from, line_to)
    Ybus_fault = build_line_fault_ybus_kron_reduced(Ybus_prefault, line_from, line_to;
        fault_location_from = fault_location_from, fault_r = fault_r, fault_x = fault_x,
        buses = buses, r = lp.r, x = lp.x, b = lp.b)
    Ybus_postfault = copy(Ybus_prefault)
    remove_line_from_ybus!(Ybus_postfault, line_from, line_to; buses = buses, r = lp.r, x = lp.x, b = lp.b)
    zf = complex(fault_r, fault_x)
    yf = inv(zf)
    @info "Prepared line fault cleared by line trip NetworkSwitch perturbations" line_fault = "$(line_from)--$(line_to)" fault_location_from = fault_location_from t_fault = t_fault t_clear = t_clear fault_impedance_pu = zf fault_admittance_pu = yf tripped_line_name = lp.name line_r = lp.r line_x = lp.x line_b = lp.b
    perts = PowerSimulationsDynamics.Perturbation[]
    push!(perts, NetworkSwitch(t_fault, sparse(Ybus_fault)))
    push!(perts, NetworkSwitch(t_clear, sparse(Ybus_postfault)))
    return perts
end

function build_bus_fault_ybus(Ybus_prefault::AbstractMatrix{ComplexF64},
                              fault_bus::Int;
                              fault_r::Float64 = FAULT_R_PU,
                              fault_x::Float64 = FAULT_X_PU,
                              buses::Vector{Int} = collect(1:39))
    zf = complex(fault_r, fault_x)
    if abs(zf) < 1e-12
        error("Fault impedance is too close to zero. Use a small nonzero value.")
    end
    bus_to_idx = bus_index_map_for_network_switch(; buses = buses)
    if !haskey(bus_to_idx, fault_bus)
        error("Bus fault target bus $(fault_bus) is not present in the NetworkSwitch bus ordering.")
    end
    Yfault = copy(Ybus_prefault)
    idx = bus_to_idx[fault_bus]
    Yfault[idx, idx] += inv(zf)
    return ComplexF64.(Yfault)
end

function make_bus_3ph_fault_cleared_by_line_trip_perturbations(sys::System;
                                                               fault_bus::Int = BUS_FAULT_BUS,
                                                               trip_line_from::Int = BUS_FAULT_TRIP_LINE_FROM,
                                                               trip_line_to::Int = BUS_FAULT_TRIP_LINE_TO,
                                                               t_fault::Float64 = FAULT_APPLY_TIME_S,
                                                               t_clear::Float64 = BUS_FAULT_CLEAR_TIME_S,
                                                               fault_r::Float64 = FAULT_R_PU,
                                                               fault_x::Float64 = FAULT_X_PU,
                                                               buses::Vector{Int} = collect(1:39))
    if t_clear <= t_fault
        error("BUS_FAULT_CLEAR_TIME_S must be greater than FAULT_APPLY_TIME_S.")
    end
    Ybus_prefault = build_switch_ybus_with_loads(sys; buses = buses)
    lp = find_line_parameters(sys, trip_line_from, trip_line_to)

    Ybus_fault = build_bus_fault_ybus(Ybus_prefault, fault_bus;
        fault_r = fault_r, fault_x = fault_x, buses = buses)

    Ybus_postfault = copy(Ybus_prefault)
    remove_line_from_ybus!(Ybus_postfault, trip_line_from, trip_line_to;
        buses = buses, r = lp.r, x = lp.x, b = lp.b)

    zf = complex(fault_r, fault_x)
    yf = inv(zf)
    @info "Prepared bus 3φ-G fault cleared by line trip NetworkSwitch perturbations" fault_bus = fault_bus tripped_line = "$(trip_line_from)--$(trip_line_to)" t_fault = t_fault t_clear = t_clear fault_duration_s = (t_clear - t_fault) fault_impedance_pu = zf fault_admittance_pu = yf tripped_line_name = lp.name line_r = lp.r line_x = lp.x line_b = lp.b

    perts = PowerSimulationsDynamics.Perturbation[]
    push!(perts, NetworkSwitch(t_fault, sparse(Ybus_fault)))
    push!(perts, NetworkSwitch(t_clear, sparse(Ybus_postfault)))
    return perts
end

function make_fault_perturbations(sys::System)
    if FAULT_SCENARIO == :line_fault_line_trip
        return make_bus_fault_cleared_by_line_trip_perturbations(sys)
    elseif FAULT_SCENARIO == :bus_fault_line_trip
        return make_bus_3ph_fault_cleared_by_line_trip_perturbations(sys)
    else
        error("Unsupported FAULT_SCENARIO=$(FAULT_SCENARIO). Use :line_fault_line_trip or :bus_fault_line_trip.")
    end
end

# Frequency extraction
function get_first_state_series(results, dev_name::String, syms::Vector{Symbol})
    for s in syms
        try
            t, x = get_state_series(results, (dev_name, s))
            return (t, x, s)
        catch
        end
    end
    return nothing
end

function normalize_speed_to_pu(w::AbstractVector)
    med = median(abs.(w))
    return (med > 10.0) ? (w ./ Ω_BASE_RAD_S) : w
end

function get_gfl_pll_speed_pu(results, dev_name::String; kp_pll::Float64, ki_pll::Float64)
    direct_syms = Symbol[:ω_pll, :w_pll, :omega_pll, :pll_ω, :pll_w, :ωpll, :wpll, :ω_lp, :w_lp]
    direct = get_first_state_series(results, dev_name, direct_syms)
    if direct !== nothing
        t, w_raw, src = direct
        return (t, normalize_speed_to_pu(w_raw), src)
    end
    vq_syms  = Symbol[:vq_pll, :v_pll_q, :v_q_pll, :Vq_pll, :pll_vq, :vqpll]
    eps_syms = Symbol[:ε_pll, :eps_pll, :ϵ_pll, :epsilon_pll, :pll_eps, :pll_ε]
    vq  = get_first_state_series(results, dev_name, vq_syms)
    eps = get_first_state_series(results, dev_name, eps_syms)
    if (vq !== nothing) && (eps !== nothing)
        t, vqv, vq_sym = vq
        _, epsv, eps_sym = eps
        ω = 1.0 .+ (kp_pll .* vqv .+ ki_pll .* epsv)
        return (t, ω, Symbol("ω_PLL_reconstructed_from_" * string(vq_sym) * "_" * string(eps_sym)))
    end
    return nothing
end

# Use actual PLL gains when available
function gfl_pll_gains_at_bus(sys::System, bus::Int)
    kp = getGflPllKp(bus)
    ki = getGflPllKi(bus)
    dyn = try dyn_inverter_at_bus(sys, bus) catch; nothing end
    dyn === nothing && return kp, ki
    pll = nothing
    for getter in (:get_freq_estimator, :get_frequency_estimator)
        try
            if isdefined(PSY, getter)
                pll = getfield(PSY, getter)(dyn)
                pll === nothing || break
            end
        catch
        end
    end
    pll === nothing && return kp, ki
    kp = try PSY.get_kp_pll(pll) catch; kp end
    ki = try PSY.get_ki_pll(pll) catch; ki end
    return kp, ki
end

function frequency_deviation_hz_at_bus(results, sys::System, bus::Int;
                                       pll_kp::Float64 = getGflPllKp(GFL_BUSES[1]),
                                       pll_ki::Float64 = getGflPllKi(GFL_BUSES[1]))
    dev  = gen_at_bus(sys, bus)
    name = PSY.get_name(dev)
    try
        t, w = get_state_series(results, (name, :ω))
        w_pu = normalize_speed_to_pu(w)
        return (t, (w_pu .- 1.0) .* F_BASE, "Bus $bus (SG)", :ω)
    catch
    end
    s_gfm = get_first_state_series(results, name, Symbol[:ω_oc, :w_oc, :ω, :w])
    if s_gfm !== nothing
        t, w, sym = s_gfm
        w_pu = normalize_speed_to_pu(w)
        return (t, (w_pu .- 1.0) .* F_BASE, "Bus $bus (GFM)", sym)
    end

    # Reconstruct GFL PLL frequency
    kp_use, ki_use = is_gfl_bus(bus) ? gfl_pll_gains_at_bus(sys, bus) : (pll_kp, pll_ki)
    pll = get_gfl_pll_speed_pu(results, name; kp_pll = kp_use, ki_pll = ki_use)
    if pll !== nothing
        t, w_pu, sym = pll
        return (t, (w_pu .- 1.0) .* F_BASE, "Bus $bus (GFL PLL)", sym)
    end
    return nothing
end

# Rotor-angle export
function unwrap_angle_rad(a::AbstractVector)
    n = length(a)
    n == 0 && return a
    out = copy(a)
    twoπ = 2π
    for k in 2:n
        Δ = out[k] - out[k - 1]
        if Δ > π
            out[k:end] .-= twoπ
        elseif Δ < -π
            out[k:end] .+= twoπ
        end
    end
    return out
end

function first_finite_index(v::AbstractVector)
    for i in eachindex(v)
        isfinite(v[i]) && return i
    end
    return nothing
end

function zero_start(ang::AbstractVector)
    i0 = first_finite_index(ang)
    i0 === nothing && return ang
    return ang .- ang[i0]
end

function uniform_grid_with_event(Tfinal::Float64, dt::Float64, t_event::Float64)
    tg = collect(0.0:dt:Tfinal)
    if !(any(abs.(tg .- t_event) .< 1e-12))
        push!(tg, t_event)
        sort!(tg)
    end
    return tg
end

function interp1_linear(t::AbstractVector, y::AbstractVector, tq::AbstractVector)
    n = length(t)
    n == 0 && return fill(NaN, length(tq))
    out = fill(NaN, length(tq))
    ord = sortperm(t)
    tt = t[ord]
    yy = y[ord]
    tmin = tt[1]
    tmax = tt[end]
    for (k, x) in pairs(tq)
        if x < tmin || x > tmax
            out[k] = NaN
            continue
        end
        i = searchsortedlast(tt, x)
        if i >= length(tt)
            out[k] = yy[end]
        elseif tt[i] == x
            out[k] = yy[i]
        else
            t0 = tt[i]; t1 = tt[i + 1]; y0 = yy[i]; y1 = yy[i + 1]
            α = (x - t0) / (t1 - t0)
            out[k] = (1 - α) * y0 + α * y1
        end
    end
    return out
end

function rotor_angle_dev_rad_at_bus(results, sys::System, bus::Int;
                                   t_step::Float64 = 1.0, for_csv::Bool = false, for_plots::Bool = false)
    dev  = gen_at_bus(sys, bus)
    name = PSY.get_name(dev)
    series = nothing
    try
        t, δ = get_state_series(results, (name, :δ))
        series = (t, δ, "Bus $bus (rotor angle δ)", :δ)
    catch
        syms = Symbol[:δ, :delta, :θ, :theta, :θ_oc, :theta_oc, :θ_olc, :theta_olc, :θ_pll, :theta_pll]
        s = get_first_state_series(results, name, syms)
        if s !== nothing
            t, th, sym = s
            series = (t, th, "Bus $bus (angle state)", sym)
        end
    end
    series === nothing && return nothing
    t, ang_raw, lab, sym = series
    ang = unwrap_angle_rad(ang_raw)
    if for_csv && RESAMPLE_FOR_CSV
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        ang = interp1_linear(t, ang, tg)
        t   = tg
    elseif for_plots && RESAMPLE_FOR_PLOTS
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        ang = interp1_linear(t, ang, tg)
        t   = tg
    end
    ang = zero_start(ang)
    return (t, ang, lab * " (Δδ from initial)", sym)
end

function save_rotor_angle_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                      fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    series = rotor_angle_dev_rad_at_bus(results, sys, bus; t_step = t_step, for_csv = true)
    if series === nothing
        @warn "Skipping rotor-angle CSV (no angle state found)" bus = bus
        return nothing
    end
    t, dδ, lab, sym = series
    df_out = DataFrame(t_s = t, rotor_angle_dev_rad = dδ)
    fname === nothing && (fname = @sprintf("rotor_angle_dev_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, df_out)
    @info "Saved rotor-angle deviation CSV" bus = bus label = lab state = string(sym) file = fpath resampled = RESAMPLE_FOR_CSV export_dt = EXPORT_DT
    return fpath
end

function save_frequency_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                    fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    series = frequency_deviation_hz_at_bus(results, sys, bus)
    if series === nothing
        @warn "Cannot export CSV: no frequency series found" bus = bus
        return nothing
    end
    t, df, lab, sym = series
    if RESAMPLE_FOR_CSV
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        df = interp1_linear(t, df, tg)
        t  = tg
    end
    df_out = DataFrame(t_s = t, df_hz = df)
    fname === nothing && (fname = @sprintf("freq_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, df_out)
    @info "Saved frequency CSV" bus = bus label = lab state = string(sym) file = fpath resampled = RESAMPLE_FOR_CSV export_dt = EXPORT_DT
    return fpath
end

# SG mechanical-power export
function get_mechanical_power_dev_pu_at_bus(results, sys::System, bus::Int;
                                            t_step::Float64 = 1.0, for_csv::Bool = false, for_plots::Bool = false)
    dev   = gen_at_bus(sys, bus)
    name  = PSY.get_name(dev)
    mBase = PSY.get_base_power(dev)
    ratio = mBase / SYS_BASE_MVA
    pm_series = nothing
    for s in (:x_g1, :x_g2, :x_g3)
        try
            t, x = get_state_series(results, (name, s))
            pm_series = (t, x, s)
            @info "SG mech power: using TGTypeI state $(s) for bus $(bus)" mBase_MVA = mBase
            break
        catch
        end
    end
    if pm_series === nothing
        @warn "No TGTypeI states (x_g1/x_g2/x_g3) found for bus $bus"
        return nothing
    end
    t, xg1, src_sym = pm_series
    Pm_sys = xg1 .* ratio
    if for_csv && RESAMPLE_FOR_CSV
        tg     = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        Pm_sys = interp1_linear(t, Pm_sys, tg)
        t      = tg
    elseif for_plots && RESAMPLE_FOR_PLOTS
        tg     = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        Pm_sys = interp1_linear(t, Pm_sys, tg)
        t      = tg
    end
    i0  = first_finite_index(Pm_sys)
    Pm0 = (i0 !== nothing) ? Pm_sys[i0] : 0.0
    ΔPm = Pm_sys .- Pm0
    label = @sprintf("Bus %d (SG)  ΔP_m [src=%s, mBase=%.0f MVA, P_m0=%.4f pu sys]", bus, string(src_sym), mBase, Pm0)
    return (t, ΔPm, Pm0, mBase, src_sym, label)
end

function save_mechanical_power_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                           fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    out = get_mechanical_power_dev_pu_at_bus(results, sys, bus; t_step = t_step, for_csv = true)
    if out === nothing
        @warn "Skipping SG mechanical power CSV" bus = bus
        return nothing
    end
    t, ΔPm, Pm0, mBase, src_sym, label = out
    Pm_abs = ΔPm .+ Pm0
    fname === nothing && (fname = @sprintf("pg_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, DataFrame(
        t_s = t, pG_pu = ΔPm, Pm_abs_pu_sysbase = Pm_abs, Pm_initial_pu_sysbase = fill(Pm0, length(t))))
    @info "Saved SG mechanical power CSV" bus = bus src = string(src_sym) mBase_MVA = mBase Pm0_pu_sysbase = Pm0 file = fpath
    return fpath
end

# Inverter power export
function filter_finite_xy(t::AbstractVector, y::AbstractVector)
    mask = [isfinite(ti) && isfinite(yi) for (ti, yi) in zip(t, y)]
    return t[mask], y[mask], mask
end

function dyn_inverter_at_bus(sys::System, busnum::Int)
    g = gen_at_bus(sys, busnum)
    target_name = PSY.get_name(g)
    for d in get_components(DynamicInverter, sys)
        if PSY.get_name(d) == target_name
            return d
        end
    end
    error("No DynamicInverter found at bus $busnum (expected name = $target_name)")
end

function get_gfm_power_dev_pu_at_bus(results, sys::System, bus::Int;
                                     t_step::Float64 = 1.0, for_csv::Bool = false, for_plots::Bool = false)
    dev  = gen_at_bus(sys, bus)
    name = PSY.get_name(dev)
    t = nothing
    P = nothing
    try
        t_s, P_s = get_activepower_series(results, name)
        t = Float64.(t_s)
        P = Float64.(P_s)
        @info "GFM power: using get_activepower_series" bus = bus dev = name
    catch
    end
    if t === nothing
        for sym in (:Pel, :P_oc, :Pe, :p_el, :active_power, :P_elec, :p_elec)
            try
                t_s, x_s = get_state_series(results, (name, sym))
                t = Float64.(t_s)
                P = Float64.(x_s)
                @info "GFM power: using state fallback" bus = bus dev = name state = string(sym)
                break
            catch
            end
        end
    end
    if t === nothing
        @warn "No GFM active-power series found" bus = bus dev = name
        return nothing
    end
    if (for_csv && RESAMPLE_FOR_CSV) || (for_plots && RESAMPLE_FOR_PLOTS)
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        P  = interp1_linear(t, P, tg)
        t  = tg
    end
    i0 = first_finite_index(P)
    P0 = (i0 !== nothing) ? P[i0] : 0.0
    ΔP = P .- P0
    label = @sprintf("Bus %d (GFM)  ΔP [P0=%.4f pu]", bus, P0)
    return (t, ΔP, P0, name, label)
end

function save_gfm_power_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                    fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    out = get_gfm_power_dev_pu_at_bus(results, sys, bus; t_step = t_step, for_csv = true)
    if out === nothing
        @warn "Skipping GFM active-power CSV" bus = bus
        return nothing
    end
    t, ΔP, P0, dev_name, label = out
    P_abs = ΔP .+ P0
    fname === nothing && (fname = @sprintf("gfm_power_dev_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, DataFrame(t_s = t, delta_p_pu = ΔP, p_abs_pu = P_abs, p_initial_pu = fill(P0, length(t))))
    @info "Saved GFM active-power CSV" bus = bus dev = dev_name P0_pu = P0 file = fpath
    return fpath
end

function get_gfl_pref_eff_series(results, sys::System, bus::Int;
                                 pll_kp::Float64 = getGflPllKp(bus),
                                 pll_ki::Float64 = getGflPllKi(bus),
                                 t_step::Float64 = 1.0, for_csv::Bool = false, for_plots::Bool = false)
    dyn = dyn_inverter_at_bus(sys, bus)
    name = PSY.get_name(dyn)
    kp_use, ki_use = gfl_pll_gains_at_bus(sys, bus)
    pll = get_gfl_pll_speed_pu(results, name; kp_pll = kp_use, ki_pll = ki_use)
    if pll === nothing
        @warn "Could not reconstruct GFL effective active-power reference: no PLL frequency series found" bus = bus dev = name
        return nothing
    end
    t, ω_pll, ωsym = pll
    outer = PSY.get_outer_control(dyn)
    ap = PSY.get_active_power_control(outer)
    p_ref = try PSY.get_P_ref(ap) catch; 0.0 end
    Kω = try get(PSY.get_ext(ap), "Kω", getGflKw(bus)) catch; getGflKw(bus) end
    p_ref_eff  = p_ref .- Kω .* (ω_pll .- 1.0)
    Δp_ref_eff = p_ref_eff .- p_ref
    if (for_csv && RESAMPLE_FOR_CSV) || (for_plots && RESAMPLE_FOR_PLOTS)
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        p_ref_eff  = interp1_linear(t, p_ref_eff, tg)
        Δp_ref_eff = interp1_linear(t, Δp_ref_eff, tg)
        ω_pll      = interp1_linear(t, ω_pll, tg)
        t = tg
    end
    label = @sprintf("Bus %d (GFL)  p_ref=%.4f pu, Kω=%.4f", bus, p_ref, Kω)
    return (t, p_ref_eff, Δp_ref_eff, fill(p_ref, length(t)), ω_pll, Kω, ωsym, label)
end

function save_gfl_pref_eff_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                       fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    out = get_gfl_pref_eff_series(results, sys, bus; t_step = t_step, for_csv = true)
    if out === nothing
        @warn "Skipping GFL effective active-power-reference CSV" bus = bus
        return nothing
    end
    t, p_ref_eff, Δp_ref_eff, p_ref_const, ω_pll, Kω, ωsym, label = out
    fname === nothing && (fname = @sprintf("gfl_pref_eff_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, DataFrame(
        t_s = t, p_ref_eff_pu = p_ref_eff, delta_pref_eff_pu = Δp_ref_eff,
        p_ref_pu = p_ref_const, omega_pll_pu = ω_pll, Kω = fill(Kω, length(t))))
    @info "Saved GFL effective active-power-reference CSV" bus = bus Kω = Kω pll_state = string(ωsym) file = fpath
    return fpath
end

# Terminal P/Q export
function terminal_power_series(results, sys::System, bus::Int; channel::Symbol)
    g = gen_at_bus(sys, bus)
    name = PSY.get_name(g)
    if channel == :P
        try
            t, p = get_activepower_series(results, name)
            return (Float64.(t), Float64.(p), :get_activepower_series)
        catch
        end
        for sym in (:Pel, :P_el, :Pe, :P_e, :p_el, :p_e, :P_oc, :active_power, :P_elec, :p_elec, :P, :p)
            try
                t, x = get_state_series(results, (name, sym))
                return (Float64.(t), Float64.(x), sym)
            catch
            end
        end
        @warn "No terminal active-power series found" bus = bus dev = name
        return nothing
    elseif channel == :Q
        try
            t, q = get_reactivepower_series(results, name)
            return (Float64.(t), Float64.(q), :get_reactivepower_series)
        catch
        end
        for sym in (:Qel, :Q_el, :Qe, :Q_e, :q_el, :q_e, :Q_oc, :reactive_power, :Q_elec, :q_elec, :Q, :q)
            try
                t, x = get_state_series(results, (name, sym))
                return (Float64.(t), Float64.(x), sym)
            catch
            end
        end
        @warn "No terminal reactive-power series found" bus = bus dev = name
        return nothing
    else
        error("terminal_power_series channel must be :P or :Q")
    end
end

function first_finite_value(v::AbstractVector)
    for x in v
        isfinite(x) && return x
    end
    return 0.0
end

function grid_with_events(events::Vector{Float64})
    tg = collect(0.0:EXPORT_DT:TF_SIM)
    for tev in events
        if !(any(abs.(tg .- tev) .< 1e-12))
            push!(tg, tev)
        end
    end
    sort!(tg)
    return tg
end

function resample_to_export_grid(t::AbstractVector, y::AbstractVector, events::Vector{Float64})
    !RESAMPLE_FOR_CSV && return Float64.(t), Float64.(y)
    tg = grid_with_events(events)
    yg = interp1_linear(t, y, tg)
    return tg, yg
end

function terminal_pq_dev(results, sys::System, bus::Int)
    ps = terminal_power_series(results, sys, bus; channel = :P)
    qs = terminal_power_series(results, sys, bus; channel = :Q)
    if ps === nothing && qs === nothing
        return nothing
    end
    if ps !== nothing
        tP, P, psrc = ps
        t_ref = tP
    else
        tQ, Qtmp, qsrc_tmp = qs
        t_ref = tQ
        P = fill(NaN, length(t_ref))
        psrc = :missing
    end
    if qs !== nothing
        tQ, Q, qsrc = qs
    else
        Q = fill(NaN, length(t_ref))
        qsrc = :missing
    end
    if ps !== nothing && qs !== nothing
        Q = interp1_linear(tQ, Q, t_ref)
    end
    P0 = first_finite_value(P)
    Q0 = first_finite_value(Q)
    ΔP = P .- P0
    ΔQ = Q .- Q0
    return (t = t_ref, P = P, Q = Q, ΔP = ΔP, ΔQ = ΔQ, P0 = P0, Q0 = Q0,
            psrc = psrc, qsrc = qsrc, dev = PSY.get_name(gen_at_bus(sys, bus)),
            label = @sprintf("Bus %d (%s) terminal P/Q", bus, model_label(bus)))
end

function save_terminal_pq_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                      event_times::Vector{Float64} = Float64[])
    mkpath(outdir)
    out = terminal_pq_dev(results, sys, bus)
    if out === nothing
        @warn "Skipping terminal P/Q CSV: no terminal P/Q series found" bus = bus
        return nothing
    end
    t, ΔP   = resample_to_export_grid(out.t, out.ΔP, event_times)
    _, ΔQ   = resample_to_export_grid(out.t, out.ΔQ, event_times)
    _, Pabs = resample_to_export_grid(out.t, out.P, event_times)
    _, Qabs = resample_to_export_grid(out.t, out.Q, event_times)
    fpath = joinpath(outdir, @sprintf("terminal_pq_bus%02d_julia.csv", bus))
    CSV.write(fpath, DataFrame(
        t_s = t, delta_p_pu = ΔP, p_abs_pu = Pabs, p_initial_pu = fill(out.P0, length(t)),
        delta_q_pu = ΔQ, delta_q_injection_pu = ΔQ, q_abs_pu = Qabs, q_injection_abs_pu = Qabs,
        q_initial_pu = fill(out.Q0, length(t)), delta_q_absorbed_pu = -ΔQ, q_absorbed_abs_pu = -Qabs,
        asset_type = fill(model_label(bus), length(t)), device_name = fill(out.dev, length(t)),
        p_source = fill(string(out.psrc), length(t)), q_source = fill(string(out.qsrc), length(t))))
    @info "Saved terminal P/Q CSV" bus = bus file = fpath P0 = out.P0 Q0 = out.Q0 psrc = string(out.psrc) qsrc = string(out.qsrc)
    return fpath
end

function print_terminal_pq_final_summary(results, sys::System; buses::Vector{Int}, tail_seconds::Float64 = 2.0)
    println("\n" * "="^110)
    println("FINAL / STEADY-STATE TERMINAL P/Q DEVIATION ESTIMATE")
    println(@sprintf("Method: last sample + mean over final %.2f s; P,Q are terminal PSID series on SYSTEM_BASE", tail_seconds))
    println("="^110)
    @printf("%6s  %-9s  %12s  %12s  %12s  %12s  %12s  %12s\n",
            "Bus", "Model", "ΔP_last", "ΔP_mean", "P0", "ΔQ_last", "ΔQ_mean", "Q0")
    println("-"^110)
    for bus in buses
        out = terminal_pq_dev(results, sys, bus)
        if out === nothing
            @printf("%6d  %-9s  %s\n", bus, model_label(bus), "NO_TERMINAL_PQ_SERIES")
            continue
        end
        p_last, p_mean, _, _, _ = tail_stats(out.t, out.ΔP; tail_seconds = tail_seconds)
        q_last, q_mean, _, _, _ = tail_stats(out.t, out.ΔQ; tail_seconds = tail_seconds)
        @printf("%6d  %-9s  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f  %12.6f\n",
                bus, model_label(bus), p_last, p_mean, out.P0, q_last, q_mean, out.Q0)
    end
    println("="^110 * "\n")
end

#Network P/Q reconstruction
normalize_pair(i::Int, j::Int) = i < j ? (i, j) : (j, i)

function line_pair(line)
    f = nothing
    t = nothing
    try
        arc = PSY.get_arc(line)
        f = PSY.get_number(arc.from)
        t = PSY.get_number(arc.to)
    catch
        try
            buses = PSY.get_buses(line)
            f = PSY.get_number(buses[1])
            t = PSY.get_number(buses[2])
        catch err
            error("Could not recover terminal buses for line $(PSY.get_name(line)). Error = $err")
        end
    end
    return normalize_pair(Int(f), Int(t))
end

function _complex_vector_or_nothing(y)
    try
        if eltype(y) <: Complex
            return ComplexF64.(y)
        end
    catch
    end
    return nothing
end

function _maybe_extract_complex_voltage_time_value(out)
    if out isa Tuple && length(out) >= 2
        t = Float64.(out[1])
        vc = _complex_vector_or_nothing(out[2])
        vc !== nothing && return t, vc
    end
    if out isa DataFrame
        names_df = names(out)
        t_col = nothing
        for c in ["t_s", "time", "t", "Time"]
            if c in names_df
                t_col = c
                break
            end
        end
        t_col === nothing && return nothing
        for c in ["V", "v", "V_complex", "voltage", "voltage_complex", "bus_voltage", "phasor", "V_phasor"]
            if c in names_df
                vc = _complex_vector_or_nothing(out[:, c])
                vc !== nothing && return Float64.(out[:, t_col]), vc
            end
        end
        for (cr, ci) in [("Vr", "Vi"), ("V_re", "V_im"), ("real", "imag"),
                         ("V_real", "V_imag"), ("voltage_real", "voltage_imag"),
                         ("re", "im"), ("d", "q"), ("vd", "vq"), ("Vd", "Vq")]
            if cr in names_df && ci in names_df
                return Float64.(out[:, t_col]), ComplexF64.(Float64.(out[:, cr]), Float64.(out[:, ci]))
            end
        end
    end
    return nothing
end

function bus_voltage_complex_phasor_series(results, sys::System, bus::Int)
    b = bus_component(sys, bus)
    bus_name = PSY.get_name(b)
    fn_candidates = Symbol[:get_voltage_series, :get_bus_voltage_series, :get_voltage_phasor_series,
        :get_bus_voltage_phasor_series, :get_complex_voltage_series, :get_bus_complex_voltage_series]
    arg_candidates = Any[bus, bus_name, b, (bus,), (bus_name,)]
    for fn_sym in fn_candidates
        fn = _find_callable(fn_sym)
        fn === nothing && continue
        for arg in arg_candidates
            try
                out = fn(results, arg)
                tv = _maybe_extract_complex_voltage_time_value(out)
                if tv !== nothing
                    t, Vc = tv
                    @info "Exact complex bus-voltage phasor selected for network P/Q" bus = bus source = string(fn_sym)
                    return (t, Vc, @sprintf("Bus %d exact complex bus-voltage phasor from %s", bus, string(fn_sym)), fn_sym, true)
                end
            catch
            end
        end
    end
    try
        vout = voltage_magnitude_pu(results, sys, bus)
        if vout !== nothing
            tV, Vmag, _, vsrc = vout
            angle_fn = _find_callable(:get_voltage_angle_series)
            if angle_fn !== nothing
                for arg in arg_candidates
                    try
                        outθ = angle_fn(results, arg)
                        if outθ isa Tuple && length(outθ) >= 2
                            tθ = Float64.(outθ[1])
                            θraw = Float64.(outθ[2])
                            θ_on_tV = interp1_linear(tθ, unwrap_angle_rad(θraw), Float64.(tV))
                            if any(isfinite.(θ_on_tV))
                                θfill = first_finite_value(θ_on_tV)
                                θ_on_tV = [isfinite(x) ? x : θfill for x in θ_on_tV]
                                Vc = ComplexF64.(Float64.(Vmag) .* exp.(1im .* θ_on_tV))
                                @info "Exact bus-voltage phasor constructed from magnitude and bus angle" bus = bus v_source = string(vsrc) angle_source = "get_voltage_angle_series"
                                return (Float64.(tV), Vc,
                                        @sprintf("Bus %d exact bus-voltage phasor from |V| and get_voltage_angle_series", bus),
                                        Symbol("Vmag_" * string(vsrc) * "__angle_get_voltage_angle_series"), true)
                            end
                        end
                    catch
                    end
                end
            end
        end
    catch err
        @warn "Could not construct exact bus-voltage phasor from magnitude and angle" bus = bus err = string(err)
    end
    @warn "Exact bus-voltage phasor unavailable; network P/Q may need a generator-device angle fallback" bus = bus model = model_label(bus)
    return nothing
end

function raw_device_angle_rad_for_network(results, sys::System, bus::Int)
    g = nothing
    try
        g = gen_at_bus(sys, bus)
    catch
        return nothing
    end
    name = PSY.get_name(g)
    syms = Symbol[:δ, :delta, :θ_oc, :theta_oc, :θ_olc, :theta_olc, :θ_pll, :theta_pll, :θ, :theta]
    s = get_first_state_series(results, name, syms)
    s === nothing && return nothing
    t, θraw, sym = s
    return (Float64.(t), unwrap_angle_rad(Float64.(θraw)), sym)
end

function _line_parameter_or_default(line, getter_sym::Symbol, default_value::Float64)
    try
        getter = getfield(PSY, getter_sym)
        return Float64(getter(line))
    catch
        return default_value
    end
end

function build_ybus_dense_from_system(sys::System; buses::Vector{Int} = NETWORK_PQ_ALL_BUSES)
    nb = length(buses)
    bus_to_idx = Dict{Int,Int}(b => i for (i, b) in enumerate(buses))
    Y = zeros(ComplexF64, nb, nb)
    n_stamped = 0
    for line in get_components(Line, sys)
        pair = line_pair(line)
        f, t = pair
        if !(haskey(bus_to_idx, f) && haskey(bus_to_idx, t))
            continue
        end
        r = _line_parameter_or_default(line, :get_r, NaN)
        x = _line_parameter_or_default(line, :get_x, NaN)
        b_sh = _line_parameter_or_default(line, :get_b, 0.0)
        if !isfinite(r) || !isfinite(x)
            @warn "Could not read line R/X for Ybus build" line = PSY.get_name(line) pair = pair
            continue
        end
        z = complex(r, x)
        if abs(z) < 1e-12
            @warn "Skipping near-zero impedance line in Ybus build" line = PSY.get_name(line) pair = pair
            continue
        end
        y = inv(z)
        ysh = 1im * b_sh / 2.0
        i = bus_to_idx[f]
        j = bus_to_idx[t]
        Y[i, i] += y + ysh
        Y[j, j] += y + ysh
        Y[i, j] -= y
        Y[j, i] -= y
        n_stamped += 1
    end
    if n_stamped == 0
        @warn "No Line components were stamped into Ybus; using embedded IEEE-39 MATPOWER branch fallback. This avoids the false all-buses-islanded NetworkSwitch error after DynamicBranch conversion."
        return build_ybus_dense_from_ieee39_fallback(; buses = buses)
    end
    return Y
end

function pf_reference_vectors(pf_data::PowerFlowData; buses::Vector{Int} = NETWORK_PQ_ALL_BUSES)
    V0 = zeros(Float64, length(buses))
    θ0 = zeros(Float64, length(buses))
    for (i, b) in enumerate(buses)
        idx = pf_data.bus_lookup[b]
        V0[i] = Float64(pf_data.bus_magnitude[idx])
        θ0[i] = Float64(pf_data.bus_angles[idx])
    end
    ref_idx = findfirst(==(39), buses)
    ref_idx !== nothing && (θ0 .-= θ0[ref_idx])
    return V0, θ0
end

function local_load_step_series(t::AbstractVector, steps::Vector{RealizedLoadStep}, bus::Int)
    ΔP = zeros(Float64, length(t))
    ΔQ = zeros(Float64, length(t))
    for s in steps
        s.bus != bus && continue
        for k in eachindex(t)
            if t[k] >= s.t_step - 1e-12
                ΔP[k] += s.deltaP_pu
                ΔQ[k] += s.deltaQ_pu
            end
        end
    end
    return ΔP, ΔQ
end

function network_bus_pq_dev(results, sys::System, pf_data::PowerFlowData, steps::Vector{RealizedLoadStep}; buses::Vector{Int} = NETWORK_PQ_ALL_BUSES)
    ev_times = unique(sort([s.t_step for s in steps]))
    t_grid = grid_with_events(ev_times)
    Vmag_mat = zeros(Float64, length(buses), length(t_grid))
    θdev_raw_mat = zeros(Float64, length(buses), length(t_grid))
    angle_source = String[]
    angle_is_exact_bus_voltage = Bool[]
    for (i, b) in enumerate(buses)
        vc_out = bus_voltage_complex_phasor_series(results, sys, b)
        if vc_out !== nothing
            tv, Vc_raw, _, src, exact_flag = vc_out
            Vmag_mat[i, :] .= interp1_linear(tv, abs.(Vc_raw), t_grid)
            θraw = unwrap_angle_rad(angle.(Vc_raw))
            θraw_on_grid = interp1_linear(tv, θraw, t_grid)
            θ0_raw = first_finite_value(θraw_on_grid)
            θdev_raw_mat[i, :] .= θraw_on_grid .- θ0_raw
            push!(angle_source, string(src))
            push!(angle_is_exact_bus_voltage, exact_flag)
            continue
        end
        vout = voltage_magnitude_pu(results, sys, b)
        if vout === nothing
            @warn "Cannot export network P/Q because bus-voltage magnitude series is missing" bus = b
            return nothing
        end
        tv, V, _, vsrc = vout
        Vmag_mat[i, :] .= interp1_linear(tv, V, t_grid)
        if b == 39
            θdev_raw_mat[i, :] .= 0.0
            push!(angle_source, "fixed_network_reference_zero")
            push!(angle_is_exact_bus_voltage, false)
            @warn "Network P/Q export uses fixed slack angle fallback, not exact bus-voltage angle" bus = b source = string(vsrc)
            continue
        end
        aout = raw_device_angle_rad_for_network(results, sys, b)
        if aout === nothing
            @warn "Cannot export network P/Q because no exact bus angle and no device-angle fallback exists" bus = b
            return nothing
        end
        ta, θraw, θsym = aout
        θraw_on_grid = interp1_linear(ta, θraw, t_grid)
        θ0_raw = first_finite_value(θraw_on_grid)
        θdev_raw_mat[i, :] .= θraw_on_grid .- θ0_raw
        push!(angle_source, "device_angle_fallback_" * string(θsym))
        push!(angle_is_exact_bus_voltage, false)
        @warn "Network P/Q export uses device/control angle fallback; Q may be biased" bus = b model = model_label(b) state = string(θsym)
    end
    ref_idx = findfirst(==(39), buses)
    ref_idx === nothing && error("Bus 39 reference is missing from network P/Q export buses=$(buses)")
    θdev_mat = similar(θdev_raw_mat)
    for k in eachindex(t_grid)
        θref_dev = θdev_raw_mat[ref_idx, k]
        θdev_mat[:, k] .= θdev_raw_mat[:, k] .- θref_dev
    end
    θdev_mat[ref_idx, :] .= 0.0
    Y = build_ybus_dense_from_system(sys; buses = buses)
    V0, θ0 = pf_reference_vectors(pf_data; buses = buses)
    Vref = V0 .* exp.(1im .* θ0)
    Sref = Vref .* conj.(Y * Vref)
    Pref = real.(Sref)
    Qref = imag.(Sref)
    P = zeros(Float64, length(buses), length(t_grid))
    Q = zeros(Float64, length(buses), length(t_grid))
    for k in eachindex(t_grid)
        θ = θ0 .+ θdev_mat[:, k]
        Vc = Vmag_mat[:, k] .* exp.(1im .* θ)
        S = Vc .* conj.(Y * Vc)
        P[:, k] .= real.(S)
        Q[:, k] .= imag.(S)
    end
    all_exact = all(angle_is_exact_bus_voltage)
    if !all_exact
        @warn "Network P/Q export is diagnostic because at least one bus used an angle fallback. For exact Q validation, PSID must expose complex bus-voltage phasors." angle_source = angle_source exact = angle_is_exact_bus_voltage
    end
    return (t = t_grid, buses = buses, P = P, Q = Q, Pref = Pref, Qref = Qref,
            angle_source = angle_source, angle_is_exact_bus_voltage = angle_is_exact_bus_voltage)
end

function save_network_pq_csv(results, sys::System, pf_data::PowerFlowData, bus::Int, outdir::String, steps::Vector{RealizedLoadStep}; net_cache = nothing)
    mkpath(outdir)
    net = net_cache
    net === nothing && (net = network_bus_pq_dev(results, sys, pf_data, steps; buses = NETWORK_PQ_ALL_BUSES))
    if net === nothing
        @warn "No network bus P/Q series for CSV" bus = bus
        return nothing
    end
    idx = findfirst(==(bus), net.buses)
    if idx === nothing
        @warn "Requested bus missing from network P/Q export" bus = bus buses = net.buses
        return nothing
    end
    Pabs = vec(net.P[idx, :])
    Qabs = vec(net.Q[idx, :])
    ΔP_network = Pabs .- net.Pref[idx]
    ΔQ_network = Qabs .- net.Qref[idx]
    ΔP_load, ΔQ_load = local_load_step_series(net.t, steps, bus)
    ΔP_with_load = ΔP_network .+ ΔP_load
    ΔQ_with_load = ΔQ_network .+ ΔQ_load
    path = joinpath(outdir, @sprintf("network_pq_bus%02d_julia.csv", bus))
    CSV.write(path, DataFrame(
        t_s = net.t, delta_p_pu = ΔP_with_load, delta_q_pu = ΔQ_with_load,
        delta_p_network_pu = ΔP_network, delta_q_network_pu = ΔQ_network,
        delta_p_with_load_pu = ΔP_with_load, delta_q_with_load_pu = ΔQ_with_load,
        p_network_abs_pu = Pabs, q_network_abs_pu = Qabs,
        p_network_ref_pu = fill(net.Pref[idx], length(net.t)), q_network_ref_pu = fill(net.Qref[idx], length(net.t)),
        delta_p_load_pu = ΔP_load, delta_q_load_pu = ΔQ_load,
        angle_source = fill(net.angle_source[idx], length(net.t)),
        angle_is_exact_bus_voltage = fill(net.angle_is_exact_bus_voltage[idx], length(net.t)),
        asset_type = fill(model_label(bus), length(net.t)),
        source = fill("Ybus EMT reconstruction using exact complex bus-voltage phasors when available; otherwise diagnostic fallback", length(net.t))))
    @info "Saved network bus P/Q CSV" bus = bus asset = model_label(bus) angle_source = net.angle_source[idx] angle_is_exact_bus_voltage = net.angle_is_exact_bus_voltage[idx] file = path
    return path
end

function print_network_pq_reconstruction_summary(net)
    net === nothing && return
    println("\n" * "="^112)
    println("NETWORK/YBUS P/Q RECONSTRUCTION SOURCE SUMMARY")
    println("="^112)
    @printf("%-8s %-12s %-10s %-70s\n", "Bus", "Model", "Exact?", "Angle source")
    println("-"^112)
    for (i, b) in enumerate(net.buses)
        if b in NETWORK_PQ_EXPORT_BUSES
            @printf("%-8d %-12s %-10s %-70s\n", b, model_label(b), string(net.angle_is_exact_bus_voltage[i]), net.angle_source[i])
        end
    end
    println("="^112 * "\n")
end

function plot_terminal_pq_panels(results, sys::System, buses::Vector{Int}, plotdir::String, load_event_times::Vector{Float64}, load_event_summary::String;
                                 fname_prefix::String = "ieee39_terminal")
    mkpath(plotdir)
    pP = plot(title = "Terminal active-power deviation (device electrical output)\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "ΔP [pu, system base]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 620),
        titlefont = font(11, :bold), guidefont = font(14, :bold), tickfont = font(12, :bold), legendfont = font(9))
    pQ = plot(title = "Terminal reactive-power deviation (device electrical output)\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "ΔQ [pu, injection sign]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 620),
        titlefont = font(11, :bold), guidefont = font(14, :bold), tickfont = font(12, :bold), legendfont = font(9))
    for bus in buses
        out = terminal_pq_dev(results, sys, bus)
        if out === nothing
            @warn "Skipping terminal P/Q panel bus" bus = bus
            continue
        end
        tP, ΔP = resample_to_export_grid(out.t, out.ΔP, load_event_times)
        tQ, ΔQ = resample_to_export_grid(out.t, out.ΔQ, load_event_times)
        plot!(pP, tP, ΔP; lw = 2.8, color = bus_plot_color(bus), label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
        plot!(pQ, tQ, ΔQ; lw = 2.8, color = bus_plot_color(bus), label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
    end
    hline!(pP, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    hline!(pQ, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    for tev in load_event_times
        vline!(pP, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = "")
        vline!(pQ, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = "")
    end
    savefig(pP, joinpath(plotdir, fname_prefix * "_active_power_all_buses.png"))
    savefig(pP, joinpath(plotdir, fname_prefix * "_active_power_all_buses.pdf"))
    savefig(pQ, joinpath(plotdir, fname_prefix * "_reactive_power_all_buses.png"))
    savefig(pQ, joinpath(plotdir, fname_prefix * "_reactive_power_all_buses.pdf"))
    combined = plot(pP, pQ; layout = (2, 1), size = (1500, 1200), dpi = 140, left_margin = 10Plots.mm, right_margin = 4Plots.mm, bottom_margin = 6Plots.mm)
    savefig(combined, joinpath(plotdir, fname_prefix * "_active_reactive_power_panels.png"))
    savefig(combined, joinpath(plotdir, fname_prefix * "_active_reactive_power_panels.pdf"))
    @info "Saved terminal active/reactive power panels" folder = plotdir prefix = fname_prefix
    return combined
end

function plot_network_pq_panels(net, buses::Vector{Int}, plotdir::String, steps::Vector{RealizedLoadStep}, load_event_times::Vector{Float64}, load_event_summary::String;
                                fname_prefix::String = "ieee39_network_reconstructed")
    net === nothing && return nothing
    mkpath(plotdir)
    pP = plot(title = "Reconstructed network active-power deviation from Ybus\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "ΔP_network + local ΔP_load [pu]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 620),
        titlefont = font(11, :bold), guidefont = font(14, :bold), tickfont = font(12, :bold), legendfont = font(9))
    pQ = plot(title = "Reconstructed network reactive-power deviation from Ybus\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "ΔQ_network + local ΔQ_load [pu]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 620),
        titlefont = font(11, :bold), guidefont = font(14, :bold), tickfont = font(12, :bold), legendfont = font(9))
    for bus in buses
        idx = findfirst(==(bus), net.buses)
        idx === nothing && continue
        Pabs = vec(net.P[idx, :])
        Qabs = vec(net.Q[idx, :])
        ΔP_network = Pabs .- net.Pref[idx]
        ΔQ_network = Qabs .- net.Qref[idx]
        ΔP_load, ΔQ_load = local_load_step_series(net.t, steps, bus)
        ΔP = ΔP_network .+ ΔP_load
        ΔQ = ΔQ_network .+ ΔQ_load
        plot!(pP, net.t, ΔP; lw = 2.8, color = bus_plot_color(bus), label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
        plot!(pQ, net.t, ΔQ; lw = 2.8, color = bus_plot_color(bus), label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
    end
    hline!(pP, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    hline!(pQ, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    for tev in load_event_times
        vline!(pP, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = "")
        vline!(pQ, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = "")
    end
    savefig(pP, joinpath(plotdir, fname_prefix * "_active_power_all_buses.png"))
    savefig(pP, joinpath(plotdir, fname_prefix * "_active_power_all_buses.pdf"))
    savefig(pQ, joinpath(plotdir, fname_prefix * "_reactive_power_all_buses.png"))
    savefig(pQ, joinpath(plotdir, fname_prefix * "_reactive_power_all_buses.pdf"))
    combined = plot(pP, pQ; layout = (2, 1), size = (1500, 1200), dpi = 140, left_margin = 10Plots.mm, right_margin = 4Plots.mm, bottom_margin = 6Plots.mm)
    savefig(combined, joinpath(plotdir, fname_prefix * "_active_reactive_power_panels.png"))
    savefig(combined, joinpath(plotdir, fname_prefix * "_active_reactive_power_panels.pdf"))
    @info "Saved reconstructed network active/reactive power panels" folder = plotdir prefix = fname_prefix
    return combined
end

# Voltage export and plots
function _as_float_vector(y)
    if eltype(y) <: Complex
        return Float64.(abs.(y))
    end
    return Float64.(y)
end

function _maybe_extract_time_value(out)
    if out isa Tuple && length(out) >= 2
        t = Float64.(out[1])
        y = _as_float_vector(out[2])
        return t, y
    end
    if out isa DataFrame
        names_df = names(out)
        t_col = nothing
        for c in ["t_s", "time", "t", "Time"]
            if c in names_df
                t_col = c
                break
            end
        end
        y_col = nothing
        for c in ["V_pu", "Vm_pu", "Vmag_pu", "voltage_pu", "magnitude", "Vm"]
            if c in names_df
                y_col = c
                break
            end
        end
        if t_col !== nothing && y_col !== nothing
            return Float64.(out[:, t_col]), _as_float_vector(out[:, y_col])
        end
    end
    return nothing
end

function _find_callable(fn_sym::Symbol)
    isdefined(Main, fn_sym) && return getfield(Main, fn_sym)
    isdefined(PSID, fn_sym) && return getfield(PSID, fn_sym)
    return nothing
end

function voltage_magnitude_pu(results, sys::System, bus::Int)
    b = bus_component(sys, bus)
    bus_name = PSY.get_name(b)
    fn_candidates = Symbol[:get_voltage_magnitude_series, :get_bus_voltage_magnitude_series,
        :get_voltage_series, :get_bus_voltage_series]
    arg_candidates = Any[bus, bus_name, b, (bus,), (bus_name,)]
    for fn_sym in fn_candidates
        fn = _find_callable(fn_sym)
        fn === nothing && continue
        for arg in arg_candidates
            try
                out = fn(results, arg)
                tv = _maybe_extract_time_value(out)
                if tv !== nothing
                    t, V = tv
                    return (t, V, @sprintf("Bus %d voltage magnitude from %s", bus, string(fn_sym)), fn_sym)
                end
            catch
            end
        end
    end
    g = nothing
    try
        g = gen_at_bus(sys, bus)
    catch
        @warn "No bus-voltage accessor and no generator/device fallback for voltage magnitude" bus = bus bus_name = bus_name
        return nothing
    end
    dev_name = PSY.get_name(g)
    voltage_state_candidates = Symbol[:V_t, :Vt, :v_t, :vt, :V_mag, :Vmag, :Vm, :v_mag, :vm, :V, :v]
    s = get_first_state_series(results, dev_name, voltage_state_candidates)
    if s !== nothing
        t, Vraw, sym = s
        V = _as_float_vector(Vraw)
        return (t, V, @sprintf("Bus %d voltage magnitude from device state %s", bus, string(sym)), sym)
    end
    vd_candidates = Symbol[:vd, :v_d, :Vd, :V_d, :vd_filter, :vr_filter, :vd_pll, :vr_cnv]
    vq_candidates = Symbol[:vq, :v_q, :Vq, :V_q, :vq_filter, :vi_filter, :vq_pll, :vi_cnv]
    vd_state = get_first_state_series(results, dev_name, vd_candidates)
    vq_state = get_first_state_series(results, dev_name, vq_candidates)
    if vd_state !== nothing && vq_state !== nothing
        t_vd, vd_raw, vd_sym = vd_state
        t_vq, vq_raw, vq_sym = vq_state
        vd = Float64.(vd_raw)
        vq_on_vd_time = interp1_linear(t_vq, Float64.(vq_raw), t_vd)
        V = sqrt.(vd .^ 2 .+ vq_on_vd_time .^ 2)
        return (Float64.(t_vd), V,
                @sprintf("Bus %d voltage magnitude reconstructed from %s/%s", bus, string(vd_sym), string(vq_sym)),
                Symbol("Vmag_from_" * string(vd_sym) * "_" * string(vq_sym)))
    end
    @warn "No EMT bus voltage magnitude series found" bus = bus bus_name = bus_name dev = dev_name
    return nothing
end

function voltage_magnitude_for_bus(results, sys::System, bus::Int;
                                   t_step::Float64 = 1.0, for_csv::Bool = false, for_plots::Bool = false)
    out = voltage_magnitude_pu(results, sys, bus)
    out === nothing && return nothing
    t, V, lab, src = out
    if (for_csv && RESAMPLE_FOR_CSV) || (for_plots && RESAMPLE_FOR_PLOTS)
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        V  = interp1_linear(t, V, tg)
        t  = tg
    end
    i0 = first_finite_index(V)
    V0 = (i0 !== nothing) ? V[i0] : 0.0
    ΔV = V .- V0
    return (t, V, ΔV, V0, lab, src)
end

function save_voltage_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                  fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    out = voltage_magnitude_for_bus(results, sys, bus; t_step = t_step, for_csv = true)
    if out === nothing
        @warn "Skipping voltage CSV: no voltage magnitude series found" bus = bus
        return nothing
    end
    t, V, ΔV, V0, lab, src = out
    fname === nothing && (fname = @sprintf("voltage_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, DataFrame(t_s = t, V_pu = V, delta_V_pu = ΔV, V_initial_pu = fill(V0, length(t))))
    @info "Saved voltage magnitude/deviation CSV" bus = bus source = string(src) file = fpath resampled = RESAMPLE_FOR_CSV export_dt = EXPORT_DT
    return fpath
end

# Extra load-bus helpers
function print_ieee39_load_bus_export_list()
    println("\n" * "="^118)
    println("IEEE-39 LOAD-BUS EMT CSV EXPORT LIST")
    println("="^118)
    println("All IEEE-39 buses with non-zero demand in the embedded case:")
    println("  " * join(string.(IEEE39_LOAD_BUSES_ALL), ", "))
    println("Extra load buses exported in addition to the original generator-bus 30..39 export:")
    println("  " * join(string.(IEEE39_EXTRA_LOAD_BUSES), ", "))
    println("="^118 * "\n")
end

function write_ieee39_load_bus_list_csv(outdir::String)
    mkpath(outdir)
    gen_export_set = Set(collect(30:39))
    rows = DataFrame(
        bus = IEEE39_LOAD_BUSES_ALL,
        already_in_original_gen_export_30_39 = [b in gen_export_set for b in IEEE39_LOAD_BUSES_ALL],
        exported_by_extra_load_bus_loop = [b in IEEE39_EXTRA_LOAD_BUSES for b in IEEE39_LOAD_BUSES_ALL])
    fpath = joinpath(outdir, "ieee39_load_bus_list_julia.csv")
    CSV.write(fpath, rows)
    @info "Saved IEEE-39 load-bus list CSV" file = fpath n_all_load_buses = length(IEEE39_LOAD_BUSES_ALL) n_extra_load_buses = length(IEEE39_EXTRA_LOAD_BUSES)
    return fpath
end

function _central_derivative(t::AbstractVector, y::AbstractVector)
    n = length(t)
    dy = fill(NaN, n)
    n < 2 && return dy
    if n == 2
        dt = t[2] - t[1]
        if abs(dt) > 1e-12
            val = (y[2] - y[1]) / dt
            dy[1] = val
            dy[2] = val
        end
        return dy
    end
    for k in 1:n
        if k == 1
            dt = t[2] - t[1]
            dy[k] = abs(dt) > 1e-12 ? (y[2] - y[1]) / dt : NaN
        elseif k == n
            dt = t[n] - t[n - 1]
            dy[k] = abs(dt) > 1e-12 ? (y[n] - y[n - 1]) / dt : NaN
        else
            dt = t[k + 1] - t[k - 1]
            dy[k] = abs(dt) > 1e-12 ? (y[k + 1] - y[k - 1]) / dt : NaN
        end
    end
    return dy
end

function bus_voltage_angle_dev_rad_for_bus(results, sys::System, bus::Int;
                                           t_step::Float64 = 1.0, for_csv::Bool = false, for_plots::Bool = false)
    vc_out = bus_voltage_complex_phasor_series(results, sys, bus)
    if vc_out === nothing
        @warn "Skipping bus-voltage angle export: exact complex bus-voltage phasor unavailable" bus = bus
        return nothing
    end
    t, Vc, lab, src, exact_flag = vc_out
    θ = zero_start(unwrap_angle_rad(angle.(Vc)))
    if (for_csv && RESAMPLE_FOR_CSV) || (for_plots && RESAMPLE_FOR_PLOTS)
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        θ  = interp1_linear(t, θ, tg)
        t  = tg
    end
    return (Float64.(t), Float64.(θ), lab * " (bus-voltage angle deviation)", src, exact_flag)
end

function save_bus_voltage_angle_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                            fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    out = bus_voltage_angle_dev_rad_for_bus(results, sys, bus; t_step = t_step, for_csv = true)
    out === nothing && return nothing
    t, θdev, lab, src, exact_flag = out
    fname === nothing && (fname = @sprintf("rotor_angle_dev_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, DataFrame(
        t_s = t, rotor_angle_dev_rad = θdev, bus_voltage_angle_dev_rad = θdev,
        angle_source = fill(string(src), length(t)), angle_is_exact_bus_voltage = fill(exact_flag, length(t))))
    @info "Saved load-bus voltage-angle deviation CSV" bus = bus source = string(src) file = fpath resampled = RESAMPLE_FOR_CSV export_dt = EXPORT_DT
    return fpath
end

function bus_voltage_frequency_deviation_hz_for_bus(results, sys::System, bus::Int;
                                                    t_step::Float64 = 1.0, for_csv::Bool = false)
    vc_out = bus_voltage_complex_phasor_series(results, sys, bus)
    if vc_out === nothing
        @warn "Skipping bus-voltage frequency export: exact complex bus-voltage phasor unavailable" bus = bus
        return nothing
    end
    t, Vc, lab, src, exact_flag = vc_out
    θ = unwrap_angle_rad(angle.(Vc))
    if for_csv && RESAMPLE_FOR_CSV
        tg = uniform_grid_with_event(TF_SIM, EXPORT_DT, t_step)
        θ  = interp1_linear(t, θ, tg)
        t  = tg
    end
    dθdt = _central_derivative(Float64.(t), Float64.(θ))
    df_hz = dθdt ./ (2π)
    return (Float64.(t), Float64.(df_hz), lab * " (derived bus-voltage frequency deviation)", src, exact_flag)
end

function save_bus_voltage_frequency_csv_for_bus(results, sys::System, bus::Int, outdir::String;
                                                fname::Union{Nothing,String} = nothing, t_step::Float64 = 1.0)
    mkpath(outdir)
    out = bus_voltage_frequency_deviation_hz_for_bus(results, sys, bus; t_step = t_step, for_csv = true)
    out === nothing && return nothing
    t, df, lab, src, exact_flag = out
    fname === nothing && (fname = @sprintf("freq_bus%02d_julia.csv", bus))
    fpath = joinpath(outdir, fname)
    CSV.write(fpath, DataFrame(
        t_s = t, df_hz = df, frequency_source = fill(string(src), length(t)),
        frequency_is_derived_from_bus_voltage_angle = fill(true, length(t)),
        angle_is_exact_bus_voltage = fill(exact_flag, length(t))))
    @info "Saved load-bus voltage-angle-derived frequency CSV" bus = bus source = string(src) file = fpath resampled = RESAMPLE_FOR_CSV export_dt = EXPORT_DT
    return fpath
end

function export_extra_load_bus_emt_csvs(results, sys::System, pf_data::PowerFlowData,
                                        csvdir::String, realized_steps::Vector{RealizedLoadStep};
                                        t_step::Float64 = 1.0, net_cache = nothing)
    if !EXPORT_EXTRA_LOAD_BUS_CSVS
        @info "Extra IEEE-39 load-bus CSV export disabled"
        return nothing
    end
    print_ieee39_load_bus_export_list()
    write_ieee39_load_bus_list_csv(csvdir)
    @info "Exporting extra IEEE-39 load-bus voltage, angle, and derived-frequency CSVs" folder = csvdir buses = IEEE39_EXTRA_LOAD_BUSES
    for bus in IEEE39_EXTRA_LOAD_BUSES
        save_voltage_csv_for_bus(results, sys, bus, csvdir; t_step = t_step)
        save_bus_voltage_angle_csv_for_bus(results, sys, bus, csvdir; t_step = t_step)
        save_bus_voltage_frequency_csv_for_bus(results, sys, bus, csvdir; t_step = t_step)
    end
    if SAVE_DIAGNOSTIC_NETWORK_PQ
        net = net_cache
        net === nothing && (net = network_bus_pq_dev(results, sys, pf_data, realized_steps; buses = NETWORK_PQ_ALL_BUSES))
        if net !== nothing
            @info "Exporting extra IEEE-39 load-bus diagnostic network P/Q CSVs" folder = csvdir buses = IEEE39_EXTRA_LOAD_BUSES
            for bus in IEEE39_EXTRA_LOAD_BUSES
                save_network_pq_csv(results, sys, pf_data, bus, csvdir, realized_steps; net_cache = net)
            end
        else
            @warn "Skipping extra load-bus network P/Q CSVs because network P/Q reconstruction returned nothing"
        end
    end
    return true
end

function plot_voltage_magnitude_panel(results, sys::System, buses::Vector{Int}, load_event_times::Vector{Float64}, load_event_summary::String, plotdir::String;
                                      fname_prefix::String = "ieee39_voltage_magnitude")
    mkpath(plotdir)
    pV = plot(title = "Bus voltage magnitudes\n" * load_event_summary, xlabel = "Time [s]", ylabel = "V [pu]",
        size = (1600, 620), titlefont = font(8), legend = :best, grid = false, framestyle = :box,
        xlims = (0.0, TF_SIM), ylims = ABS_VOLTAGE_YLIM, yticks = ABS_VOLTAGE_YTICKS, widen = false)
    for bus in buses
        out = voltage_magnitude_for_bus(results, sys, bus; t_step = isempty(load_event_times) ? 0.25 : minimum(load_event_times), for_plots = true)
        if out === nothing
            @warn "Skipping voltage magnitude panel: no voltage series" bus = bus
            continue
        end
        t, V, _, V0, _, src = out
        plot!(pV, t, V; lw = 2, color = bus_plot_color(bus), label = @sprintf("Bus %d (%s), V0=%.4f", bus, string(src), V0))
    end
    for (k, tev) in enumerate(load_event_times)
        vline!(pV, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
    end
    png_path = joinpath(plotdir, fname_prefix * ".png")
    pdf_path = joinpath(plotdir, fname_prefix * ".pdf")
    savefig(pV, png_path)
    savefig(pV, pdf_path)
    @info "Saved voltage magnitude panel" png = png_path pdf = pdf_path
    return pV
end

function plot_voltage_deviation_panel(results, sys::System, buses::Vector{Int}, load_event_times::Vector{Float64}, load_event_summary::String, plotdir::String;
                                      fname_prefix::String = "ieee39_voltage_deviation")
    mkpath(plotdir)
    pDV = plot(title = "Bus voltage-magnitude deviations from initial value\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "ΔV [pu]", size = (1600, 620), titlefont = font(8), legend = :best,
        grid = false, framestyle = :box, xlims = (0.0, TF_SIM), widen = false)
    for bus in buses
        out = voltage_magnitude_for_bus(results, sys, bus; t_step = isempty(load_event_times) ? 0.25 : minimum(load_event_times), for_plots = true)
        if out === nothing
            @warn "Skipping voltage deviation panel: no voltage series" bus = bus
            continue
        end
        t, _, ΔV, V0, _, src = out
        plot!(pDV, t, ΔV; lw = 2, color = bus_plot_color(bus), label = @sprintf("Bus %d (%s), V0=%.4f", bus, string(src), V0))
    end
    hline!(pDV, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "zero reference")
    for (k, tev) in enumerate(load_event_times)
        vline!(pDV, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
    end
    png_path = joinpath(plotdir, fname_prefix * ".png")
    pdf_path = joinpath(plotdir, fname_prefix * ".pdf")
    savefig(pDV, png_path)
    savefig(pDV, pdf_path)
    @info "Saved voltage deviation panel" png = png_path pdf = pdf_path
    return pDV
end

function save_individual_voltage_plots(results, sys::System, buses::Vector{Int}, outdir::String, load_event_times::Vector{Float64}, load_event_summary::String;
                                       t_step::Float64 = 0.25)
    mkpath(outdir)
    for bus in buses
        out = voltage_magnitude_for_bus(results, sys, bus; t_step = t_step, for_plots = true)
        if out === nothing
            @warn "Skipping individual voltage plot: no voltage series" bus = bus
            continue
        end
        t, V, ΔV, V0, _, src = out
        pV = plot(t, V; title = @sprintf("Voltage magnitude at bus %d [V0=%.4f pu]\n%s", bus, V0, load_event_summary),
            xlabel = "Time [s]", ylabel = "V [pu]", lw = 2, color = bus_plot_color(bus),
            label = @sprintf("Bus %d (%s)", bus, string(src)), size = (1500, 700), titlefont = font(8),
            legend = :best, xlims = (0.0, TF_SIM), ylims = ABS_VOLTAGE_YLIM, yticks = ABS_VOLTAGE_YTICKS, widen = false)
        for (k, tev) in enumerate(load_event_times)
            vline!(pV, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
        end
        savefig(pV, joinpath(outdir, @sprintf("bus%02d_voltage_magnitude.pdf", bus)))
        savefig(pV, joinpath(outdir, @sprintf("bus%02d_voltage_magnitude.png", bus)))
        pDV = plot(t, ΔV; title = @sprintf("Voltage-magnitude deviation at bus %d [V0=%.4f pu]\n%s", bus, V0, load_event_summary),
            xlabel = "Time [s]", ylabel = "ΔV [pu]", lw = 2, color = bus_plot_color(bus),
            label = @sprintf("Bus %d (%s)", bus, string(src)), size = (1500, 700), titlefont = font(8),
            legend = :best, xlims = (0.0, TF_SIM), widen = false)
        hline!(pDV, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "zero reference")
        for (k, tev) in enumerate(load_event_times)
            vline!(pDV, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
        end
        savefig(pDV, joinpath(outdir, @sprintf("bus%02d_voltage_deviation.pdf", bus)))
        savefig(pDV, joinpath(outdir, @sprintf("bus%02d_voltage_deviation.png", bus)))
    end
    return true
end

# Steady-state statistics
function tail_stats(t::AbstractVector, y::AbstractVector; tail_seconds::Float64 = 2.0)
    n = length(t)
    n == 0 && return (NaN, NaN, NaN, 0, (NaN, NaN))
    t_end = t[end]
    idx0 = findfirst(>=(t_end - tail_seconds), t)
    idx0 = idx0 === nothing ? max(1, n - 1) : idx0
    ys = y[idx0:end]
    μ = mean(ys)
    σ = std(ys)
    y_last = y[end]
    return (y_last, μ, σ, length(ys), (t[idx0], t_end))
end

function print_final_steady_state_freq(results, sys::System; buses::Vector{Int}, tail_seconds::Float64 = 2.0)
    println("\n" * "="^80)
    println("FINAL / STEADY-STATE FREQUENCY DEVIATION ESTIMATE")
    println(@sprintf("Method: last sample + mean/std over last %.2f s", tail_seconds))
    println("="^80)
    println(@sprintf("%6s  %10s  %12s  %12s  %8s  %13s", "Bus", "Δf_last", "Δf_mean", "Δf_std", "Ntail", "TailWindow[s]"))
    println("-"^80)
    for bus in buses
        series = frequency_deviation_hz_at_bus(results, sys, bus)
        if series === nothing
            println(@sprintf("%6d  %s", bus, "NO_SERIES"))
            continue
        end
        t, df, lab, sym = series
        y_last, μ, σ, n_tail, (t0, t1) = tail_stats(t, df; tail_seconds = tail_seconds)
        println(@sprintf("%6d  %10.6f  %12.6f  %12.6f  %8d  [%6.2f, %6.2f]  (%s, %s)",
                         bus, y_last, μ, σ, n_tail, t0, t1, lab, string(sym)))
    end
    println("="^80 * "\n")
end

# Small-signal stability check
function report_small_signal(sim)
    ss = small_signal_analysis(sim)
    @info "Small-signal analysis" stable = ss.stable
    evs = try ss.eigenvalues catch; nothing end
    evs === nothing && return ss
    worst = nothing; worstζ = Inf
    println("\n" * "-"^60); println("LEAST-DAMPED OSCILLATORY MODES (ζ < 5%)")
    @printf("%10s %10s %8s %8s\n", "real", "imag", "f[Hz]", "ζ[%]"); println("-"^60)
    for λ in evs
        ω = imag(λ); ω == 0 && continue; abs(λ) < 1e-6 && continue
        ζ = -real(λ) / abs(λ)
        ζ < worstζ && (worstζ = ζ; worst = λ)
        ζ < 0.05 && @printf("%10.4f %10.4f %8.3f %8.2f\n", real(λ), ω, ω / (2π), ζ * 100)
    end
    println("-"^60)
    worst !== nothing && @printf("LEAST-DAMPED: f=%.3f Hz  ζ=%.2f%%  -> %s\n",
        imag(worst) / (2π), worstζ * 100, worstζ > 0 ? "stable" : "UNSTABLE")
    println("-"^60)
    unstable = sort([λ for λ in evs if real(λ) > 1e-3]; by = real, rev = true)
    if !isempty(unstable)
        println("\n" * "="^60); println("PARTICIPATION IN UNSTABLE MODES (driving states/devices)")
        println("="^60)
        P = nothing
        try P = summary_participation_factors(ss) catch e; @warn "participation unavailable" err = string(e) end
        if P !== nothing
            cols = names(P)
            for λr in unique(round.(real.(unstable); digits = 2))
                idx = findfirst(e -> isapprox(real(e), λr; atol = 5e-2) && imag(e) >= 0, evs)
                idx === nothing && continue
                e = evs[idx]
                @printf("\nMODE  real=%.2f  f=%.2f Hz:\n", real(e), imag(e) / (2π))
                idx + 1 <= length(cols) || continue
                sub = sort(DataFrame(state = P[!, 1], p = abs.(P[!, cols[idx + 1]])), :p, rev = true)
                for r in eachrow(first(sub, 8)); @printf("    %-42s %.3f\n", string(r.state), r.p); end
            end
        end
        println("="^60)
    end
    return ss
end

function make_virtual_inertia_block(; Ta::Float64 = 0.1, kd::Float64 = 0.0, kω::Float64 = 0.0, P_ref::Float64 = 1.0)
    try
        return VirtualInertia(Ta = Ta, kd = kd, kω = kω, P_ref = P_ref)
    catch
        return VirtualInertia(Ta = Ta, kd = kd, k_w = kω, P_ref = P_ref)
    end
end

#Attach SG dynamic model
function attach_synchronous_generator!(sys::System, g::Generator;
    H::Float64,
    D::Float64 = D_SG_DEFAULT,
    droop_R::Float64 = R_SG_PU,
    machine_params::Dict{Symbol,Float64},
    avr_params::Dict{Symbol,Float64}
)
    machine = RoundRotorQuadratic(
        R = machine_params[:Rs],
        Td0_p = machine_params[:Tdo_p],
        Td0_pp = machine_params[:Tdo_pp],
        Tq0_p = machine_params[:Tqo_p],
        Tq0_pp = machine_params[:Tqo_pp],
        Xd = machine_params[:Xd],
        Xq = machine_params[:Xq],
        Xd_p = machine_params[:Xd_p],
        Xq_p = machine_params[:Xq_p],
        Xd_pp = machine_params[:Xd_pp],
        Xl = machine_params[:Xl],
        Se = (0.0, 0.0),
    )

    shaft = SingleMass(H = H, D = D)

    avr = AVRTypeI(
        Ka = avr_params[:Ka],
        Ke = avr_params[:Ke],
        Kf = avr_params[:Kf],
        Ta = avr_params[:Ta],
        Te = avr_params[:Te],
        Tf = avr_params[:Tf],
        Tr = avr_params[:Tr],
        Va_lim = (-10.0, 10.0),
        Ae = 0.001,
        Be = 1.0,
    )

    tg = TGTypeI(
        R = droop_R,
        Ts = TG_TS,
        Tc = TG_EPS,
        T3 = TG_EPS,
        T4 = TG_EPS,
        T5 = TG_EPS,
        valve_position_limits = (min = -5.0, max = 5.0),
    )

    pss = PSSFixed(V_pss = 0.0)

    dyn = DynamicGenerator(
        name = PSY.get_name(g),
        ω_ref = 1.0,
        machine = machine,
        shaft = shaft,
        avr = avr,
        prime_mover = tg,
        pss = pss,
    )

    add_component!(sys, dyn, g)
    return dyn
end

#Attach GFM inverter dynamic model
function attach_grid_forming_inverter!(sys::System, g::Generator;
    droop_R::Float64 = R_GFM_PU,
    kq::Float64 = 0.05,
    Ta::Float64 = GFM_TA_DEFAULT,
    kd::Float64 = GFM_KD_DEFAULT,
    kω::Float64 = GFM_KW_DEFAULT,
    kpv::Float64 = 1.0,
    kiv::Float64 = 1000.0,
    kpc::Float64 = 3.0,
    kic::Float64 = 600.0,
    kffv::Float64 = 0.0,
    kffi::Float64 = 0.0,
    rv::Float64 = 0.0,
    lv::Float64 = 0.0,
    ωad::Float64 = 2000.0,
    kad::Float64 = 10.0,
    lf::Float64 = 0.08,
    rf::Float64 = 0.05,
    cf::Float64 = 0.074,
    lg::Float64 = 0.2,
    rg::Float64 = 0.04
)
    p_ref0, q_ref0 = get_generator_pref_qref(g)
    v_ref0 = get_bus_vref(sys, g)

    converter = AverageConverter(rated_voltage = 1.0, rated_current = 1.0)
    vi = make_virtual_inertia_block(Ta = Ta, kd = kd, kω = kω, P_ref = p_ref0)
    reactive = ReactivePowerDroop(kq = kq, ωf = 10.0, V_ref = v_ref0)
    outer = OuterControl(vi, reactive)

    inner = VoltageModeControl(
        kpv = kpv,
        kiv = kiv,
        kffv = kffv,
        rv = rv,
        lv = lv,
        kpc = kpc,
        kic = kic,
        kffi = kffi,
        ωad = ωad,
        kad = kad,
    )

    dc = FixedDCSource(voltage = 1200.0)
    pll = FixedFrequency()
    filt = LCLFilter(lf = lf, rf = rf, cf = cf, lg = lg, rg = rg)

    @info "GFM controller references initialized" bus = PSY.get_number(PSY.get_bus(g)) dev = PSY.get_name(g) P_ref = p_ref0 Q_ref = q_ref0 V_ref = v_ref0 droop_R = droop_R Ta = Ta kd = kd kω = kω kpv = kpv kiv = kiv kpc = kpc kic = kic rf = rf rg = rg

    dyn = DynamicInverter(
        name = PSY.get_name(g),
        ω_ref = 1.0,
        converter = converter,
        outer_control = outer,
        inner_control = inner,
        dc_source = dc,
        freq_estimator = pll,
        filter = filt,
    )

    add_component!(sys, dyn, g)
    return dyn
end
#Attach GFL models
function attach_grid_following_inverter!(sys::System, g::Generator;
    kp_p::Float64 = 0.8,
    ki_p::Float64 = 8.0,
    ωz_p::Float64 = 80.0,
    kp_q::Float64 = 0.8,
    ki_q::Float64 = 12.0,
    ωf_q::Float64 = 120.0,
    kω_droop::Float64 = GFL_KW_DEFAULT,
    kpc::Float64 = 1.2,
    kic::Float64 = 120.0,
    kffv::Float64 = 0.0,
    pll_ωlp::Float64 = 80.0,
    pll_kp::Float64 = getGflPllKp(PSY.get_number(PSY.get_bus(g))),
    pll_ki::Float64 = getGflPllKi(PSY.get_number(PSY.get_bus(g))),
    lf::Float64 = 0.08,
    rf::Float64 = 0.05,
    cf::Float64 = 0.074,
    lg::Float64 = 0.2,
    rg::Float64 = 0.04,
    use_safety_supervisor::Bool = false,
    filter_enabled::Bool = false,
    k_prox_override::Union{Nothing,Float64} = nothing
)
    bus_no = PSY.get_number(PSY.get_bus(g))
    converter = AverageConverter(rated_voltage = 1.0, rated_current = 1.0)

    p_ref0, q_ref0 = get_generator_pref_qref(g)
    v_ref0 = get_bus_vref(sys, g)

    if use_safety_supervisor
        isdefined(Main, :ActivePowerPISafetyFilter) ||
            error("gfl_cbf_outer_control.jl must be included before building the GFL model.")

        mbase = Float64(try PSY.get_base_power(g) catch; SYS_BASE_MVA end)
        sys_to_dev = SYS_BASE_MVA / mbase
        k_prox = isnothing(k_prox_override) ? gfl_cbf_k(bus_no) : Float64(k_prox_override)

        ap = ActivePowerPISafetyFilter(
            Kp_p = kp_p,
            Ki_p = ki_p,
            ωz = ωz_p,
            τd = ROCOF_TAU_D,
            α_lo = GFL_CBF_ALPHA_LO,
            α_hi = GFL_CBF_ALPHA_HI,
            k_prox = k_prox,
            ε = GFL_CBF_EPS,
            ω_min_dev = GFL_CBF_FLOOR_HZ / F_BASE,
            ω_max_dev = GFL_CBF_CEIL_HZ / F_BASE,
            filter_enabled = filter_enabled,
            sys_to_dev = sys_to_dev,
            z_min_sys = GFL_CBF_Z_MIN_SYS_PU,
            z_max_sys = GFL_CBF_Z_MAX_SYS_PU,
            I_rated = GFL_I_RATED,
            enforce_head = ENFORCE_HEADROOM,
            P_min = GFL_P_MIN,
            P_ref = p_ref0,
            ext = Dict{String,Any}("Kω" => kω_droop),
        )
    else
        ap = ActivePowerPI(Kp_p = kp_p, Ki_p = ki_p, ωz = ωz_p, P_ref = p_ref0)
        try
            PSY.get_ext(ap)["Kω"] = kω_droop
        catch
        end
    end

    rp = ReactivePowerPI(
        Kp_q = kp_q,
        Ki_q = ki_q,
        ωf = ωf_q,
        V_ref = v_ref0,
        Q_ref = q_ref0,
    )
    outer = OuterControl(ap, rp)
    inner = CurrentModeControl(kpc = kpc, kic = kic, kffv = kffv)
    dc = FixedDCSource(voltage = 1200.0)
    pll = KauraPLL(ω_lp = pll_ωlp, kp_pll = pll_kp, ki_pll = pll_ki)
    filt = LCLFilter(lf = lf, rf = rf, cf = cf, lg = lg, rg = rg)

    @info "GFL controller initialized" bus=bus_no P_ref=p_ref0 Q_ref=q_ref0 Kω=kω_droop pll_kp=pll_kp pll_ki=pll_ki supervisor=use_safety_supervisor filter=filter_enabled

    dyn = DynamicInverter(
        name = PSY.get_name(g),
        ω_ref = 1.0,
        converter = converter,
        outer_control = outer,
        inner_control = inner,
        dc_source = dc,
        freq_estimator = pll,
        filter = filt,
    )

    add_component!(sys, dyn, g)
    return dyn
end

#TGTypeI 
function get_tgtypei_states(results, sys::System, bus::Int)
    dev  = gen_at_bus(sys, bus)
    name = PSY.get_name(dev)
    out = Dict{Symbol,Any}()
    for s in (:x_g1, :x_g2, :x_g3)
        try
            t, x = get_state_series(results, (name, s))
            out[s] = (Float64.(t), Float64.(x))
        catch
        end
    end
    return out
end

function plot_tgtypei_response!(results, sys::System, bus::Int, plotdir::String, load_event_times::Vector{Float64}, load_event_summary::String)
    mkpath(plotdir)
    st = get_tgtypei_states(results, sys, bus)
    isempty(st) && (@warn "No TGTypeI states found for diagnostic plot" bus = bus; return nothing)
    p = plot(title = @sprintf("TGTypeI states at bus %d\n%s", bus, load_event_summary),
        xlabel = "Time [s]", ylabel = "state value [pu, device base]", size = (1500, 700),
        titlefont = font(8), legend = :best, grid = false, framestyle = :box, xlims = (0.0, TF_SIM), widen = false)
    for (sym, (t, x)) in st
        plot!(p, t, x; lw = 2, label = string(sym))
    end
    for (k, tev) in enumerate(load_event_times)
        vline!(p, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
    end
    savefig(p, joinpath(plotdir, @sprintf("tgtypei_bus%02d.pdf", bus)))
    savefig(p, joinpath(plotdir, @sprintf("tgtypei_bus%02d.png", bus)))
    @info "Saved TGTypeI diagnostic plot" bus = bus folder = plotdir
    return p
end

# Preview plots
function build_bus_plot_metadata_julia(buses::Vector{Int})
    meta = Dict{Int,NamedTuple}()
    for b in buses
        meta[b] = (model = model_label(b), color = bus_plot_color(b), label = @sprintf("Bus %d (%s)", b, model_label(b)))
    end
    return meta
end

function order_buses_for_plot_julia(buses::Vector{Int})
    sgs  = [b for b in buses if is_sg_bus(b)]
    gfms = [b for b in buses if is_gfm_bus(b)]
    gfls = [b for b in buses if is_gfl_bus(b)]
    return vcat(sort(sgs), sort(gfms), sort(gfls))
end

compact_axis_title_julia(s::String; maxlen::Int = 60) =
    length(s) <= maxlen ? s : (s[1:maxlen] * "…")

build_panel_title_julia(base::String, load_event_summary::String) =
    base * "\n" * load_event_summary

function style_panel_plot!(p)
    plot!(p; grid = false, framestyle = :box, xlims = (0.0, TF_SIM), widen = false,
          titlefont = font(9), guidefont = font(11), tickfont = font(10), legendfont = font(8))
    return p
end


#These plots use the per-bus CSV files saved during the simulation.
function _read_csv_safe(path::String)
    isfile(path) || return nothing
    try
        return CSV.read(path, DataFrame)
    catch e
        @warn "Could not read CSV for overlay" file = path err = string(e)
        return nothing
    end
end

#bus frequency deviation [Hz] 
function overlay_frequency_csvs(freqcsvdir::String, plotdir::String, buses::Vector{Int},
                                load_event_times::Vector{Float64}, load_event_summary::String;
                                fname_prefix::String = "ieee39_overlay_frequency")
    mkpath(plotdir)
    p = plot(title = "Bus frequency deviation overlay (from saved CSVs)\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "Δf [Hz]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 700),
        titlefont = font(11, :bold), guidefont = font(13, :bold), tickfont = font(11), legendfont = font(9))
    n_plotted = 0
    for bus in buses
        fpath = joinpath(freqcsvdir, @sprintf("freq_bus%02d_julia.csv", bus))
        df = _read_csv_safe(fpath)
        df === nothing && (@warn "Overlay skipping bus (no freq CSV)" bus = bus file = fpath; continue)
        ("t_s" in names(df) && "df_hz" in names(df)) || (@warn "freq CSV missing t_s/df_hz" bus = bus file = fpath; continue)
        plot!(p, df.t_s, df.df_hz; lw = 2.6, color = bus_plot_color(bus),
              label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
        n_plotted += 1
    end
    hline!(p, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    for (k, tev) in enumerate(load_event_times)
        vline!(p, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
    end
    savefig(p, joinpath(plotdir, fname_prefix * ".png"))
    savefig(p, joinpath(plotdir, fname_prefix * ".pdf"))
    @info "Saved frequency overlay from CSVs" folder = plotdir buses_plotted = n_plotted
    return p
end

#rotor-angle  and inverter electrical angle deviation [rad] read from rotor_angle_dev_busNN_julia.csv files.
function overlay_rotor_angle_csvs(csvdir::String, plotdir::String, buses::Vector{Int},
                                  load_event_times::Vector{Float64}, load_event_summary::String;
                                  fname_prefix::String = "ieee39_overlay_rotor_angle")
    mkpath(plotdir)
    p = plot(title = "Rotor-angle deviation overlay (from saved CSVs)\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "Δδ [rad]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 700),
        titlefont = font(11, :bold), guidefont = font(13, :bold), tickfont = font(11), legendfont = font(9))
    n_plotted = 0
    for bus in buses
        fpath = joinpath(csvdir, @sprintf("rotor_angle_dev_bus%02d_julia.csv", bus))
        df = _read_csv_safe(fpath)
        df === nothing && (@warn "Overlay skipping bus (no rotor-angle CSV)" bus = bus file = fpath; continue)
        ("t_s" in names(df) && "rotor_angle_dev_rad" in names(df)) || (@warn "rotor-angle CSV missing t_s/rotor_angle_dev_rad" bus = bus file = fpath; continue)
        plot!(p, df.t_s, df.rotor_angle_dev_rad; lw = 2.6, color = bus_plot_color(bus),
              label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
        n_plotted += 1
    end
    hline!(p, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    for (k, tev) in enumerate(load_event_times)
        vline!(p, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
    end
    savefig(p, joinpath(plotdir, fname_prefix * ".png"))
    savefig(p, joinpath(plotdir, fname_prefix * ".pdf"))
    @info "Saved rotor-angle overlay from CSVs" folder = plotdir buses_plotted = n_plotted
    return p
end

#SG mechanical-power deviation [pu] read from pg_busNN_julia.csv files.
function overlay_pg_csvs(pgcsvdir::String, plotdir::String, buses::Vector{Int},
                         load_event_times::Vector{Float64}, load_event_summary::String;
                         fname_prefix::String = "ieee39_overlay_pg")
    mkpath(plotdir)
    p = plot(title = "SG mechanical-power deviation overlay (from saved CSVs)\n" * load_event_summary,
        xlabel = "Time [s]", ylabel = "ΔP_m [pu, system base]", legend = :outertopright, grid = false,
        framestyle = :box, xlims = (0.0, TF_SIM), widen = false, dpi = 140, size = (1500, 700),
        titlefont = font(11, :bold), guidefont = font(13, :bold), tickfont = font(11), legendfont = font(9))
    n_plotted = 0
    for bus in buses
        fpath = joinpath(pgcsvdir, @sprintf("pg_bus%02d_julia.csv", bus))
        df = _read_csv_safe(fpath)
        df === nothing && (@warn "Overlay skipping bus (no pg CSV)" bus = bus file = fpath; continue)
        ("t_s" in names(df) && "pG_pu" in names(df)) || (@warn "pg CSV missing t_s/pG_pu" bus = bus file = fpath; continue)
        plot!(p, df.t_s, df.pG_pu; lw = 2.6, color = bus_plot_color(bus),
              label = @sprintf("Bus %d (%s)", bus, model_label(bus)))
        n_plotted += 1
    end
    hline!(p, [0.0]; lw = 0.8, ls = :dot, c = :gray, label = "")
    for (k, tev) in enumerate(load_event_times)
        vline!(p, [tev]; lw = 1, ls = :dash, c = :black, alpha = 0.35, label = (k == 1 ? "event time(s)" : ""))
    end
    savefig(p, joinpath(plotdir, fname_prefix * ".png"))
    savefig(p, joinpath(plotdir, fname_prefix * ".pdf"))
    @info "Saved SG mechanical-power overlay from CSVs" folder = plotdir buses_plotted = n_plotted
    return p
end

#MAIN
