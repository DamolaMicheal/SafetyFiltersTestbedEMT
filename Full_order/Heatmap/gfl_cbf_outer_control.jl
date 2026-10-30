# GFL active-power outer control with local PLL RoCoF estimation and dynamic safety filter.
#
# The corrective state z_i is stored on the 100 MVA system base. It is
# converted to the inverter device base only at
# the active-power-reference interface.

mutable struct ActivePowerPISafetyFilter <: PSY.ActivePowerControl
    Kp_p::Float64
    Ki_p::Float64
    ωz::Float64
    τd::Float64
    α_lo::Float64
    α_hi::Float64
    k_prox::Float64
    ε::Float64
    ω_min_dev::Float64
    ω_max_dev::Float64
    filter_enabled::Bool
    sys_to_dev::Float64
    z_min_sys::Float64
    z_max_sys::Float64
    I_rated::Float64
    enforce_head::Bool
    P_min::Float64
    P_ref::Float64
    ext::Dict{String,Any}
    states::Vector{Symbol}
    n_states::Int
end

function ActivePowerPISafetyFilter(;
    Kp_p,
    Ki_p,
    ωz,
    τd = 0.02,
    α_lo = 2.0,
    α_hi = 2.0,
    k_prox = 0.01,
    ε = 0.02,
    ω_min_dev = -0.05 / 60.0,
    ω_max_dev = +0.05 / 60.0,
    filter_enabled = true,
    sys_to_dev = 1.0,
    z_min_sys = -Inf,
    z_max_sys = +Inf,
    I_rated = 1.0,
    enforce_head = false,
    P_min = 0.0,
    P_ref = 1.0,
    ext = Dict{String,Any}(),
)
    τd > 0.0 || error("τd must be positive.")
    ε > 0.0 || error("ε must be positive.")
    k_prox > 0.0 || error("k_prox must be positive.")
    ω_min_dev < 0.0 < ω_max_dev ||
        error("The CBF band must satisfy ω_min_dev < 0 < ω_max_dev.")
    z_min_sys <= z_max_sys || error("z_min_sys must not exceed z_max_sys.")

    return ActivePowerPISafetyFilter(
        Float64(Kp_p), Float64(Ki_p), Float64(ωz), Float64(τd),
        Float64(α_lo), Float64(α_hi), Float64(k_prox), Float64(ε),
        Float64(ω_min_dev), Float64(ω_max_dev), Bool(filter_enabled),
        Float64(sys_to_dev), Float64(z_min_sys), Float64(z_max_sys),
        Float64(I_rated), Bool(enforce_head), Float64(P_min), Float64(P_ref),
        ext,
        [:σp_oc, :p_oc, :chi_rocof, :z_cbf],
        4,
    )
end

PSY.get_Kp_p(x::ActivePowerPISafetyFilter) = x.Kp_p
PSY.get_Ki_p(x::ActivePowerPISafetyFilter) = x.Ki_p
PSY.get_ωz(x::ActivePowerPISafetyFilter) = x.ωz
PSY.get_P_ref(x::ActivePowerPISafetyFilter) = x.P_ref
PSY.get_ext(x::ActivePowerPISafetyFilter) = x.ext
PSY.get_states(x::ActivePowerPISafetyFilter) = x.states
PSY.get_n_states(x::ActivePowerPISafetyFilter) = x.n_states
PSY.set_P_ref!(x::ActivePowerPISafetyFilter, val) = (x.P_ref = Float64(val))

get_rocof_tau(x::ActivePowerPISafetyFilter) = x.τd
get_cbf_alpha_lo(x::ActivePowerPISafetyFilter) = x.α_lo
get_cbf_alpha_hi(x::ActivePowerPISafetyFilter) = x.α_hi
get_cbf_prox_k(x::ActivePowerPISafetyFilter) = x.k_prox
get_cbf_epsilon(x::ActivePowerPISafetyFilter) = x.ε
get_cbf_omega_min_dev(x::ActivePowerPISafetyFilter) = x.ω_min_dev
get_cbf_omega_max_dev(x::ActivePowerPISafetyFilter) = x.ω_max_dev
get_filter_enabled(x::ActivePowerPISafetyFilter) = x.filter_enabled
get_cbf_sys_to_dev(x::ActivePowerPISafetyFilter) = x.sys_to_dev
get_cbf_z_min_sys(x::ActivePowerPISafetyFilter) = x.z_min_sys
get_cbf_z_max_sys(x::ActivePowerPISafetyFilter) = x.z_max_sys
get_cbf_Irated(x::ActivePowerPISafetyFilter) = x.I_rated
get_cbf_enforce_headroom(x::ActivePowerPISafetyFilter) = x.enforce_head
get_cbf_Pmin(x::ActivePowerPISafetyFilter) = x.P_min

@inline function project_zero_to_interval(a, b)
    b >= a - 1e-12 ||
        error("The local CBF proximal interval is not ordered: a=$a, b=$b.")
    return a > 0.0 ? a : (b < 0.0 ? b : 0.0)
end

@inline function local_safety_target(
    ω_dev,
    rocof_hat,
    z_sys,
    control::ActivePowerPISafetyFilter,
)
    a = z_sys + control.k_prox * (
        -control.α_lo * (ω_dev - control.ω_min_dev) - rocof_hat
    )
    b = z_sys + control.k_prox * (
        +control.α_hi * (control.ω_max_dev - ω_dev) - rocof_hat
    )
    return project_zero_to_interval(a, b), a, b
end

function PSID.mdl_outer_ode!(
    device_states::AbstractArray,
    output_ode::AbstractArray,
    inner_vars::AbstractArray,
    ω_sys,
    dynamic_device::PSID.DynamicWrapper{
        PSY.DynamicInverter{
            C,
            PSY.OuterControl{ActivePowerPISafetyFilter,PSY.ReactivePowerPI},
            IC, DC, P, F, L,
        },
    },
    h, t,
) where {
    C <: PSY.Converter,
    IC <: PSY.InnerControl,
    DC <: PSY.DCSource,
    P <: PSY.FrequencyEstimator,
    F <: PSY.Filter,
    L <: Union{Nothing,PSY.OutputCurrentLimiter},
}
    external_ix = PSID.get_input_port_ix(
        dynamic_device,
        PSY.OuterControl{ActivePowerPISafetyFilter,PSY.ReactivePowerPI},
    )
    Vr_filter = device_states[external_ix[1]]
    Vi_filter = device_states[external_ix[2]]
    Ir_filter = device_states[external_ix[3]]
    Ii_filter = device_states[external_ix[4]]

    θ_pll = inner_vars[PSID.θ_freq_estimator_var]
    ω_pll = inner_vars[PSID.ω_freq_estimator_var]

    outer_control = PSY.get_outer_control(dynamic_device)
    active = PSY.get_active_power_control(outer_control)
    reactive = PSY.get_reactive_power_control(outer_control)

    Kp_p = PSY.get_Kp_p(active)
    Ki_p = PSY.get_Ki_p(active)
    ωz = PSY.get_ωz(active)
    Kω = get(PSY.get_ext(active), "Kω", 0.0)

    Kp_q = PSY.get_Kp_q(reactive)
    Ki_q = PSY.get_Ki_q(reactive)
    ωf = PSY.get_ωf(reactive)

    p_ref = PSID.get_P_ref(dynamic_device)
    q_ref = PSID.get_Q_ref(dynamic_device)

    local_ix = PSID.get_local_state_ix(
        dynamic_device,
        PSY.OuterControl{ActivePowerPISafetyFilter,PSY.ReactivePowerPI},
    )
    state = @view device_states[local_ix]
    σp_oc = state[1]
    p_oc = state[2]
    chi = state[3]
    z_sys = state[4]
    σq_oc = state[5]
    q_oc = state[6]

    p_elec_out = Ir_filter * Vr_filter + Ii_filter * Vi_filter
    q_elec_out = -Ii_filter * Vr_filter + Ir_filter * Vi_filter

    # The Kaura PLL returns absolute per-unit frequency. The safety filter uses
    # the local deviation from nominal and the local dirty-derivative RoCoF,
    # matching the measurement-based realization in the paper.
    ω_dev = ω_pll - 1.0
    rocof_hat = (ω_dev - chi) / active.τd
    dchi_dt = (ω_dev - chi) / active.τd

    # If the nominal measured CBF inequalities are satisfied while z_i=0,
    # then a_i <= 0 <= b_i, the projection is zero, and z_i remains dormant.
    # The filter is anticipatory: it can act before the frequency itself reaches
    # a boundary if the measured RoCoF makes a CBF derivative condition active.
    target = active.filter_enabled ?
        local_safety_target(ω_dev, rocof_hat, z_sys, active)[1] :
        zero(z_sys)
    dz_dt = active.filter_enabled ?
        (-z_sys + target) / active.ε :
        -z_sys / active.ε

    z_cmd_sys = active.filter_enabled ?
        clamp(z_sys, active.z_min_sys, active.z_max_sys) :
        0.0
    z_cmd_dev = z_cmd_sys * active.sys_to_dev

    # Optional physical actuation enforcement. It is disabled here so the
    # frequency-safety correction is not altered by an additional hard limit.
    if active.enforce_head
        V_t = sqrt(Vr_filter^2 + Vi_filter^2)
        i_mag2 = Ir_filter^2 + Ii_filter^2
        i_q2 = max(i_mag2 - (p_elec_out / max(V_t, 1e-6))^2, 0.0)
        P_head = V_t * sqrt(max(active.I_rated^2 - i_q2, 0.0)) - p_elec_out
        P_shed = p_elec_out - active.P_min
        z_cmd_dev = clamp(z_cmd_dev, -max(P_shed, 0.0), max(P_head, 0.0))
    end

    p_ref_eff = p_ref - Kω * ω_dev + z_cmd_dev

    # PSID GFL dq convention:
    #   active-power PI -> d-axis current reference,
    #   reactive-power PI -> negative q-axis current reference.
    Id_cmd, dσp_dt = PSID.pi_block(
        p_ref_eff - p_oc, σp_oc, Kp_p, Ki_p
    )
    _, dpoc_dt = PSID.low_pass(
        p_elec_out, p_oc, 1.0, 1.0 / ωz
    )

    Iq_pi, dσq_dt = PSID.pi_block(
        q_ref - q_oc, σq_oc, Kp_q, Ki_q
    )
    Iq_cmd = -Iq_pi
    _, dqoc_dt = PSID.low_pass(
        q_elec_out, q_oc, 1.0, 1.0 / ωf
    )

    if active.enforce_head
        iq_budget = sqrt(max(active.I_rated^2 - Id_cmd^2, 0.0))
        Iq_cmd = clamp(Iq_cmd, -iq_budget, iq_budget)
    end

    output_ode[local_ix[1]] = dσp_dt
    output_ode[local_ix[2]] = dpoc_dt
    output_ode[local_ix[3]] = dchi_dt
    output_ode[local_ix[4]] = dz_dt
    output_ode[local_ix[5]] = dσq_dt
    output_ode[local_ix[6]] = dqoc_dt

    inner_vars[PSID.θ_oc_var] = θ_pll
    inner_vars[PSID.ω_oc_var] = ω_pll
    inner_vars[PSID.Id_oc_var] = Id_cmd
    inner_vars[PSID.Iq_oc_var] = Iq_cmd
    return
end

function PSID.initialize_outer!(
    device_states,
    static::PSY.StaticInjection,
    dynamic_device::PSID.DynamicWrapper{
        PSY.DynamicInverter{
            C,
            PSY.OuterControl{ActivePowerPISafetyFilter,PSY.ReactivePowerPI},
            IC, DC, P, F, L,
        },
    },
    inner_vars::AbstractVector,
) where {
    C <: PSY.Converter,
    IC <: PSY.InnerControl,
    DC <: PSY.DCSource,
    P <: PSY.FrequencyEstimator,
    F <: PSY.Filter,
    L <: Union{Nothing,PSY.OutputCurrentLimiter},
}
    external_ix = PSID.get_input_port_ix(
        dynamic_device,
        PSY.OuterControl{ActivePowerPISafetyFilter,PSY.ReactivePowerPI},
    )
    Vr_filter = device_states[external_ix[1]]
    Vi_filter = device_states[external_ix[2]]
    Ir_filter = device_states[external_ix[3]]
    Ii_filter = device_states[external_ix[4]]

    # Current PSID releases expose converter current through inverter inner_vars.
    # Some earlier branches also appended these currents to the outer-control
    # input-port list. Support both layouts so the custom controller remains
    # compatible with the user's local development checkout.
    Ir_cnv = if length(external_ix) >= 6
        device_states[external_ix[5]]
    else
        inner_vars[PSID.Ir_cnv_var]
    end
    Ii_cnv = if length(external_ix) >= 6
        device_states[external_ix[6]]
    else
        inner_vars[PSID.Ii_cnv_var]
    end

    θ0_oc = inner_vars[PSID.θ_freq_estimator_var]
    I_dq_cnv = PSID.ri_dq(θ0_oc + pi / 2) * [Ir_cnv; Ii_cnv]
    p_elec_out = Ir_filter * Vr_filter + Ii_filter * Vi_filter
    q_elec_out = -Ii_filter * Vr_filter + Ir_filter * Vi_filter

    outer_control = PSY.get_outer_control(dynamic_device)
    active = PSY.get_active_power_control(outer_control)
    reactive = PSY.get_reactive_power_control(outer_control)
    Ki_p = PSY.get_Ki_p(active)
    Ki_q = PSY.get_Ki_q(reactive)

    inner_vars[PSID.P_ES_var] = p_elec_out

    outer_ix = PSID.get_local_state_ix(
        dynamic_device,
        PSY.OuterControl{ActivePowerPISafetyFilter,PSY.ReactivePowerPI},
    )
    state = @view device_states[outer_ix]

    # Corrected PSID initialization consistent with the dq equations above.
    state[1] = I_dq_cnv[PSID.d] / Ki_p
    state[2] = p_elec_out

    ω0 = try
        Float64(inner_vars[PSID.ω_freq_estimator_var])
    catch
        Float64(PSID.get_ω_ref(dynamic_device))
    end
    if !isfinite(ω0) || abs(ω0 - 1.0) > 0.2
        ω0 = Float64(PSID.get_ω_ref(dynamic_device))
    end
    state[3] = ω0 - 1.0
    state[4] = 0.0

    state[5] = -I_dq_cnv[PSID.q] / Ki_q
    state[6] = q_elec_out

    inner_vars[PSID.θ_oc_var] = θ0_oc
    inner_vars[PSID.ω_oc_var] = ω0
    inner_vars[PSID.Id_oc_var] = I_dq_cnv[PSID.d]
    inner_vars[PSID.Iq_oc_var] = I_dq_cnv[PSID.q]

    # Match the stock PSID steady-state initialization. The dynamic-device
    # references are set to the initialized electrical powers so that
    # dσp/dt = dσq/dt = 0 exactly at t = 0. This does not add a CBF offset:
    # ω_dev(0)=0 and z_i(0)=0, hence p_ref_eff(0)=p_ref(0).
    try
        PSID.set_P_ref(dynamic_device, p_elec_out)
    catch
    end
    try
        PSY.set_P_ref!(active, p_elec_out)
    catch
    end
    try
        PSID.set_Q_ref(dynamic_device, q_elec_out)
    catch
    end
    try
        PSY.set_Q_ref!(reactive, q_elec_out)
    catch
    end

    return
end
