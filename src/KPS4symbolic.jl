# Copyright (c) 2020, 2021, 2022, 2024 Uwe Fechner
# SPDX-License-Identifier: MIT

#= Model of a kite-power system in implicit form: residual = f(y, yd)

This model implements a 3D mass-spring system with reel-out. It uses six tether segments (the number can be
configured in the file data/settings.yaml). The kite is modelled using 4 point masses and 3 aerodynamic 
surfaces. The spring constant and the damping decrease with the segment length. The aerodynamic kite forces
are acting on three of the four kite point masses. 

Four point kite model, included from KiteModels.jl.

Scientific background: http://arxiv.org/abs/1406.6218 =#

# Array of connections of bridlepoints.
# First point, second point, unstressed length.
using KiteModels
using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Parameters, DocStringExtensions, Rotations
using ModelingToolkit: Symbolics, @register_symbolic
using OrdinaryDiffEqCore
using DataInterpolations
using ModelingToolkit: t_nounits as t, D_nounits as D
using KiteUtils

@with_kw struct SP_sym
    p1::Int = 0         # number of the first point
    p2::Int = 0         # number of the second point
    length::Num = 0   # current unstressed spring length
    k::Num = 0 # spring constant [N/m]
    c::Num = 0 # damping coefficent [Ns/m]
    kite_spring::Bool = false
end

const DRAG_CORR = 0.93       # correction of the drag for the 4-point model
const PRE_STRESS = 0.9998
const G_EARTH = 9.81

"""
    mutable struct KPS4{S, T, P, Q, SP} <: AbstractKiteModel

State of the kite power system, using a 4 point kite model. Parameters:
- S: Scalar type, e.g. SimFloat
  In the documentation mentioned as Any, but when used in this module it is always SimFloat and not Any.
- T: Vector type, e.g. MVector{3, SimFloat}
- P: number of points of the system, segments+1
- Q: number of springs in the system, P-1
- SP: struct type, describing a spring
Normally a user of this package will not have to access any of the members of this type directly,
use the input and output functions instead.

$(TYPEDFIELDS)
"""
@with_kw mutable struct KPS4Symbolic{S} <: KiteUtils.AbstractKiteModel
    "Reference to the settings struct"
    set::Settings
    "Reference to the KCU model (Kite Control Unit as implemented in the package KitePodModels"
    kcu::KCU
    "Reference to the atmospheric model as implemented in the package AtmosphericModels"
    am::AtmosphericModel
    "Reference to winch model as implemented in the package WinchModels"
    wm::AbstractWinchModel
    sys::Union{ModelingToolkit.ODESystem,Nothing} = nothing
    t_0::Float64 = 0.0
    iter::Int64 = 0
    prob::Union{OrdinaryDiffEqCore.ODEProblem,Nothing} = nothing
    integrator::Union{OrdinaryDiffEqCore.ODEIntegrator,Nothing} = nothing
    get_state::Function = () -> nothing

    stiffness_factor::S = 1
    bridle_factor::S = 1 # TODO: set

    set_torque::S = 0
    set_depower::S = 0
    set_steering::S = 0
    downwind_dir::S = 0
    wind_spd_gnd::S = 6
end

function n_kite_points(s::KPS4Symbolic)
    return 4
end
function n_points(s::KPS4Symbolic)
    return n_kite_points(s) + s.set.segments + 1  # Kite points: 4, tether points (incl origin): s.set.segments. kcu: 1
end
function kcu_index(s::KPS4Symbolic)
    return s.set.segments + 1
end
function kite_index(s::KPS4Symbolic)
    return kcu_index(s) + 2
end

function KPS4Symbolic(kcu::KCU)
    if kcu.set.winch_model == "AsyncMachine"
        wm = AsyncMachine(kcu.set)
    elseif kcu.set.winch_model == "TorqueControlledMachine"
        wm = TorqueControlledMachine(kcu.set)
    end
    # wm.last_set_speed = kcu.set.v_reel_out
    am = AtmosphericModel(set=kcu.set)
    s = KPS4Symbolic(set=kcu.set, am=am, kcu=kcu, wm=wm)
    # clear!(s)  # TODO: add back in
    return s
end

function init_masses(s::KPS4Symbolic, l_0)
    MASS_FACTOR = 1.0

    if s.set.version == 1
        # for compatibility with the python code and paper
        mass_per_meter = 0.011
    else
        mass_per_meter = s.set.rho_tether * π * (s.set.d_tether / 2000.0)^2
    end

    masses = [Num(0) for _ in 1:n_points(s)]

    for i in 1:s.set.segments
        masses[i] += 0.5 * mass_per_meter * l_0
        masses[i+1] += 0.5 * mass_per_meter * l_0
    end
    masses[kcu_index(s)] += s.set.kcu_mass * MASS_FACTOR

    k1 = s.set.rel_nose_mass
    k2 = s.set.rel_top_mass * (1.0 - s.set.rel_nose_mass)
    k3 = 0.5 * (1.0 - s.set.rel_top_mass) * (1.0 - s.set.rel_nose_mass)

    masses[kite_index(s)-1] += k1 * s.set.mass * MASS_FACTOR
    masses[kite_index(s)] += k2 * s.set.mass * MASS_FACTOR
    masses[kite_index(s)+1] += k3 * s.set.mass * MASS_FACTOR
    masses[kite_index(s)+2] += k3 * s.set.mass * MASS_FACTOR
    masses
end

function get_springs(s::KPS4Symbolic, l_0, initial_kite_positions)
    kite_springs = [
        [1, 2],  # s1, KCU, A
        [4, 2],  # s2, C,   A                        
        [4, 5],  # s3, C,   D
        [3, 4],  # s4, B,   C
        [5, 1],  # s5, D,   KCU
        [4, 1],  # s6, C,   KCU
        [3, 5],  # s7, B,   D
        [5, 2],  # s8, D,   A
        [2, 3]   # s9, A,   B
    ]

    springs = [SP_sym(0, 0, 0, 0, 0, false) for _ in 1:s.set.segments+length(kite_springs)]
    initial_lengths = zeros(s.set.segments + length(kite_springs))

    particles = KiteUtils.get_particles(s.set.height_k, s.set.h_bridle, s.set.width, s.set.m_k)[2:end]  # Do not include the origin to make indexing easier
    for i in 1:s.set.segments
        k_tether = s.set.c_spring / l_0
        c_tether = s.set.damping / l_0
        springs[i] = SP_sym(i, i + 1, l_0, k_tether, c_tether, false)
        initial_lengths[i] = norm(initial_kite_positions[i+1, :] - initial_kite_positions[i, :])
    end

    for (i, (p1, p2)) in enumerate(kite_springs)
        segment_length = norm(particles[p2] - particles[p1])
        l_0_bridle = segment_length * PRE_STRESS
        k_bridle = s.set.e_tether * (s.set.d_line / 2000.0)^2 * pi / l_0_bridle
        c_bridle = s.set.damping / l_0_bridle
        springs[i+s.set.segments] = SP_sym(p1 + s.set.segments, p2 + s.set.segments, l_0_bridle, k_bridle, c_bridle, true)
        initial_lengths[i+s.set.segments] = segment_length * PRE_STRESS
    end
    springs, initial_lengths
end

function winch_calculations(v_reel_out, f_winch, set_speed, set_torque)
    if s.wm isa AsyncMachine
        throw("AsyncMachine currently not supported")
    else
        a_tether = calc_acceleration(s.wm, v_reel_out, f_winch; set_speed=nothing, set_torque=set_torque, use_brake=false)
    end
    return a_tether
end

function scalar(eqs::Vector{Equation})
    return reduce(vcat, Symbolics.scalarize.(eqs))
end

function get_symbolic_3d_vec(var, ind)
    return [var[ind, 1], var[ind, 2], var[ind, 3]]
end
function get_symbolic_3d_vec(var)
    return [var[1], var[2], var[3]]
end
function vec_3d(var, ind)
    return get_symbolic_3d_vec(var, ind)
end
function vec_3d(var)
    return get_symbolic_3d_vec(var)
end

function calc_azimuth(pos, v_wind_gnd)
    dir_pos = pos / norm(pos)
    dir_wind = v_wind_gnd / norm(v_wind_gnd)

    az_wind = atan(dir_wind[2], dir_wind[1])
    az_pos = atan(dir_pos[2], dir_pos[1])
    return az_pos - az_wind
end

function spline_derivative(spline::LinearInterpolation, t_s)
    return LinearInterpolation(map(x -> DataInterpolations.derivative(spline, x), t_s), t_s, extrapolation=spline.extrapolation_left)
end

function calc_aero_coeff(spline::LinearInterpolation, spline_derivative::LinearInterpolation, alpha)
    return spline(alpha)
end

@register_symbolic calc_aero_coeff(spline::LinearInterpolation, spline_derivative::LinearInterpolation, alpha)
Symbolics.derivative(::typeof(calc_aero_coeff), args::NTuple{3,Any}, ::Val{3}) = args[2](args[3])


function construct_sys(s::KPS4Symbolic)
    n = n_points(s)

    if s.wm isa AsyncMachine
        POS0, VEL0, ACC0, L_TETHER0, V_TETHER0, A_TETHER0, AOA0, SPEED_WIND_GND0, DOWNWIND_DIR0, SET_TORQUE, POS_KITE0 = calc_initial_guess(s)
    else
        POS0, VEL0, ACC0, L_TETHER0, V_TETHER0, A_TETHER0, AOA0, SPEED_WIND_GND0, DOWNWIND_DIR0, SET_SPEED, POS_KITE0 = calc_initial_guess(s)
    end

    # State variables
    @variables pos(t)[1:n, 1:3] = POS0
    @variables vel(t)[1:n, 1:3] = VEL0
    @variables acc(t)[1:n, 1:3] = ACC0
    @variables l_tether(t) = L_TETHER0
    @variables v_tether(t) = V_TETHER0
    @variables a_tether(t) = A_TETHER0

    # Input variables
    @variables speed_wind_gnd(t) = SPEED_WIND_GND0 [input = true]
    @variables downwind_direction(t) = DOWNWIND_DIR0 [input = true]
    @variables set_speed(t) = 0 [input = true]
    @variables set_torque(t) = 0 [input = true]
    @variables rel_depower(t) = s.set.depower_zero / 100 [input = true]
    @variables rel_steering(t) = 0 [input = true]

    # Intermediary variables
    L_0 = l_tether / s.set.segments

    masses = init_masses(s, L_0)
    springs, L_SPRINGS0 = get_springs(s, L_0, POS0)
    forces = zeros(Num, n, 3)

    @variables v_wind_gnd(t)[1:3] = [cos(DOWNWIND_DIR0), sin(DOWNWIND_DIR0), 0] * SPEED_WIND_GND0

    m = length(springs)
    @variables sp_height(t)[1:m]
    @variables sp_length(t)[1:m] = L_SPRINGS0
    @variables sp_uv_dir(t)[1:m, 1:3]

    @variables sp_speed(t)[1:m]
    @variables sp_k(t)[1:m]
    @variables sp_c(t)[1:m]
    @variables sp_force(t)[1:m, 1:3]

    @variables sp_air_dens(t)[1:m]
    @variables sp_avg_vel(t)[1:m, 1:3]
    @variables sp_wind_vel(t)[1:m, 1:3]
    @variables sp_app_vel(t)[1:m, 1:3]
    @variables sp_perp_app_vel(t)[1:m, 1:3]
    @variables sp_drag(t)[1:m, 1:3]

    @variables kcu_app_vel(t)[1:3]
    @variables kcu_perp_app_vel(t)[1:3]
    @variables kcu_drag(t)[1:3]

    @variables e_x(t)[1:3], e_y(t)[1:3], e_z(t)[1:3]

    @variables kite_wind_vel(t)[1:3]
    @variables kite_air_dens(t)

    n_k = n_kite_points(s)
    @variables kite_app_vel(t)[1:n_k, 1:3]
    @variables kite_perp_app_vel(t)[1:n_k, 1:3]

    @variables kite_clipped_wind_angle(t)[1:n_k]
    @variables kite_angle_of_attack(t)[1:n_k] = AOA0

    @variables kite_cl(t)[1:n_k], kite_cd(t)[1:n_k]

    @variables kite_lift_direction(t)[1:n_k, 1:3]
    @variables kite_lift(t)[1:n_k, 1:3], kite_drag(t)[1:n_k, 1:3]

    @variables kite_pitch(t)
    @variables kite_pitching_moment_force(t)[1:3], kite_steering_moment_force(t)[1:3]

    # Outputs
    @variables pos_kite(t)[1:3] [output = true]
    @variables elevation_kite(t) [output = true]
    @variables azimuth_kite(t) [output = true]
    @variables range_kite(t) [output = true]
    @variables vel_kite(t)[1:3] [output = true]
    @variables radial_vel(t) [output = true]
    @variables tangential_vel(t) [output = true]
    @variables course_angle(t) [output = true]
    @variables orient_euler(t)[1:3] [output = true]
    @variables orient_euler_rate(t)[1:3] [output = true]
    @variables tether_length(t) [output = true]
    @variables reel_speed(t) [output = true]
    @variables winch_force(t) [output = true]
    @variables c_l(t) [output = true]

    # Equations for spings (equivalent to loop! + inner_loop! + calc_particle_forces)
    eq_sps = Vector{Symbolics.Equation}([])
    for (i, sp) in enumerate(springs)
        p1 = sp.p1
        p2 = sp.p2

        println("spring: $i ($p1 -> $p2)")

        l_0 = sp.length # Unstressed length
        segment = vec_3d(pos, p1) - vec_3d(pos, p2)
        rel_vel = vec_3d(vel, p1) - vec_3d(vel, p2)

        c = sp.c
        c1 = s.set.rel_damping * c

        k = sp.k * s.stiffness_factor
        k1 = k * s.set.rel_compr_stiffness
        k2 = k * 0.1

        if s.set.version == 1
            area = sp_length[i] * s.set.d_tether * 0.001
        elseif sp.kite_spring
            area = sp_length[i] * s.set.d_line * 0.001 * s.bridle_factor # 6.0 = A_real/A_simulated
        else
            area = sp_length[i] * s.set.d_tether * 0.001
        end

        eq_sp = [
            sp_height[i] ~ 0.5 * (pos[p1, 3] + pos[p2, 3]),
            sp_length[i] ~ norm(segment),
            vec_3d(sp_uv_dir, i) ~ segment / sp_length[i],

            # Spring force
            sp_speed[i] ~ vec_3d(sp_uv_dir, i) ⋅ rel_vel,
            sp_k[i] ~ ifelse((sp_length[i] - l_0) > 0.0, k, sp.kite_spring ? k1 : k2),
            sp_c[i] ~ ifelse((sp_length[i] - l_0) > 0.0, sp.kite_spring ? c1 : c, c),
            vec_3d(sp_force, i) ~ (k * (sp_length[i] - l_0) + (c * sp_speed[i])) * vec_3d(sp_uv_dir, i),

            # Drag force
            sp_air_dens[i] ~ calc_rho(s.am, sp_height[i]),
            vec_3d(sp_avg_vel, i) ~ 0.5 * (vec_3d(vel, p1) + vec_3d(vel, p2)),
            vec_3d(sp_wind_vel, i) ~ calc_wind_factor(s.am, sp_height[i]) * v_wind_gnd,
            vec_3d(sp_app_vel, i) ~ vec_3d(sp_wind_vel, i) - vec_3d(sp_avg_vel, i),
            vec_3d(sp_perp_app_vel, i) ~ vec_3d(sp_app_vel, i) - vec_3d(sp_app_vel, i) ⋅ vec_3d(sp_uv_dir, i) * vec_3d(sp_uv_dir, i),
            vec_3d(sp_drag, i) ~ (-0.5 * sp_air_dens[i] * s.set.cd_tether * norm(vec_3d(sp_perp_app_vel, i)) * area) * vec_3d(sp_perp_app_vel, i)
        ]

        eq_sps = vcat(eq_sps, scalar(eq_sp))

        @inbounds forces[p1, :] += 0.5 * vec_3d(sp_drag, i) + vec_3d(sp_force, i)
        @inbounds forces[p2, :] += 0.5 * vec_3d(sp_drag, i) - vec_3d(sp_force, i)
    end

    # KCU drag
    i = s.set.segments
    sp = springs[s.set.segments]
    p2 = sp.p2

    kcu_area = π * (s.set.kcu_diameter / 2)^2

    eq_kcu_drag = scalar([
        kcu_app_vel ~ vec_3d(sp_wind_vel, i) - vec_3d(vel, p2)
        kcu_perp_app_vel ~ kcu_app_vel - kcu_app_vel ⋅ vec_3d(sp_uv_dir, i) * vec_3d(sp_uv_dir, i)
        kcu_drag ~ (-0.5 * sp_air_dens[i] * s.set.cd_kcu * norm(vec_3d(kcu_perp_app_vel)) * kcu_area) * kcu_perp_app_vel
    ])


    @inbounds forces[sp.p2, :] += 0.5 * kcu_drag

    # Reference Frames
    j_A = 1
    j_B = 2
    j_C = 3
    j_D = 4

    i_B = kite_index(s)
    i_A = i_B + j_A - j_B
    i_C = i_B + j_C - j_B
    i_D = i_B + j_D - j_B

    pos_B, pos_C, pos_D = vec_3d(pos, i_B), vec_3d(pos, i_C), vec_3d(pos, i_D)
    pos_centre = 0.5 * (pos_C + pos_D)
    delta = pos_B - pos_centre

    eq_refrence_frame = scalar([
        e_x ~ e_y × e_z,
        e_y ~ (pos_C - pos_D) / norm(pos_C - pos_D),
        e_z ~ -delta / norm(delta)
    ])

    # Aerodynamic
    # TODO: side_slip, side_cl, lift_force, drag_force
    eq_aero_base = scalar([
        kite_wind_vel ~ calc_wind_factor(s.am, pos[i_B, 3]) * v_wind_gnd,
        kite_air_dens ~ calc_rho(s.am, pos[i_B, 3]),
    ])

    eq_aero_vel = vcat(
        vec(scalar([kite_app_vel ~ -vel[i_A:end, :] .+ kite_wind_vel'])),
        scalar([
            vec_3d(kite_perp_app_vel, j_A) ~ vec_3d(kite_app_vel, j_A) - (vec_3d(kite_app_vel, j_A) ⋅ e_y) * e_y,
            vec_3d(kite_perp_app_vel, j_B) ~ vec_3d(kite_app_vel, j_B) - (vec_3d(kite_app_vel, j_B) ⋅ e_y) * e_y,
            vec_3d(kite_perp_app_vel, j_C) ~ vec_3d(kite_app_vel, j_C) - (vec_3d(kite_app_vel, j_C) ⋅ e_z) * e_z,
            vec_3d(kite_perp_app_vel, j_D) ~ vec_3d(kite_app_vel, j_D) - (vec_3d(kite_app_vel, j_D) ⋅ e_z) * e_z,
        ])
    )

    ks = deg2rad(s.set.max_steering)
    alpha_depower = calc_alpha_depower(s.kcu, rel_depower)

    eq_aero_aoa = scalar([
        kite_clipped_wind_angle[j_A] ~ 0,
        kite_clipped_wind_angle[j_B] ~ min(max(vec_3d(kite_perp_app_vel, j_B) ⋅ e_x / norm(vec_3d(kite_perp_app_vel, j_B)), -1), 1),
        kite_clipped_wind_angle[j_C] ~ min(max(vec_3d(kite_perp_app_vel, j_C) ⋅ e_x / norm(vec_3d(kite_perp_app_vel, j_C)), -1), 1),
        kite_clipped_wind_angle[j_D] ~ min(max(vec_3d(kite_perp_app_vel, j_D) ⋅ e_x / norm(vec_3d(kite_perp_app_vel, j_D)), -1), 1),
        kite_angle_of_attack[j_A] ~ 0,
        kite_angle_of_attack[j_B] ~ rad2deg(π - acos(kite_clipped_wind_angle[j_B]) - alpha_depower) + s.set.alpha_zero,
        kite_angle_of_attack[j_C] ~ rad2deg(π - acos(kite_clipped_wind_angle[j_C]) + rel_steering * ks) + s.set.alpha_ztip,
        kite_angle_of_attack[j_D] ~ rad2deg(π - acos(kite_clipped_wind_angle[j_D]) - rel_steering * ks) + s.set.alpha_ztip,
    ])

    spline_cd = LinearInterpolation(s.set.cd_list, s.set.alpha_cd, extrapolation=ExtrapolationType.Linear)
    dspline_cd_dalpha = spline_derivative(spline_cd, s.set.alpha_cd)

    spline_cl = LinearInterpolation(s.set.cl_list, s.set.alpha_cl, extrapolation=ExtrapolationType.Linear)
    dspline_cl_dalpha = spline_derivative(spline_cl, s.set.alpha_cl)

    if s.set.version == 3
        drag_corr = 1.0
    else
        drag_corr = DRAG_CORR
    end

    eq_aero_coeff = scalar([
        kite_cl[j_A] ~ 0,
        kite_cl[j_B] ~ calc_aero_coeff(spline_cl, dspline_cl_dalpha, kite_angle_of_attack[j_B]),
        kite_cl[j_C] ~ calc_aero_coeff(spline_cl, dspline_cl_dalpha, kite_angle_of_attack[j_C]),
        kite_cl[j_D] ~ calc_aero_coeff(spline_cl, dspline_cl_dalpha, kite_angle_of_attack[j_D]),
        kite_cd[j_A] ~ 0,
        kite_cd[j_B] ~ calc_aero_coeff(spline_cd, dspline_cd_dalpha, kite_angle_of_attack[j_B]) * drag_corr,
        kite_cd[j_C] ~ calc_aero_coeff(spline_cd, dspline_cd_dalpha, kite_angle_of_attack[j_C]) * drag_corr,
        kite_cd[j_D] ~ calc_aero_coeff(spline_cd, dspline_cd_dalpha, kite_angle_of_attack[j_D]) * drag_corr,
    ])

    rel_side_area = s.set.rel_side_area / 100.0    # defined in percent
    K = 1 - rel_side_area                        # correction factor for the drag

    eq_aero_force = scalar([
        vec_3d(kite_lift_direction, j_A) ~ zeros(3),
        vec_3d(kite_lift_direction, j_B) ~ vec_3d(kite_app_vel, j_B) × e_y / norm(vec_3d(kite_app_vel, j_B) × e_y),
        vec_3d(kite_lift_direction, j_C) ~ vec_3d(kite_app_vel, j_C) × e_z / norm(vec_3d(kite_app_vel, j_C) × e_z),
        vec_3d(kite_lift_direction, j_D) ~ e_z × vec_3d(kite_app_vel, j_D) / norm(e_z × vec_3d(kite_app_vel, j_D)),
        vec_3d(kite_lift, j_A) ~ zeros(3),
        vec_3d(kite_lift, j_B) ~ (-0.5 * kite_air_dens * (norm(vec_3d(kite_perp_app_vel, j_B)))^2 * s.set.area * kite_cl[j_B]) * vec_3d(kite_lift_direction, j_B),
        vec_3d(kite_lift, j_C) ~ (-0.5 * kite_air_dens * (norm(vec_3d(kite_perp_app_vel, j_C)))^2 * s.set.area * kite_cl[j_C]) * vec_3d(kite_lift_direction, j_C) * rel_side_area,
        vec_3d(kite_lift, j_D) ~ (-0.5 * kite_air_dens * (norm(vec_3d(kite_perp_app_vel, j_D)))^2 * s.set.area * kite_cl[j_D]) * vec_3d(kite_lift_direction, j_D) * rel_side_area,
        vec_3d(kite_drag, j_A) ~ zeros(3),
        vec_3d(kite_drag, j_B) ~ (-0.5 * kite_air_dens * norm(vec_3d(kite_app_vel, j_B)) * s.set.area * kite_cd[j_B]) * vec_3d(kite_app_vel, j_B) * K,
        vec_3d(kite_drag, j_C) ~ (-0.5 * kite_air_dens * norm(vec_3d(kite_app_vel, j_C)) * s.set.area * kite_cd[j_C]) * vec_3d(kite_app_vel, j_C) * K * rel_side_area,
        vec_3d(kite_drag, j_D) ~ (-0.5 * kite_air_dens * norm(vec_3d(kite_app_vel, j_D)) * s.set.area * kite_cd[j_D]) * vec_3d(kite_app_vel, j_D) * K * rel_side_area,
    ])

    eq_aero_moments = scalar([
        kite_pitch ~ 0, # TODO: Implement,
        kite_pitching_moment_force ~ (0.5 * kite_air_dens * s.set.area * norm(vec_3d(kite_perp_app_vel, j_A))^2 * s.set.cmq * D.(kite_pitch) * s.set.cord_length) * e_z,
        kite_steering_moment_force ~ (0.5 * kite_air_dens * s.set.area * (0.5 * (norm(vec_3d(kite_perp_app_vel, j_C)) + norm(vec_3d(kite_perp_app_vel, j_D))))^2 * s.set.smc * rel_steering * ks) * e_x,
    ])

    eq_kite_aero = vcat(eq_aero_base, eq_aero_vel, eq_aero_aoa, eq_aero_coeff, eq_aero_force, eq_aero_moments)

    if s.set.version == 3
        F_a_b = vec_3d(kite_lift, j_B) + vec_3d(kite_drag, j_B) - vec_3d(kite_drag, j_C) - vec_3d(kite_drag, j_D)
    else
        F_a_b = vec_3d(kite_lift, j_B) + vec_3d(kite_drag, j_B)
    end

    forces[i_A, :] += kite_pitching_moment_force
    forces[i_B, :] += F_a_b
    forces[i_C, :] += vec_3d(kite_lift, j_C) + vec_3d(kite_drag, j_C) - 0.5 * kite_pitching_moment_force + 0.5 * kite_steering_moment_force
    forces[i_D, :] += vec_3d(kite_lift, j_D) + vec_3d(kite_drag, j_D) - 0.5 * kite_pitching_moment_force - 0.5 * kite_steering_moment_force

    eq_misc = scalar([
        v_wind_gnd ~ [cos(downwind_direction), sin(downwind_direction), 0] * speed_wind_gnd
    ])

    eq_outputs = scalar([
        pos_kite ~ vec_3d(pos, kite_index(s)),
        elevation_kite ~ KiteUtils.calc_elevation(vec_3d(pos, kite_index(s))),
        azimuth_kite ~ calc_azimuth(vec_3d(pos, kite_index(s)), [v_wind_gnd[i] for i in 1:3]),
        range_kite ~ norm(vec_3d(pos, kite_index(s))),
        vel_kite ~ vec_3d(vel, kite_index(s)),
        radial_vel ~ vec_3d(pos, kite_index(s)) ⋅ vec_3d(vel, kite_index(s)) / norm(vec_3d(pos, kite_index(s))),
        tangential_vel ~ sqrt(norm(vec_3d(vel, kite_index(s)))^2 - radial_vel^2),
        course_angle ~ 0,
        orient_euler ~ [0, 0, 0],
        orient_euler_rate ~ [0, 0, 0],
        tether_length ~ l_tether,
        reel_speed ~ v_tether,
        winch_force ~ norm(vec_3d(forces, 1)),
        c_l ~ kite_cl[j_B]
    ])

    G = [Num(0), Num(0), Num(-G_EARTH)]

    eq_particles = vcat(
        vec(scalar([
            D.(pos[2:end, :]) ~ vel[2:end, :],
            D.(vel[2:end, :]) ~ acc[2:end, :],
            acc[2:end, :] ~ Symbolics.Arr{Num,2}(-forces ./ masses .+ G')[2:end, :]
        ])),
        scalar([
            D.(vec_3d(pos, 1)) ~ zeros(3),
            D.(vec_3d(vel, 1)) ~ zeros(3),
            vec_3d(acc, 1) ~ zeros(3)
        ])
    )
    eq_tether = [
        D.(l_tether) ~ v_tether,
        D.(v_tether) ~ a_tether,
        a_tether ~ winch_calculations(v_tether, winch_force, set_speed, set_torque)
    ]

    eq_total = vcat(eq_particles, eq_tether, eq_misc, eq_sps, eq_kcu_drag, eq_refrence_frame, eq_kite_aero, eq_outputs)

    @parameters begin
        speed_wind_gnd_trim = SPEED_WIND_GND0
        downwind_direction_trim = DOWNWIND_DIR0
        pos_kite_x_tim = POS_KITE0[1]
        pos_kite_y_tim = POS_KITE0[2]
        pos_kite_z_tim = POS_KITE0[3]
    end

    eq_trimming = [
        speed_wind_gnd ~ speed_wind_gnd_trim,
        downwind_direction ~ downwind_direction_trim,
        pos_kite[1] ~ pos_kite_x_tim,
        pos_kite[2] ~ pos_kite_y_tim,
        pos_kite[3] ~ pos_kite_z_tim,
    ]

    @named trimming_sys = ODESystem(vcat(eq_total, eq_trimming), t)
    @time "Reducing the trimming model" simple_trimming_sys = structural_simplify(trimming_sys)
    @time "Setting up the trimming problem" trimming_prob = ODEProblem(simple_trimming_sys, [D.(pos) => VEL0, D.(vel) => ACC0], (0, 10))
    @time "Solving the trimming problem" trimming_sol = solve(trimming_prob, Rodas4P(autodiff=false))

    sys = ODESystem(eq_total, t)
    @time "Reducing the model" reduced_sys = structural_simplify(sys, fully_determined=false)

    return reduced_sys, trimming_sol, [pos, vel, acc, l_tether, v_tether, a_tether], ModelingToolkit.inputs(sys), ModelingToolkit.outputs(sys)
end

function calc_initial_guess(s::KPS4Symbolic)
    downwind_dir = wrap2pi(pi - deg2rad(s.set.upwind_dir))
    v_wind_gnd = [cos(downwind_dir), sin(downwind_dir), 0] * s.set.v_wind

    el_rad = deg2rad(s.set.elevations[1])
    az_rad = deg2rad(s.set.azimuths[1]) + downwind_dir

    range = s.set.kite_distances[1]
    tether_length = s.set.l_tethers[1]
    reel_speed = s.set.v_reel_outs[1]
    a_tether = 0

    unit_vector = [cos(el_rad) * cos(az_rad), cos(el_rad) * sin(az_rad), sin(el_rad)]
    pos_kite = range * unit_vector
    pos_kcu = tether_length * unit_vector
    vel_kite = [0, 0, 0] # TODO: implement

    n = n_points(s)

    v_wind_kite = calc_wind_factor(s.am, pos_kite[3]) * v_wind_gnd
    angular_rate_kite = (pos_kite × vel_kite) / norm(pos_kite)^2
    _, _, pos_A, pos_B, pos_C, pos_D = KiteUtils.get_particles(s.set.height_k, s.set.h_bridle, s.set.width, s.set.m_k, pos_kcu, unit_vector, v_wind_kite)

    m = s.set.segments
    POS0 = [(0:m) / m .* pos_kcu'; pos_A'; pos_B'; pos_C'; pos_D']
    VEL0 = reel_speed * -unit_vector' .* (1:n) / n .+ vel_kite ⋅ unit_vector * unit_vector' + stack([angular_rate_kite × POS0[i, :] for i in 1:n], dims=1)
    ACC0 = zeros(n, 3)

    AOA0 = [0, s.set.alpha_zero, s.set.alpha_ztip, s.set.alpha_ztip]

    return POS0, VEL0, ACC0, tether_length, reel_speed, a_tether, AOA0, s.set.v_wind, downwind_dir, 0, pos_kite
end


# TODO: remove
set_data_path("./data")
SETFILE = "system_v9.yaml"
SET = deepcopy(load_settings(SETFILE))
SET.segments = 1
TEST_KCU = KCU(SET)
s = KPS4Symbolic(TEST_KCU)
