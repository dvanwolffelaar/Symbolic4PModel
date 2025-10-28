using DataInterpolations, ModelingToolkit, Parameters, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D

@with_kw mutable struct Settings
    alpha_cd::Vector{Float64}
    cd_list::Vector{Float64}
end

function make_cd_interp(set::Settings)
    t_s = set.alpha_cd
    u_s = set.cd_list
    spl = CubicSpline(u_s, t_s, extrapolation=ExtrapolationType.Linear)
    spl_d = QuadraticSpline(map(x -> DataInterpolations.derivative(spl, x), t_s), t_s, extrapolation=spl.extrapolation_left)

    return spl, spl_d
end

SET = Settings(
    alpha_cd=[-180.0, -170.0, -140.0, -90.0, -20.0, 0.0, 20.0, 90.0, 140.0, 170.0, 180.0],
    cd_list=[0.5, 0.5, 0.5, 1.0, 0.2, 0.1, 0.2, 1.0, 0.5, 0.5, 0.5]
)

spline, spline_d = make_cd_interp(SET)

function calc_aero_coeff(spline::CubicSpline, spline_d::QuadraticSpline, alpha)
    return spline(alpha)
end

@register_symbolic calc_aero_coeff(spline::CubicSpline, spline_d::QuadraticSpline, alpha)
Symbolics.derivative(::typeof(calc_aero_coeff), args::NTuple{3,Any}, ::Val{3}) = args[2](args[3])


@variables x(t) y(t) z(t) r(t) ρ(t)
@variables u(t) [input = true] v(t) [input = true]

eq_total = [
    D(x) ~ (-x + u) / r,
    D(y) ~ (-y) / r,
    D(z) ~ (-z + v) / ρ,
    r ~ x^2 + y^2,
    ρ ~ calc_aero_coeff(spline, spline_d, z)
]

@parameters begin
    r_trim = 2
    rho_trim = 0.2
end

eq_trimming = [
    r ~ r_trim,
    ρ ~ rho_trim
]

trimming_conditions = [D.(x) => 0, D.(y) => 0, D.(z) => 0]
initial_guesses = [x => 1, y => 0, z => 3, u => 2, v => 1]

@named trimming_sys = ODESystem(vcat(eq_total, eq_trimming), t)
@time simple_trimmings_sys = structural_simplify(trimming_sys, fully_determined=false)
@time trimming_prob = ModelingToolkit.InitializationProblem(simple_trimmings_sys, 0, trimming_conditions, guesses=initial_guesses)
@time trimming_sol = solve(trimming_prob, abstol=1e-16)

inputs = ModelingToolkit.inputs(trimming_sys)
trimming_sol[inputs]