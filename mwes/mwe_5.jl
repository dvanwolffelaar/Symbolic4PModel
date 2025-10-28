using DataInterpolations, ModelingToolkit, Parameters, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D

@with_kw mutable struct Settings
    alpha_cd::Vector{Float64}
    cd_list::Vector{Float64}
end

function make_cd_interp(set::Settings)
    CubicSpline(set.cd_list, set.alpha_cd, extrapolation=ExtrapolationType.Linear)
end

SET = Settings(
    alpha_cd=[-180.0, -170.0, -140.0, -90.0, -20.0, 0.0, 20.0, 90.0, 140.0, 170.0, 180.0],
    cd_list=[0.5, 0.5, 0.5, 1.0, 0.2, 0.1, 0.2, 1.0, 0.5, 0.5, 0.5]
)

spline = make_cd_interp(SET)

function calc_aero_coeff(spline::CubicSpline, alpha)
    return spline(alpha)
end

@register_symbolic calc_aero_coeff(spline::CubicSpline, alpha)

@variables x(t) y(t) z(t) r(t) ρ(t)

eq_total = [
    D(x) ~ (-x + y) / r,
    D(y) ~ (-y + z) / r,
    D(z) ~ (-z) / ρ,
    r ~ x^2 + y^2,
    ρ ~ spline(z)
]

@time println("Comparisson for the setup time using either symbolic functions or direct splines")

@named sys = ODESystem(eq_total, t)
@time "Reducing the model (Using symbolic function for spline)" simple_sys = structural_simplify(sys)
@time "Setting up the problem (Using symbolic function for spline)" ode_prob = ODEProblem(simple_sys, [D.(x) => 0, D.(y) => 0, D.(z) => 0], (0, 10), guesses=[x => 1, y => 0, z => 3])
@time "Solving the problem (Using symbolic function for spline)" sol = solve(ode_prob, Rodas4P(autodiff=false), abstol=1e-16);

done = true
