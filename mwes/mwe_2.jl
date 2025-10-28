using ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Parameters, DocStringExtensions, Rotations
using NonlinearSolve, SteadyStateDiffEq, OrdinaryDiffEq
using ModelingToolkit: Symbolics, @register_symbolic
using ModelingToolkit: t_nounits as t, D_nounits as D


@variables begin
    x(t)
    y(t)
    z(t)
    r(t)
    rho(t)
end

@variables u(t) [input = true] v(t) [input = true]

eq_total = [
    D.(x) ~ (-x + u) / r,
    D.(y) ~ (-y) / r,
    D.(z) ~ (-z + v) / rho,
    r ~ x^2 + y^2,
    rho ~ x^2 + y^2 + z^2
]

@parameters begin
    r_trim = 2
    rho_trim = 4
end

eq_trimming = [
    r ~ r_trim,
    rho ~ rho_trim
]

@named trimming_sys = ODESystem(vcat(eq_total, eq_trimming), t)
simple_trimmings_sys = structural_simplify(trimming_sys)
trimming_ode_prob = ODEProblem(simple_trimmings_sys, [D.(x) => 0, D.(y) => 0, D.(z) => 0], (0, 10), guesses=[x => 1, y => 2, z => 3, u => 2, v => 1])
trimming_sol = solve(trimming_ode_prob, Rodas4P(autodiff=false), abstol=1e-16)

inputs = ModelingToolkit.inputs(trimming_sys)
trimming_sol[inputs]

