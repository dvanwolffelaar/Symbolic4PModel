using Parameters, Statistics
using DataInterpolations, Dierckx
using ModelingToolkit
using ModelingToolkit: Symbolics, @register_symbolic
using ModelingToolkit: t_nounits as t, D_nounits as D

@with_kw mutable struct Settings
    alpha_cd::Vector{Float64} = []
    cd_list::Vector{Float64} = []
end

function make_cd_interp_num(set::Settings)
    Spline1D(set.alpha_cd, set.cd_list)
end

function make_cd_interp(set::Settings)
    CubicSpline(set.cd_list, set.alpha_cd)
end

SET = Settings(alpha_cd=[-180.0, -170.0, -140.0, -90.0, -20.0, 0.0, 20.0, 90.0, 140.0, 170.0, 180.0], cd_list=[0.5, 0.5, 0.5, 1.0, 0.2, 0.1, 0.2, 1.0, 0.5, 0.5, 0.5])
dierckx_spline = make_cd_interp_num(SET)
interpolations_spline = make_cd_interp(SET)
test_grid = -180:1:180

difference = dierckx_spline(test_grid) - interpolations_spline(test_grid)
mean_e = mean(difference)
mean_abs_e = mean(abs.(difference))
max_e = maximum(abs.(difference))
min_e = minimum(abs.(difference))

println("Mean error: $mean_e Mean absolute error $mean_abs_e")
println("Min error: $min_e Min error $max_e")
