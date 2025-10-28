using DataInterpolations
using KiteUtils
using ControlPlots, LaTeXStrings
using Statistics

set_data_path("./data")
SETFILE = "system_v9.yaml"
SET = deepcopy(load_settings(SETFILE))

function interp_derivative(spline, t_s, order=1, spline_type=CubicSpline)
    return spline_type(map(x -> DataInterpolations.derivative(spline, x, order), t_s), t_s, extrapolation=spline.extrapolation_left)
end

function generate_splines(t, u)
    spline_cl1 = CubicSpline(u, t, extrapolation=ExtrapolationType.Linear)
    spline_cl2 = LinearInterpolation(u, t, extrapolation=ExtrapolationType.Linear)
    spline_cl_d1 = interp_derivative(spline_cl1, t)
    spline_cl_d2 = interp_derivative(spline_cl2, t, 1, LinearInterpolation)
    spline_cl_dd1 = interp_derivative(spline_cl1, t, 2)
    spline_cl_dd2 = interp_derivative(spline_cl_d1, t)

    return ((spline_cl1, spline_cl2), (spline_cl1, spline_cl_d2), (spline_cl_dd1, spline_cl_dd2))
end

(s_cl_1, s_cl_2), (s_cl_d_1, s_cl_d_2), (s_cl_dd_1, s_cl_dd_2) = generate_splines(SET.alpha_cl, SET.cl_list)
(s_cd_1, s_cd_2), (s_cd_d_1, s_cd_d_2), (s_cd_dd_1, s_cd_dd_2) = generate_splines(SET.alpha_cd, SET.cd_list)

alpha_s = -10:0.1:20

cl_1 = s_cl_1(alpha_s)
cl_2 = s_cl_2(alpha_s)
cl_d_1 = s_cl_d_1(alpha_s)
cl_d_2 = s_cl_d_2(alpha_s)
cl_dd_1 = s_cl_dd_1(alpha_s)
cl_dd_2 = s_cl_dd_2(alpha_s)

cd_1 = s_cd_1(alpha_s)
cd_2 = s_cd_2(alpha_s)
cd_d_1 = s_cd_d_1(alpha_s)
cd_d_2 = s_cd_d_2(alpha_s)
cd_dd_1 = s_cd_dd_1(alpha_s)
cd_dd_2 = s_cd_dd_2(alpha_s)

plt.close("all")

display(plot(alpha_s, [cl_1, cl_2]; xlabel=L"\mathrm{AoA}~\alpha", ylabel=L"CL", labels=["CL_Cubic", "CL_linear"], fig="CL"))
display(plot(alpha_s, [cd_1, cd_2]; xlabel=L"\mathrm{AoA}~\alpha", ylabel=L"CD", labels=["CD_Cubic", "CD_linear"], fig="CD"))
display(plot(alpha_s, [cl_d_1, cl_d_2]; xlabel=L"\mathrm{AoA}~\alpha", ylabel=L"\frac{dC_L}{d\alpha}", labels=["CL_Cubic", "CL_linear"], fig="CL_D"))
display(plot(alpha_s, [cd_d_1, cd_d_2]; xlabel=L"\mathrm{AoA}~\alpha", ylabel=L"\frac{dC_D}{d\alpha}", labels=["CD_Cubic", "CD_linear"], fig="CD_D"))
display(plot(alpha_s, [cl_dd_1, cl_dd_2, cd_dd_1, cd_dd_2]; xlabel=L"\mathrm{AoA}~\alpha", ylabel=L"\frac{d^2C_L, C_D}{d\alpha^2}", labels=["CL1", "CL2", "CD1", "CD2"], fig="CL_CD_DD"))
