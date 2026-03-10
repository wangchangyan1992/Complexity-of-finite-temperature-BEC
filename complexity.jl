include(joinpath(@__DIR__, "tools.jl"))
using LinearAlgebra
using BenchmarkTools
using PyPlot
using NLsolve
using QuadGK
using  LsqFit
pygui(true)


#%%------------------------------------------------------------
function paraSol(alp_1, r_1)
    function f(x)
        c0, c1 = x[1], x[2]
        f1 = (sinh(-alp_1) / sinh(c1))^2 + (2c0/c1)^2 - 1
        f2 = 2c0/c1 * tanh(c1) - tanh(r_1-2c0)
        return [f1, f2]
    end

    sol = nlsolve(f, [0.1, 2])

    # return sol, f(sol.zero)
    return sol.zero, f(sol.zero)
end

function Complexity(temp_1, r_1)
    alp_1 = log(tanh(1 / (4temp_1)))
    sol, check = paraSol(alp_1, r_1)
    c0, c1 = sol
    return sqrt(2c0^2 + c1^2/2)
end

function bComp(temps, r_1s)
    alp_1s = log.(tanh.(1 ./ (4temps)))
    comp_s = []
    return comp_s = [Complexity(temp_1, r_1) / 2 for r_1 in r_1s, temp_1 in temps]
end

function R(alp_1, g0)
    g = 2tanh(g0)
    f(x) = sqrt(4g / (1 + cosh(x)^2) / (1 + cosh(x)^2 - g))
    return quadgk(f, alp_1, 0)[1]
end

function gSol(alp_1, r_1)
    f(g0) = R(alp_1, g0[1]) - r_1
    sol = nlsolve(f, [0.1])
    return 2tanh(sol.zero[1]), f(sol.zero)
end

function b1Sol(temp_1, r_1)
    alp_1 = log(tanh(1 / (4temp_1)))
    g = gSol(alp_1, r_1)[1]
    f(x) = sqrt((1 + cosh(x)^2) / (1 + cosh(x)^2 - g))
    return quadgk(f, alp_1, 0)[1]
end

function sjComp(temps, r_1s)
    return comp_s = [b1Sol(temp_1, r_1) / 2 for r_1 in r_1s, temp_1 in temps]
end

function bGeodesic(temp_1, r_1)
    alp_1 = log(tanh(1 / (4temp_1)))
    sol, check = paraSol(alp_1, r_1)
    c0, c1 = sol
    ts = [range(0, 0.01, 1001)[1:end-1]; range(0.01, 1, 1001)]
    alp_s = [-asinh(sqrt(1 - (2c0/c1)^2) * sinh(c1 * t)) for t in ts]
    rs = [2c0 * t + atanh(2c0/c1 * tanh(c1 * t)) for t in ts]
    temps = @. 1 / (4atanh(exp(alp_s)))
    return rs, temps, alp_s
end

function sjGeodesic(temp_1, r_1)
    alp_1 = log(tanh(1 / (4temp_1)))
    f(g0) = R(alp_1, g0[1]) - r_1
    sol = nlsolve(f, [0.1])
    g0 = sol.zero[1]

    alp_s = range(0, alp_1, 1001)
    rs = [R(alp, g0) for alp in alp_s]
    temps = @. 1 / (4atanh(exp(alp_s)))
    return rs, temps, alp_s
end

function LinearFit(alp_s, comp)
    @. model(x, p) = p[1] * x + p[2]
    p0 = [0.5, 0.5]
    fit = curve_fit(model, alp_s, comp, p0)
    # fit_comp = model(alp_s, fit.param)
    return fit.param[1]
end

filename(name) = joinpath(@__DIR__, name)


#%%------------------------------------------------------------
temps = range(0.1, 50, 1001)
alp_s = -log.(tanh.(1 ./ (4temps)))
# rs = 0.3:0.5:20
rs = [0.5, 1, 1.5, 2, 3]

b_comps = bComp(temps, rs)
sj_comps = sjComp(temps, rs)



#%%------------------------------------------------------------
ax = pfig()

# for i = axes(b_comps, 1)
#     ax.plot(alp_s, b_comps[i, :])
#     ax.plot(alp_s, b_fac_alp[i][2])
# end

for i = axes(sj_comps, 1)
    ax.plot(alp_s, sj_comps[i, :])
    ax.plot(alp_s, sj_fac_alp[i][2])
end







#%%------------------------------------------------------------
temps = range(2, 50, 1001)
alp_s = -log.(tanh.(1 ./ (4temps)))
r1s = 0.3:0.5:18


b_comps_fac = bComp(temps, r1s)
sj_comps_fac = sjComp(temps, r1s)
b_fac_alp = [LinearFit(alp_s, b_comps_fac[i, :]) for i in axes(b_comps_fac, 1)]
sj_fac_alp = [LinearFit(alp_s, sj_comps_fac[i, :]) for i in axes(sj_comps_fac, 1)]




#%%------------------------------------------------------------
@chain line(r1s, b_fac_alp) line(r1s, sj_fac_alp) plot




#%%------------------------------------------------------------
r1s = 0.3:0.5:18













#%%------------------------------------------------------------
temp_s = [0.5, 1, 2, 3, 5]
r = 2

b_geo_s = [bGeodesic(temp, r) for temp in temp_s]

sj_geo_s = [sjGeodesic(temp, r) for temp in temp_s]


#%%------------------------------------------------------------
@chain line(b_geo[1], b_geo[2]) line(sj_geo[1], sj_geo[2]) plot



#%%------------------------------------------------------------
ax = pfig(1, 3, figsize=(12, 3.5), fontsize=30, framewidth=2.8, ticksize=8, label_pos=(0.02, 0.865))

for i = 1:len(b_geo_s)
    ax[1].plot(b_geo_s[i][1], b_geo_s[i][2], linewidth=2.5)
end

for i = axes(b_comps, 1)
    ax[2].plot(temps, b_comps[i, :], linewidth=2.5)
end

ax[3].plot(r1s, b_fac_alp, linewidth=2.5)

layout(
    ax[1], 
    xlabel=L"r", 
    ylabel=L"T/\varepsilon", 
    legend=[L"T_1/\varepsilon=%$(temp)" for temp in temp_s], 
    loc = "center left", 
    fontsize=15
    )
layout(
    ax[2], 
    xlabel=L"T_1/\varepsilon", 
    ylabel=L"\mathcal{C}_B", 
    legend=[L"r_1=%$(rs[i])" for i = 1:len(rs)], 
    fontsize=15
    )
layout(
    ax[3], 
    xlabel=L"r_1", 
    ylabel=L"f_B(r_1)", 
    ylim = [0.24, 0.38]
)

savefig(filename("fig_b.pdf"))    



#%%------------------------------------------------------------
ax = pfig(1, 3, figsize=(12, 3.5), fontsize=30, framewidth=2.8, ticksize=8, label_pos=(0.02, 0.865))

for i = 1:len(sj_geo_s)
    ax[1].plot(sj_geo_s[i][1], sj_geo_s[i][2], linewidth=2.5)
end

for i = axes(sj_comps, 1)
    ax[2].plot(temps, sj_comps[i, :], linewidth=2.5)
end

ax[3].plot(r1s, sj_fac_alp, linewidth=2.5)

layout(
    ax[1], 
    xlabel=L"r", 
    ylabel=L"T/\varepsilon", 
    legend=[L"T_1/\varepsilon=%$(temp)" for temp in temp_s], 
    loc = "center left", 
    fontsize=16
    )
layout(
    ax[2], 
    xlabel=L"T_1/\varepsilon", 
    ylabel=L"\mathcal{C}_S", 
    legend=[L"r_1=%$(rs[i])" for i = 1:len(rs)], 
    fontsize=18
    )
layout(
    ax[3], 
    xlabel=L"r_1", 
    ylabel=L"f_S(r_1)", 
    ylim = [0.489, 0.502], 
    ydtick = 0.01
)

savefig(filename("fig_sj.pdf"))    



#%%------------------------------------------------------------
line(b_geo_s[5][1],  b_geo_s[5][2]) |> plot





#%%------------------------------------------------------------









#%%------------------------------------------------------------
using QuantumAlgebra



#%%------------------------------------------------------------
# ops = [a(1), a(2), a'(3), a'(4)]
@boson_ops b
@boson_ops c
@boson_ops d

ops = [a(), c(), b'(), d'()]

opsQud(mx, ops) = ops' * mx * ops

sigx = [0 1; 1 0]
sigy = [0 -im; im 0]
sigz = [1 0; 0 -1]
ks = [sigz, im * sigy, -im * sigx]
k0 = I(2)

xs1 = [kron(ks[i], k0) for i = 1:3]
xs2 = [kron(ks[i], sigz) for i = 1:3]
xs3 = [kron(ks[i], sigx) for i = 2:3]
xs = [xs1; xs2; xs3; [kron(sigz, sigx), kron(k0, sigy)]] / 2

kappa = diagm([1, 1, -1, -1])
xs_op = [opsQud(kappa * xs[i], ops) for i = 1:10]



#%%------------------------------------------------------------
corrl = [vacExpVal(xs_op[i] * xs_op[j]) - vacExpVal(xs_op[i])*vacExpVal(xs_op[j]) |> julia_expression for i=1:10, j=1:10]



#%%------------------------------------------------------------
sigs = [sigx, sigy, sigz, I(2)]
xs = [kron(sigs[i], sigs[j]) for i=1:4, j=1:4]

xs_op = [opsQud(kappa * xs[i], ops) for i=1:16]
corrl = [vacExpVal(xs_op[i] * xs_op[j]) - vacExpVal(xs_op[i])*vacExpVal(xs_op[j]) |> julia_expression for i=1:16, j=1:16]



#%%------------------------------------------------------------
n = Pr"n"
a() * a() * a()
