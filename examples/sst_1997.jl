

using DEtection

using StatsBase, LinearAlgebra, Plots, StatsPlots, CSV
using Kronecker
using Missings
using Distributions, Random
using ProgressMeter
using ReverseDiff: JacobianTape, JacobianConfig, jacobian, jacobian!, compile
using ForwardDiff
using Pipe: @pipe
using Statistics
using JLD2
using RCall
using TexTables, KernelDensity
using DataFrames, DataFramesMeta, Chain
using NetCDF
using MultivariateStats
using StatsModels, Combinatorics, IterTools
using Dates
using KernelDensity
using Tables
@rlibrary ggplot2

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using NetCDF, Dates, CSV, DataFrames, DataFramesMeta, Chain





###################### load data ######################

## load data

# ncinfo("../DEtection/data/data_SST_011901_112021.nc")
y = ncread("../DEtection/data/data_SST_011901_112021.nc", "Y")
x = ncread("../DEtection/data/data_SST_011901_112021.nc", "X")
T = ncread("../DEtection/data/data_SST_011901_112021.nc", "T")
sst = ncread("../DEtection/data/data_SST_011901_112021.nc", "anom")

sst = replace(sst, -999.0 => missing)

# contourf(x, y, sst[:,:,1,1000]')

# contourf(x, y, sst[:,:,1,300]', clim = (-2, 4))
# contourf(x, y, sst[:,:,1,301]', clim = (-2, 4))
# contourf(x, y, sst[:,:,1,302]', clim = (-2, 4))
# contourf(x, y, sst[:,:,1,1374]', clim = (-2, 4))
# contourf(x, y, sst[:,:,1,1375]', clim = (-2, 4))



land_id = map(!, isequal.(sst[:, :, 1, 1], missing))
coords = hcat(collect.(vec(collect(Base.product(x, y))))...)
location_inds = DataFrame(x = coords[1, :], y = coords[2, :], sea = reshape(land_id, :))
sea_inds = @chain location_inds begin
    @subset(:sea .== true)
end

function reshape_sst(Z, location_inds, sea_inds, x, y)

    N = size(Z, 2)
    tmp = DataFrame(x = sea_inds[:, :x], y = sea_inds[:, :y])
    if N == 1
        tmpZ = DataFrame(sst = Z[:])
    else
        tmpZ = DataFrame(Z, :auto)
    end

    tmp = hcat(tmp, tmpZ)
    tmp = rightjoin(tmp, location_inds, on = [:x, :y])
    tmp = sort!(tmp, [:y, :x])

    if N == 1
        out = reshape(tmp[:, 3], length(x), length(y))
    else
        out = [reshape(tmp[:, i], length(x), length(y)) for i in 3:(size(tmp, 2)-1)]
    end

    return out
end


Z = reshape(sst[:, :, 1, :], :, length(T))
Z = Array{Float64,2}(Z[location_inds[:, :sea], :])



###################### EOF functions ######################
struct EOF_features
    Phi::Array{Float64}
    spat_mean::Array{Float64}
end

function EOF(Y, k)

    nT = size(Y, 2)
    Y = copy(Y')

    # detrend data
    spat_mean = mean(Y, dims = 1)
    Yt = (1 / sqrt(nT - 1)) * Y - ones(nT) * spat_mean

    # svd
    E = svd(Yt)
    Phi = E.V[:, 1:k]

    TS = (Y - ones(nT) * spat_mean) * Phi

    eofs = EOF_features(Phi, spat_mean)

    return TS, eofs

end

function EOF(sst, k, inds, train, test)

    nT_all = length(inds)
    nT = length(train)
    Z = sst[inds, :]
    Y = sst[train, :]

    # detrend data
    spat_mean = mean(Z, dims = 1)
    Yt = (1 / sqrt(nT - 1)) * (Y - ones(nT) * spat_mean)

    # svd
    E = svd(Yt)
    Phi = E.V[:, 1:k]

    TS = (Z - ones(nT_all) * spat_mean) * Phi
    TS_train = TS[(1:nT), :]
    TS_test = TS[(nT:end), :]

    eofs = EOF_features(Phi, spat_mean)

    return TS_train, TS_test, eofs

end

function reconstruct_sst(Y, eofs)
    return Y * eofs.Phi' .+ eofs.spat_mean
end


###################### compute EOFs ######################

inds = 300:1200
train = 300:1155
test = 1155:1200
endNt = train[end]
startNt = train[1]


k = 10
Y, Y_test, eofs = EOF(Z', k, inds, train, test)
Y = copy(Y')
Y_test = copy(Y_test')

###################### library ######################


ΛNames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  a = u[1, :]
  b = u[2, :]
  c = u[3, :]
  d = u[4, :]
  e = u[5, :]
  f = u[6, :]
  g = u[7, :]
  h = u[8, :]
  i = u[9, :]
  j = u[10, :]
  
  return [a, b, c, d, e, f, g, h, i, j,
            a .* b, a .* c, a .* d, a .* e, a .* f, a .* g, a .* h, a .* i, a .* j,
            b .* c, b .* d, b .* e, b .* f, b .* g, b .* h, b .* i, b .* j,
            c .* d, c .* e, c .* f, c .* g, c .* h, c .* i, c .* j,
            d .* e, d .* f, d .* g, d .* h, d .* i, d .* j,
            e .* f, e .* g, e .* h, e .* i, e .* j,
            f .* g, f .* h, f .* i, f .* j,
            g .* h, g .* i, g .* j,
            h .* i, h .* j,
            i .* j,
            a.^2, b.^2, c.^2, d.^2, e.^2, f.^2, g.^2, h.^2, i.^2, j.^2]

end

# function Lambda(x::Vector) # length 65
#     a = x[1]
#     b = x[2]
#     c = x[3]
#     d = x[4]
#     e = x[5]
#     f = x[6]
#     g = x[7]
#     h = x[8]
#     i = x[9]
#     j = x[10]
#     return ([a, b, c, d, e, f, g, h, i, j,
#         a * b, a * c, a * d, a * e, a * f, a * g, a * h, a * i, a * j,
#         b * c, b * d, b * e, b * f, b * g, b * h, b * i, b * j,
#         c * d, c * e, c * f, c * g, c * h, c * i, c * j,
#         d * e, d * f, d * g, d * h, d * i, d * j,
#         e * f, e * g, e * h, e * i, e * j,
#         f * g, f * h, f * i, f * j,
#         g * h, g * i, g * j,
#         h * i, h * j,
#         i * j,
#         a^2, b^2, c^2, d^2, e^2, f^2, g^2, h^2, i^2, j^2])
# end


######################### sample sgmcmc #########################



nbasis = 175
TimeStep = Vector(range(1/12, size(Y,2)*(1/12), step = 1/12))
batch_size = 50
buffer = 5
v0 = 1e-6
v1 = 1e4
order = 1
learning_rate = 1e0


model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 100)

sysnames = ["aₜ", "bₜ", "cₜ", "dₜ", "eₜ", "fₜ", "gₜ", "hₜ", "iₜ", "jₜ"]
print_equation(sysnames, model, pars, posterior, cutoff_prob=0.95, p=0.95)


post = posterior_summary(model, pars, posterior)
post.M
post.M_hpd
post.gamma
sqrt.(post.ΣV)
post.ΣU
post.π










samps = DEtection_sampler(Y, 1e0, 1e0,
    nits = 20000,
    burnin = 19000,
    nbasis = 175,
    h = 1 / 12,
    batch_size = 50,
    buffer = 5,
    v0 = 1e-6,
    v1 = 1e4,
    order = 1,
    latent_space = 10,
    Lambda)
#

round.(mean(samps["gamma"], dims = 3), digits = 3)
round.(mean(samps['M'], dims = 3)[:, :, 1], digits = 3)

mean(samps['R'], dims = 3)
mean(samps['Q'], dims = 3)

plot(samps['M'][1, 3, :])
plot(samps['M'][2, 3, :])


######################### EOF prediction functions #########################

function new_state(X, M)
    val = M * Lambda(X)
    return Array(val)
end

function rk(fn, X, pars, h)

    k1 = h * fn(X, pars)
    k2 = h * fn(X + k1 / 2, pars)
    k3 = h * fn(X + k2 / 2, pars)
    k4 = h * fn(X + k3, pars)

    Xt = X + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return Xt
end

function predict_state(X, M, h, nsamps)
    ncovs = size(M, 1)
    Y_sim = Array{Float64}(undef, ncovs, nsamps)
    Y_sim[:, 1] = rk(new_state, X, M, h)
    for i in 2:nsamps
        Y_sim[:, i] = rk(new_state, Y_sim[:, i-1], M, h)
    end
    return Y_sim
end

function pred_nino_hpd(Y_test, sst, n_pred, h, sample, start_t, endNt, T, location_inds, sea_inds, x, y)

    # set up predictions for EOFs
    pred_sst_pc = [predict_state(Y_test[:, start_t], sample['M'][:, :, i], h, n_pred) for i in 1:size(sample['M'], 3)]
    inds = [any(isnan.(pred_sst_pc[i])) for i in 1:size(sample['M'], 3)]
    ests = pred_sst_pc .* .!inds


    pred_sst_pc = nothing
    inds = nothing

    pstart = start_t + endNt - 1
    pend = start_t + endNt + n_pred - 2
    nino_true = [mean(sst[((360 .- x).<170).&((360 .- x).>120), (y.>-5).&(y.<5), 1, i]) for i in pstart:pend]

    nino_rec = [[mean(reshape_sst(reconstruct_sst(ests[j]', eofs)', location_inds, sea_inds, x, y)[i][((360 .- x).<170).&((360 .- x).>120), (y.>-5).&(y.<5)]) for i in 1:n_pred] for j in 1:size(ests, 1)]
    hpd_ints = hcat(collect.([hpd(reduce(hcat, nino_rec)[i, :], p = 0.95) for i in 1:n_pred])...)

    # nino_pred = DataFrame(start = start_t, 
    #   Ind = 1:n_pred, 
    #   date = Dates.Date(1960,1,1) + Dates.Month.(Int.(floor.(T[pstart:pend]))), 
    #   True = nino_true, 
    #   Pred = vcat(nino_true[1], mean(nino_rec)),
    #   Lower = vcat(nino_true[1], hpd_ints[1,:]),
    #   Upper = vcat(nino_true[1], hpd_ints[2,:]))
    nino_pred = DataFrame(start = start_t,
        Ind = 1:n_pred,
        date = Dates.Date(1960, 1, 1) + Dates.Month.(Int.(floor.(T[pstart:pend]))),
        True = nino_true,
        Pred = mean(nino_rec),
        Lower = hpd_ints[1, :],
        Upper = hpd_ints[2, :])


    return nino_pred
end


######################### predict #########################

start_date = Dates.Date(1960, 1, 1)
Tpred = start_date + Dates.Month(Int(floor(T[endNt])))

forecast = 2 # note 1 is the last day of the training set
nino_df = pred_nino_hpd(Y_test, sst, 12, 1 / 12, samps, forecast, endNt, T, location_inds, sea_inds, x, y)

nino_df = @chain nino_df begin
    @subset(:Ind .<= 10)
end


######################### plot prediction #########################

tick_marks = 1:1:size(nino_df, 1)
DateTick = Dates.format.(nino_df[tick_marks, :date], "u yy")

p = plot(nino_df[:, :Ind], nino_df[:, :True], color = :blue, legend = false, grid = false)
p = plot!(xticks = (tick_marks, DateTick), xtickfontsize = 10, ytickfontsize = 10, xrotation = 45)
p = plot!(nino_df[:, :Ind], nino_df[:, :Pred], color = :red)
p = plot!(nino_df[:, :Ind], nino_df[:, :Lower], fillrange = nino_df[:, :Upper], fillcolor = :red, fillalpha = 0.35, color = :black)
p = plot!(nino_df[:, :Ind], nino_df[:, :Upper], color = :black)
