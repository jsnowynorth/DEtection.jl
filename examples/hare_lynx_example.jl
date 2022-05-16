


using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using DataFrames, DataFramesMeta, Chain, CSV
using Pipe: @pipe



###################### load data ######################
@pipe hare = "../DEtection/data/hare_lynx_data.csv" |>
             CSV.File |>
             DataFrame

@pipe Y = hare[:, 2:3] |> Matrix |> transpose |> copy

# plot the data
plot(hare[:, :Hare], hare[:, :Lynx], legend = false, color = colormap("RdBu"))


###################### library ######################
# function Lambda(x::Vector)
#     X = x[1]
#     Y = x[2]
#     return ([X, Y, X * Y, X^2, Y^2, X^2 * Y, X * Y^2, X^3, Y^3])
# end

ΛNames = ["Phi"]
function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  H = u[1,:]
  L = u[2,:]
  
  return [H, L, H .* L, H .^2, L .^2, H .^2 .* L, H .* L .^2, H .^3, L .^3]

end

######################### run sampler #########################

nbasis = 40
TimeStep = Vector(range(0.1, size(hare,1)*0.1, step = 0.1))
batch_size = 10
buffer = 2
v0 = 1e-6
v1 = 1e4
order = 1
learning_rate = 1e0

model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 1000)

print_equation(["Hₜ", "Lₜ"], model, pars, posterior, cutoff_prob=0.95, p=0.95)

post = posterior_summary(model, pars, posterior)
post.M
post.M_hpd
post.gamma
sqrt.(post.ΣV)
post.ΣU
post.π

post_mean, post_sd = posterior_surface(model, pars, posterior)

plot(post_mean[1,:])
plot!(Y[1,:])


# samps = DEtection_sampler(Y, 1e1, 1e1,
#     nits = 10000,
#     burnin = 5000,
#     nbasis = 35,
#     h = 0.1,
#     batch_size = 20,
#     buffer = 2,
#     v0 = 1e-6,
#     v1 = 1e4,
#     order = 1,
#     latent_space = 2,
#     Lambda)
# #

# par_names = ["H"; "L"; "H*L"; "H^2"; "L^2"; "H^2*L"; "H*L^2"; "H^3"; "L^3"]
# sys_names = ["dH/dt"; "dL/dt"]
# eqs = print_equation(M = samps['M'],
#     M_prob = samps["gamma"],
#     par_names = par_names,
#     sys_names = sys_names,
#     cutoff_prob = 0.99)
# eqs["mean"]
# eqs["lower"]
# eqs["upper"]


# plot(samps['M'][:1, :1, :], linecolor = "black")
# plot(samps['M'][:1, :2, :], linecolor = "black")
# plot(samps['M'][:1, :3, :], linecolor = "black")
# plot(samps['M'][:2, :1, :], linecolor = "black")
# plot(samps['M'][:2, :2, :], linecolor = "black")
# plot(samps['M'][:2, :3, :], linecolor = "black")