


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
ΛNames = ["Phi"]
function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  H = u[1,:]
  L = u[2,:]
  
  return [H, L, H .* L, H.^2, L.^2, H.^2 .* L, H .* L .^2, H.^3, L.^3]

end

######################### run sampler #########################

nbasis = 35
TimeStep = Vector(range(0.1, size(hare,1)*0.1, step = 0.1))
batch_size = 10
buffer = 2
v0 = 1e-6
v1 = 1e4
order = 1
learning_rate = 1e1

model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 10000)

model


######################### output #########################

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

plot(post_mean[2,:])
plot!(Y[2,:])

