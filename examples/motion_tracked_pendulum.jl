


using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using DataFrames, DataFramesMeta, Chain, CSV
using Pipe: @pipe



###################### load data ######################
@pipe pend = "../DEtection/data/motion_tracked_pendulum.csv" |>
             CSV.File |>
             DataFrame

Y = Matrix(pend[:, 2]')

# plot the data
# plot(pend[:, :time], pend[:, :theta], legend = false)

# plot(pend[:, :time])

# pend[2, :time] - pend[1, :time]

###################### library ######################

ΛNames = ["Phi", "Phi_t"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  ϕt = Φ[2]
  x = (A * ϕ')'
  xt = (A * ϕt')'
  
  return [x, sin.(x), cos.(x), x ./ sin.(x), x ./ cos.(x), xt, xt.^2, xt .* sin.(x), xt .* cos.(x), sin.(xt), cos.(xt)]
  
end

######################### run sampler #########################

nbasis = 250
TimeStep = Vector(pend[:, :time])
batch_size = 20
buffer = 2
v0 = 1e-6
v1 = 1e4
order = 2
learning_rate = 1e-8

model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 1000, orderTime = 2, learning_rate_end = 1e-8)

model

######################### output #########################

print_equation(["Xₜₜ"], model, pars, posterior, cutoff_prob=0.9, p=0.95)


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
