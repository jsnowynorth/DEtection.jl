



using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics


###################### functions to generate data ######################

function lorenz_63(X, pars)
  sigma = pars[1]
  beta = pars[2]
  rho = pars[3]
  x = X[1]
  y = X[2]
  z = X[3]
  x_prime = sigma * (y - x)
  y_prime = x * (rho - z) - y
  z_prime = x * y - beta * z
  return Array([x_prime, y_prime, z_prime])
end

function rk(fn, X, pars, h)

  k1 = h * fn(X, pars)
  k2 = h * fn(X + k1 / 2, pars)
  k3 = h * fn(X + k2 / 2, pars)
  k4 = h * fn(X + k3, pars)

  Xt = X + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
  return Xt

end

###################### create data ######################
X = [-8, 7, 27]
pars = [10, 8 / 3, 28]
h = 0.01
end_time = 10
nsamps = convert(Int, (end_time / h))

Y = Array{Float64}(undef, 3, nsamps)
Y[:, 1] = rk(lorenz_63, X, pars, h)
for i in 2:nsamps
  Y[:, i] = rk(lorenz_63, Y[:, i-1], pars, h)
end

# Y_prime = [lorenz_63(Y[:, i], pars) for i in 1:nsamps]
# Y_prime = reduce(hcat, Y_prime)


######################### missing data #########################
Y_missing = copy(Y)
Y_missing = Array{Union{Missing,Float64}}(missing, 3, nsamps)
obs_inds = sample(1:nsamps, convert(Int, nsamps * 0.95), replace = false)
Y_missing[:, obs_inds] = copy(Y[:, obs_inds])

######################### noisy data #########################
Random.seed!(1)
# add varying amounts of noise
R = 1 * Matrix{Float64}(I, 3, 3)
# R = 5 * Matrix{Float64}(I, 3, 3)
# R = 10 * Matrix{Float64}(I, 3, 3)

Z = copy(Y) + rand(MvNormal([0.0; 0.0; 0.0], R), nsamps)
Z_missing = copy(Z)
Z_missing = Array{Union{Missing,Float64}}(missing, 3, nsamps)
Z_missing[:, obs_inds] = copy(Z[:, obs_inds])


###################### library ######################

ΛNames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  X = u[1,:]
  Y = u[2,:]
  Z = u[3,:]
  
  return [X, Y, Z, X .* Y, X .* Z, Y .* Z, X.^2, Y.^2, Z.^2, X.^2 .* Y, X .* Y.^2, X.^2 .* Z, X .* Z.^2, Y.^2 .* Z, Y .* Z.^2, X.^3, Y.^3, Z.^3, X .* Y .* Z]

end

# function Λ(A, Φ)
    
#   ϕ = Φ[1]

#   X = A[1,:]' * ϕ'
#   Y = A[2,:]' * ϕ'
#   Z = A[3,:]' * ϕ'
  
#   return [X, Y, Z, X .* Y, X .* Z, Y .* Z, X.^2, Y.^2, Z.^2, X.^2 .* Y, X .* Y.^2, X.^2 .* Z, X .* Z.^2, Y.^2 .* Z, Y .* Z.^2, X.^3, Y.^3, Z.^3, X .* Y .* Z]

# end


######################### run sampler #########################

nbasis = 400
TimeStep = Vector(range(0.01, end_time, step = 0.01))
batch_size = 50
buffer = 20
v0 = 1e-6
v1 = 1e4
order = 1
learning_rate = 1e1


model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 1000)
model, pars, posterior = DEtection_sampler(Z_missing, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 5000)

model


######################### output #########################

print_equation(["Xₜ", "Yₜ", "Zₜ"], model, pars, posterior, cutoff_prob=0.95, p=0.95)


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

plot(post_mean[3,:])
plot!(Y[3,:])


plot(posterior.M[1,1,:])
plot(posterior.M[1,2,:])
plot(posterior.M[2,1,:])
plot(posterior.M[2,2,:])
plot(posterior.M[2,5,:])
plot(posterior.M[3,3,:])
plot(posterior.M[3,4,:])

plot(posterior.ΣU[1,1,:])
plot(posterior.ΣV[1,1,:])


plot(posterior.A[1,20,:])





learning_rate = 1e1


pars, model = create_pars(Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames)


pars = update_M!(pars, model)
pars = update_gamma!(pars, model)
pars = update_ΣV!(pars, model)
pars = update_ΣU!(pars, model)
pars = update_A!(pars, model)

# pars.a_V
pars.ΣV
pars.M .* pars.gamma
pars.ΣU
# pars.a_U

plot(pars.State.State["U"][1,:])
plot!(Y[1,:])

plot(pars.State.State["U"][2,:])
plot!(Y[2,:])

plot(pars.State.State["U"][3,:])
plot!(Y[3,:])