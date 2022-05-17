

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics


###################### functions to generate data ######################

function pendulum(X, pars)
  g = pars[1]
  l = pars[2]
  k = pars[3]
  m = pars[4]

  W1 = X[1]
  W2 = X[2]
  T1 = X[3]
  T2 = X[4]

  dW1 = -(g / l) * sin(T1) - (k / m) * (T1 - T2)
  dW2 = -(g / l) * sin(T2) + (k / m) * (T1 - T2)
  dT1 = W1
  dT2 = W2
  return Array([dW1, dW2, dT1, dT2])
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

Random.seed!(1)
nsamps = 1000

# pars = ('g' = 9.8, 'l' = 1, 'k' = 1, 'm' = 1)
pars = [9.8, 1, 1, 1]
X_start = [pi / 2, -pi / 4, -0.5, 0.2]
h = 0.01

Y = Array{Float64}(undef, 4, nsamps)
Y[:, 1] = rk(pendulum, X_start, pars, h)
for i in 2:nsamps
  Y[:, i] = rk(pendulum, Y[:, i-1], pars, h)
end


Y_prime = reduce(hcat, [pendulum(Y[:, i], pars)[1:2] for i in 1:nsamps])
Y = Y[3:4, :] # Y[1:2,:] are the first order derivatives, see pendulum() above

plot(Y')
plot(Y_prime')



###################### library ######################

ΛNames = ["Phi"]
function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  θ₁ = u[1,:]
  θ₂ = u[2,:]
  
  return [θ₁, θ₂, θ₁ .* θ₂, θ₁.^2, θ₂.^2, sin.(θ₁), sin.(θ₂), cos.(θ₁), cos.(θ₂)]

end

######################### run sampler #########################

nbasis = 200
TimeStep = Vector(range(0.01, size(Y,2)*0.01, step = 0.01))
batch_size = 50
buffer = 10
v0 = 1e-6
v1 = 1e4
learning_rate = 1e-8


model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 10000, orderTime = 2)

model

######################### output #########################

print_equation(["θ₁ₜ", "θ₂ₜ"], model, pars, posterior, cutoff_prob=0.95, p=0.95)


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

plot(posterior.M[1,1,:])
plot(posterior.M[1,6,:])

