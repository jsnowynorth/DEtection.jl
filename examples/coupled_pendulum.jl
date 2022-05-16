

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics


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
Y = Y[3:4, :]

plot(Y')

plot(Y_prime')




# function Lambda(x::Vector)
#   X = x[1]
#   Y = x[2]
#   return ([X, Y, X * Y, X^2, Y^2, sin(X), sin(Y), cos(X), cos(Y)])
# end


ΛNames = ["Phi"]
function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  θ₁ = u[1,:]
  θ₂ = u[2,:]
  
  return [θ₁, θ₂, θ₁ .* θ₂, θ₁.^2, θ₂.^2, sin.(θ₁), sin.(θ₂), cos.(θ₁), cos.(θ₂)]

end


nbasis = 200
TimeStep = Vector(range(0.01, size(Y,2)*0.01, step = 0.01))
batch_size = 50
buffer = 10
v0 = 1e-6
v1 = 1e4
learning_rate = 1e-8


model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 1000, orderTime = 2)


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




# samps = DEtection_sampler(Y, 1e-8, 1e-8,
#   nits = 5000,
#   burnin = 1000,
#   nbasis = 200,
#   h = 0.01,
#   batch_size = 50,
#   buffer = 10,
#   v0 = 1e-6,
#   v1 = 1e4,
#   order = 2,
#   latent_space = 2,
#   Lambda)
# # 

# par_names = ["X"; "Y"; "X*Y"; "X^2"; "Y^2"; "sin(X)"; "sin(Y)"; "cos(X)"; "cos(Y)"]
# sys_names = ["dX/dt"; "dY/dt"]
# eqs = print_equation(M = samps['M'],
#   M_prob = samps["gamma"],
#   par_names = par_names,
#   sys_names = sys_names,
#   cutoff_prob = 0.99)
# eqs["mean"]
# eqs["lower"]
# eqs["upper"]


# round.(mean(samps['M'], dims = 3)[:, :, 1], digits = 3)
# round.(mean(samps["gamma"], dims = 3), digits = 3)


# A_hat = mean(samps['A'], dims = 3)[:, :, 1]

# X_rec = (samps["Phi"] * A_hat)'
# X_prime_rec = (samps["Phi_prime"] * A_hat)'

# plot([h:h:h*nsamps;], X_rec[:1, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y[:1, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_rec[:2, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y[:2, :], linecolor = "black")


# plot([30:1:980;], X_prime_rec[:1, 30:980], linecolor = "blue")
# plot!([30:1:980;], Y_prime[:1, 30:980], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:1, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:1, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:2, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:2, :], linecolor = "black")


# plot(samps['M'][:1, :1, :], linecolor = "black")
# plot(samps['M'][:1, :2, :], linecolor = "black")
# plot(samps['M'][:1, :6, :], linecolor = "black")
# plot(samps['M'][:2, :1, :], linecolor = "black")
# plot(samps['M'][:2, :2, :], linecolor = "black")
# plot(samps['M'][:2, :7, :], linecolor = "black")

# # plot(samps['A'][:100, :1, :], linecolor = "black")

# histogram(samps['M'][:1, :1, :])
# histogram(samps['M'][:1, :11, :])
# histogram(samps['M'][:2, :2, :])
# histogram(samps['M'][:2, :12, :])


# round.(mean(samps['R'], dims = 3), digits = 3)
# round.(mean(samps['Q'], dims = 3), digits = 3)

