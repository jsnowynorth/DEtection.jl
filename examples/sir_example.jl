


using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics

###################### create data ######################

function sir(X, pars)

    β = pars[1]
    γ = pars[2]

    S = X[1]
    I = X[2]
    R = X[3]

    N = S + I + R
    S_prime = -(β * I * S) / N
    I_prime = (β * I * S) / N - γ * I
    R_prime = γ * I
    return Array([S_prime, I_prime, R_prime])
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
X = [99, 1, 0]
pars = [15, 0.9]
h = 0.01
end_time = 5
nsamps = convert(Int, (end_time / h))

Y = Array{Float64}(undef, 3, nsamps)
Y[:, 1] = copy(X)
for i in 2:nsamps
    Y[:, i] = rk(sir, Y[:, i-1], pars, h)
end


######################### noisy data #########################

R = 1 * Matrix{Float64}(I, 3, 3)
Z = copy(Y) + rand(MvNormal([0.0; 0.0; 0.0], R), nsamps)
Z = [(Z[i, j] < 0 ? 0 : Z[i, j]) for i in 1:3, j in 1:nsamps]
Z = Matrix{Float64}(Z)

Y_prime = [sir(Y[:, i], pars) for i in 1:nsamps]
Z_prime = [sir(Z[:, i], pars) for i in 1:nsamps]
Y_prime = reduce(hcat, Y_prime)
Z_prime = reduce(hcat, Z_prime)


# plot(h:h:end_time, Y', label = ['S' 'I' 'R'], color = ["red" "purple" "green"], lw = 3, m = 2)
# scatter!(h:h:end_time, Z', primary = false, color = ["red" "purple" "green"])


###################### library ######################

ΛNames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  S = u[1,:]
  I = u[2,:]
  R = u[3,:]
  
  return [S, I, S .* I, S.^2, I.^2, S.^2 .* I, S .* I.^2, S.^3, I.^3]

end


######################### run sampler #########################

nbasis = 200
TimeStep = Vector(range(0.01, end_time, step = 0.01))
batch_size = 20
buffer = 10
v0 = 1e-8
v1 = 1e2
order = 1
learning_rate = 1e-3


model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 100)
model, pars, posterior = DEtection_sampler(Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 1000)



print_equation(["Sₜ", "Iₜ", "Rₜ"], model, pars, posterior, cutoff_prob=0.99, p=0.95)


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


plot(posterior.M[1,3,:])
plot(posterior.M[1,2,:])
plot(posterior.M[2,1,:])
plot(posterior.M[2,2,:])
plot(posterior.M[2,5,:])
plot(posterior.M[3,3,:])
plot(posterior.M[3,4,:])

plot(posterior.ΣU[1,1,:])
plot(posterior.ΣV[1,1,:])












function Lambda(x::Vector)
    S = x[1]
    I = x[2]
    R = x[3]
    return ([S, I, S * I, S^2, I^2, S^2 * I, S * I^2, S^3, I^3])
end

######################### run sgmcmc #########################
samps = DEtection_sampler(Z, 1e1, 1e1,
    nits = 10000,
    burnin = 5000,
    nbasis = 200,
    h = 0.01,
    batch_size = 20,
    buffer = 10,
    v0 = 1e-8,
    v1 = 1e2,
    order = 1,
    latent_space = 3,
    Lambda)
#

par_names = ["S"; "I"; "S*I"; "S^2"; "I^2"; "S^2*I"; "S*I^2"; "S^3"; "I^3"]
sys_names = ["dS/dt"; "dI/dt"; "dR/dt"]
eqs = print_equation(M = samps['M'],
    M_prob = samps["gamma"],
    par_names = par_names,
    sys_names = sys_names,
    cutoff_prob = 0.999)
eqs["mean"]
eqs["lower"]
eqs["upper"]


