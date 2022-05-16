


using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics


###################### data functions ######################
function lotka_volterra(X, pars)

    a = pars[1]
    b = pars[2]
    d = pars[3]
    g = pars[4]

    x = X[1]
    y = X[2]

    x_prime = a * x - b * x * y
    y_prime = d * x * y - g * y

    return Array([x_prime, y_prime])
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
X = [10, 10]
pars = [1.1, 0.4, 0.1, 0.4]
h = 0.1
end_time = 100
nsamps = convert(Int, (end_time / h))

Y = Array{Float64}(undef, 2, nsamps)
Y[:, 1] = X
for i in 2:nsamps
    Y[:, i] = rk(lotka_volterra, Y[:, i-1], pars, h)
end

Y_prime = [lotka_volterra(Y[:, i], pars) for i in 1:nsamps]
Y_prime = reduce(hcat, Y_prime)

######################### noisy data #########################
Random.seed!(1)
R = 1 * Matrix{Float64}(I, 2, 2)

Z_tmp = copy(Y) + rand(MvNormal([0.0; 0.0], R), nsamps)
Z = [(Z_tmp[i, j] < 0 ? 0 : Z_tmp[i, j]) for i in 1:2, j in 1:nsamps]

Z = Matrix{Float64}(Z)

###################### plot data ######################
# plt_df = DataFrame(Y', [:X, :Y])
# plt_df[!, :time] = [h:h:end_time;]

# plt_df = stack(plt_df, 1:2)

# gr()
# @df plt_df plot(
#     :time,
#     :value,
#     group = :variable,
#     layout = (2,1)
# )


# plot(Y[1,:], label = "Prey", lw = 3)
# plot!(Z[1,:], label = "", seriestype = :scatter)

# plot(Y[2,:], label = "Predator", lw = 3)
# plot!(Z[2,:], label = "", seriestype = :scatter)

###################### library ######################


ΛNames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  X = u[1,:]
  Y = u[2,:]
  
  return [X, Y, X .* Y, X.^2, Y.^2, X.^2 .* Y, X .* Y.^2, X.^3, Y.^3]

end


nbasis = 400
TimeStep = Vector(range(0.1, end_time, step = 0.1))
batch_size = 50
buffer = 20
v0 = 1e-6
v1 = 1e4
order = 1
learning_rate = 1e-2


model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 100)
model, pars, posterior = DEtection_sampler(Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 100)



print_equation(["Xₜ", "Yₜ"], model, pars, posterior, cutoff_prob=0.95, p=0.95)


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










function Lambda(x::Vector)
    X = x[1]
    Y = x[2]
    return ([X, Y, X * Y, X^2, Y^2, X^2 * Y, X * Y^2, X^3, Y^3])
end


######################### run sampler #########################
samps = DEtection_sampler(Z, 1e0, 1e0,
    nits = 1000,
    burnin = 500,
    nbasis = 400,
    h = 0.1,
    batch_size = 50,
    buffer = 20,
    v0 = 1e-6,
    v1 = 1e4,
    order = 1,
    latent_space = 2,
    Lambda)
#

par_names = ["X"; "Y"; "X*Y"; "X^2"; "Y^2"; "X^2*Y"; "X*Y^2"; "X^3"; "Y^3"]
sys_names = ["dX/dt"; "dY/dt"]
eqs = print_equation(M = samps['M'],
    M_prob = samps["gamma"],
    par_names = par_names,
    sys_names = sys_names,
    cutoff_prob = 0.9)
eqs["mean"]
eqs["lower"]
eqs["upper"]

