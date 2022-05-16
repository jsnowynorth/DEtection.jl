



using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics



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


# ###################### plot data ######################
# plt_df = DataFrame(Z_missing', [:X, :Y, :Z])
# plt_df[!, :time] = [h:h:end_time;]

# plt_df = stack(plt_df, 1:3)

# # gr()
# @df plt_df plot(
#     :time,
#     :value,
#     group = :variable,
#     layout = (3,1)
# )


###################### library ######################
# function Lambda(x::Vector)
#   X = x[1]
#   Y = x[2]
#   Z = x[3]
#   # M * [1, X, Y, Z, X*Y, X*Z, Y*Z, X^2, Y^2, Z^2, X^2*Y, X*Y^2, X^2*Z, X*Z^2, Y^2*Z, Y*Z^2, X^3, Y^3, Z^3, X*Y*Z]
#   return ([X, Y, Z, X * Y, X * Z, Y * Z, X^2, Y^2, Z^2, X^2 * Y, X * Y^2, X^2 * Z, X * Z^2, Y^2 * Z, Y * Z^2, X^3, Y^3, Z^3, X * Y * Z])
# end

ΛNames = ["Phi"]

function Λ(A, Φ)
    
  ϕ = Φ[1]
  u = A * ϕ'

  X = u[1,:]
  Y = u[2,:]
  Z = u[3,:]
  
  return [X, Y, Z, X .* Y, X .* Z, Y .* Z, X.^2, Y.^2, Z.^2, X.^2 .* Y, X .* Y.^2, X.^2 .* Z, X .* Z.^2, Y.^2 .* Z, Y .* Z.^2, X.^3, Y.^3, Z.^3, X .* Y .* Z]

end

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
model, pars, posterior = DEtection_sampler(Z_missing, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 1000)


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

pars.M .* pars.gamma

plot(pars.State.State["U"][1,:])
plot!(Y[1,:])

plot(pars.State.State["U"][2,:])
plot!(Y[2,:])

plot(pars.State.State["U"][3,:])
plot!(Y[3,:])



V = collect(Missings.replace(Z_missing, 0)) # convert to Float64
na_inds = hcat(getindex.(findall(ismissing, Z_missing), 1), getindex.(findall(ismissing, Z_missing), 2))

V[na_inds[:,1], na_inds[:,2]] = V[na_inds[:,1], na_inds[:,2] .- 1]
Z_missing[na_inds[:,1], na_inds[:,2]]

plot(V')

VM = copy(Z_missing)

tmp_ind = copy(na_inds)
if tmp_ind[1,2] == 1
  tmp_ind[1,2] = 3
end
tmp_ind[:,2] = tmp_ind[:,2] .- 1

VM[na_inds[:,1], na_inds[:,2]] = VM[tmp_ind[:,1], tmp_ind[:,2]]
still_missing = hcat(getindex.(findall(ismissing, VM), 1), getindex.(findall(ismissing, VM),2))
while size(still_missing, 1) > 1
  tmp_ind = copy(still_missing)
  tmp_ind[:,2] = tmp_ind[:,2] .- 1
  VM[still_missing[:,1], still_missing[:,2]] = VM[tmp_ind[:,1], tmp_ind[:,2]]
  still_missing = hcat(getindex.(findall(ismissing, VM), 1), getindex.(findall(ismissing, VM),2))
end

plot(VM')

Matrix(VM)


# samps = DEtection_sampler(Z_missing, 1e1, 1e1,
#   nits = 10000,
#   burnin = 5000,
#   nbasis = 400,
#   TimeStep = 0.01,
#   batch_size = 50,
#   buffer = 20,
#   v0 = 1e-6,
#   v1 = 1e4,
#   order = 1,
#   latent_space = 3,
#   Λ = Λ)
# #

# par_names = ["X"; "Y"; "Z"; "X*Y"; "X*Z"; "Y*Z"; "X^2"; "Y^2"; "Z^2"; "X^2*Y"; "X*Y^2"; "X^2*Z"; "X*Z^2"; "Y^2*Z"; "Y*Z^2"; "X^3"; "Y^3"; "Z^3"; "X*Y*Z"]
# sys_names = ["dX/dt"; "dY/dt"; "dZ/dt"]
# eqs = print_equation(M = samps['M'],
#   M_prob = samps["gamma"],
#   par_names = par_names,
#   sys_names = sys_names,
#   cutoff_prob = 0.99)
# #
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

# plot([h:h:h*nsamps;], X_rec[:3, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y[:3, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:1, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:1, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:2, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:2, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:3, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:3, :], linecolor = "black")



# ######################### run sgmcmc missing component #########################
# # run = sgmcmc(3,
# #               nits = 2000,
# #               burnin = 1000,
# #               Y = Y[2:3,:],
# #               nbasis = 200,
# #               h = 0.01,
# #               batch_size = 50,
# #               learning_rate = 1e0,
# #               decay_end = 1e-1,
# #               buffer = 20,
# #               v0 = 1e-6,
# #               v1 = 1e4)

# function Lambda(x::Vector)
#   X = x[1]
#   Y = x[2]
#   Z = x[3]
#   return ([X, Y, Z, X * Y, X * Z, Y * Z, X^2, Y^2, Z^2])
# end

# # run = sgmcmc(;nits = 5010,
# #               burnin = 10,
# #               Y = Z[2:3,:],
# #               nbasis = 200,
# #               h = 0.01,
# #               batch_size = 50,
# #               learning_rate = 1e2,
# #               decay_end = 1e1,
# #               buffer = 20,
# #               v0 = 1e-6,
# #               v1 = 1e4,
# #               order = 1,
# #               latent_space = 3)

# # save_object("/Users/joshuanorth/Desktop/lorenz_missing_system.jld2", run)
# # run = load_object("/Users/joshuanorth/Desktop/julia_results/lorenz_missing_system.jld2")

# # run = sgmcmc(;nits = 5000,
# #               burnin = 1000,
# #               Y = Y[2:3,:],
# #               nbasis = 200,
# #               h = 0.01,
# #               batch_size = 50,
# #               learning_rate = 1e2,
# #               decay_end = 1e1,
# #               buffer = 20,
# #               v0 = 1e-6,
# #               v1 = 1e4,
# #               order = 1,
# #               latent_space = 3)


# # save_object("/Users/joshuanorth/Desktop/lorenz_noise.jld2", run)
# # run = load_object("/Users/joshuanorth/Desktop/lorenz_noise.jld2")

# # par_names = ["X"; "Y"; "Z"; "X*Y"; "X*Z"; "Y*Z"; "X^2"; "Y^2"; "Z^2"; "X^2*Y"; "X*Y^2"; "X^2*Z"; "X*Z^2"; "Y^2*Z"; "Y*Z^2"; "X^3"; "Y^3"; "Z^3"; "X*Y*Z"]
# # par_names = ["X"; "Y"; "Z"; "X*Y"; "X*Z"; "Y*Z"; "X^2"; "Y^2"; "Z^2"]
# par_names = ["Y"; "Z"; "X"; "Y*Z"; "X*Y"; "X*Z"; "Y^2"; "Z^2"; "X^2"]
# # sys_names = ["dX/dt"; "dY/dt"; "dZ/dt"]
# sys_names = ["dY/dt"; "dZ/dt"; "dX/dt"]
# eqs = print_equation(M = run['M'][:, :, 3000:5000],
#   M_prob = run["gamma"][:, :, 3000:5000],
#   par_names = par_names,
#   sys_names = sys_names,
#   cutoff_prob = 0.99)
# eqs["mean"]
# eqs["lower"]
# eqs["upper"]


# round.(mean(run['M'], dims = 3)[:, :, 1], digits = 3)
# round.(mean(run["gamma"], dims = 3), digits = 3)

# A_hat = mean(run['A'], dims = 3)[:, :, 1]

# X_rec = (run["Phi"] * A_hat)'
# X_prime_rec = (run["Phi_prime"] * A_hat)'

# plot([h:h:h*nsamps;], X_rec[:1, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y[:2, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_rec[:2, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y[:3, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_rec[:3, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y[:1, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:1, :], linecolor = "blue", ylims = (-150, 150))
# plot!([h:h:h*nsamps;], Y_prime[:2, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:2, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:3, :], linecolor = "black")

# plot([h:h:h*nsamps;], X_prime_rec[:3, :], linecolor = "blue")
# plot!([h:h:h*nsamps;], Y_prime[:1, :], linecolor = "black")

# plot(run['M'][:3, :1, :], linecolor = "black")
# plot(run['M'][:3, :3, :], linecolor = "black")
# plot(run['M'][:1, :3, :], linecolor = "black")
# plot(run['M'][:1, :1, :], linecolor = "black")
# plot(run['M'][:1, :6, :], linecolor = "black")
# plot(run['M'][:2, :2, :], linecolor = "black")
# plot(run['M'][:2, :5, :], linecolor = "black")

# histogram(run['M'][:3, :1, :])
# histogram(run['M'][:3, :3, :])
# histogram(run['M'][:1, :3, :])
# histogram(run['M'][:1, :1, :])
# histogram(run['M'][:1, :6, :])
# histogram(run['M'][:2, :2, :])
# histogram(run['M'][:2, :5, :])


# round.(mean(run['R'], dims = 3), digits = 3)
# round.(mean(run['Q'], dims = 3), digits = 3)


# anim = @animate for i in 1:50:3991
#   ind = i
#   X_rec = (run["Phi"] * run['A'][:, :, ind])'

#   l = @layout [a; b; c; d]

#   eqs = print_equation(M = run['M'][:, :, ind],
#     M_prob = run["gamma"][:, :, ind],
#     par_names = par_names,
#     sys_names = sys_names,
#     cutoff_prob = 0.95)

#   p1 = plot(xlims = (0, 10), ylims = (-10, 10), grid = false, ticks = false, axis = false)
#   p1 = annotate!(5, 5, text(eqs["mean"][1]))
#   p1 = annotate!(5, 0, eqs["mean"][2])
#   p1 = annotate!(5, -5, eqs["mean"][3])
#   p1 = annotate!(5, 10, "Iteration " * string(i))


#   p2 = plot([h:h:h*nsamps;], X_rec[:1, :], linecolor = "blue", ylims = (-25, 25), label = "Estimate", legend = false)
#   p2 = plot!([h:h:h*nsamps;], Y[:2, :], linecolor = "black", label = "Truth", legend = false)

#   p3 = plot([h:h:h*nsamps;], X_rec[:2, :], linecolor = "blue", ylims = (5, 50), label = "Estimate", legend = false)
#   p3 = plot!([h:h:h*nsamps;], Y[:3, :], linecolor = "black", label = "Truth", legend = false)

#   p4 = plot([h:h:h*nsamps;], X_rec[:3, :], linecolor = "blue", ylims = (-30, 30), label = "Estimate", legend = false)
#   p4 = plot!([h:h:h*nsamps;], Y[:1, :], linecolor = "black", label = "Truth", legend = false)


#   plot(p1, p2, p3, p4, layout = l, size = (800, 600))

# end

# # gif(anim, "/Users/joshuanorth/Desktop/learning_missing_component.gif", fps = 5)
# gif(anim, "/Users/joshuanorth/Desktop/learning_missing_component_noise.gif", fps = 5)
