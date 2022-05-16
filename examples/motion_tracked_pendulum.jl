


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
# function Lambda(x::Vector)
#     x = x[1]
#     return [x, sin(x), cos(x), x/sin(x), x/cos(x), x1, x1^2, x1*sin(x), x1*cos(x), sin(x1), cos(x1)]
# end


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
buffer = 10
v0 = 1e-6
v1 = 1e2
order = 2
learning_rate = 1e-8

model, pars, posterior = DEtection_sampler(Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames, nits = 10000, orderTime = 2, learning_rate_end = 1e-8)

print_equation(["Xₜₜ"], model, pars, posterior, cutoff_prob=0.95, p=0.95)


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
plot(posterior.M[1,6,:])

plot(posterior.ΣU[1,1,:])
plot(posterior.ΣV[1,1,:])


plot(posterior.A[1,20,:])





samps = DEtection_sampler(Y, 1e-4, 1e-8,
    nits = 1000,
    burnin = 500,
    nbasis = 250,
    h = 0.1666,
    batch_size = 20,
    buffer = 10,
    v0 = 1e-6,
    v1 = 1e4,
    order = 2,
    latent_space = 1,
    Lambda)
#

par_names = ["H"; "L"; "H*L"; "H^2"; "L^2"; "H^2*L"; "H*L^2"; "H^3"; "L^3"]
sys_names = ["dH/dt"; "dL/dt"]
eqs = print_equation(M = samps['M'],
    M_prob = samps["gamma"],
    par_names = par_names,
    sys_names = sys_names,
    cutoff_prob = 0.99)
eqs["mean"]
eqs["lower"]
eqs["upper"]


plot(samps['M'][:1, :1, :], linecolor = "black")
plot(samps['M'][:1, :2, :], linecolor = "black")
plot(samps['M'][:1, :3, :], linecolor = "black")
plot(samps['M'][:2, :1, :], linecolor = "black")
plot(samps['M'][:2, :2, :], linecolor = "black")
plot(samps['M'][:2, :3, :], linecolor = "black")


function prep_lambda()
    
end


function create_pars(Y::Matrix{Float64}, nbasis::Int, h::Float64, batch_size::Int, learning_rate::Float64, buffer::Int, v0::Float64, v1::Float64, order::Int, latent_space::Int, Lambda::Function)

    pars = Dict()
    pars['Y'] = copy(Y)
    pars['m'] = size(pars['Y'], 1)
    pars['n'] = latent_space
  
    # data parameters
    pars["Time"] = size(pars['Y'], 2)
    pars['h'] = h
    pars["nbasis"] = nbasis
    pars["batch_size"] = batch_size
    pars["learning_rate"] = learning_rate
    pars["buffer"] = buffer
  
    # x = [pars['h']:pars['h']:(pars['h']*pars["Time"]);]
    x = range(pars['h'], pars['h'] * pars["Time"], length = pars["Time"])
    knot_locs = range(2 * pars['h'], stop = pars['h'] * pars["Time"] - 2 * pars['h'], length = nbasis)
    degree = 3
    @rput x
    @rput knot_locs
    @rput degree
    @rput order
  
    btmp = R"splines2::bSpline(x, knots = knot_locs, degree, intercept = TRUE)"
    btmpprime = R"splines2::dbs(x, knots = knot_locs, degree, derivs = order, intercept = TRUE)"
  
    pars["Phi"] = rcopy(btmp)
    pars["Phi_prime"] = rcopy(btmpprime)
  
    pars["Time_na"] = copy(pars["Time"])
  
    if pars['n'] - pars['m'] > 0
      # alpha_add = zeros(size(pars["Phi"],2), (pars['n'] - pars['m']))
      alpha_add = rand(Normal(0.0, 1), size(pars["Phi"], 2), (pars['n'] - pars['m']))
      pars['A'] = hcat(inv(pars["Phi"]' * pars["Phi"]) * pars["Phi"]' * pars['Y']', alpha_add)
    else
      pars['A'] = inv(pars["Phi"]' * pars["Phi"]) * pars["Phi"]' * pars['Y']'
    end
  
    pars['X'] = (pars["Phi"] * pars['A'])'
    pars["X_prime"] = (pars["Phi_prime"] * pars['A'])'
    pars['P'] = size(Lambda(pars['X'][:, 1]), 1)
  
    H = ones(pars['m'], pars["Time"])
    pars['H'] = [make_H(pars['m'], pars['n'], H[:, i]) for i in 1:pars["Time"]]
  
    # parameters
    pars['M'] = zeros(pars['n'], pars['P'])
    pars['R'] = Matrix{Float64}(I, pars['m'], pars['m'])
    pars['Q'] = Matrix{Float64}(I, pars['n'], pars['n'])
    pars["sigma_r"] = ones(pars['m'])
  
    # hyperpriors
    pars["a_R"] = ones(pars['m'])
    pars["A_R"] = fill(1e6, pars['m'])
    pars["nu_R"] = 2
  
    pars["a_Q"] = ones(pars['n'])
    pars["A_Q"] = fill(1e6, pars['n'])
    pars["nu_Q"] = 2
  
    # SSVS parameters
    pars["v0"] = v0
    pars["v1"] = v1
    pars["gamma"] = ones(pars['n'], pars['P'])
    pars["Sigma_M"] = Diagonal([(vec(pars["gamma"])[i] == 1 ? pars["v1"] : pars["v0"]) for i in 1:(pars['n']*pars['P'])])
  
    pars["Lambda"] = Lambda
  
    # for prediction
    return pars
  end