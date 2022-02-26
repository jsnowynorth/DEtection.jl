
function make_H(m, n, one_inds)
  H_tmp = zeros(m, n)
  H_tmp[diagind(H_tmp)] .= one_inds
  return H_tmp
end

# base
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

function create_pars(Y::Matrix{Float64}, nbasis::Int, h::Float64, batch_size::Int, learning_rate::Vector{Float64}, buffer::Int, v0::Float64, v1::Float64, order::Int, latent_space::Int, Lambda::Function)

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

  x = [pars['h']:pars['h']:(pars['h']*pars["Time"]);]
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

# missing data
function create_pars(Y::Matrix{Union{Missing,Float64}}, nbasis::Int, h::Float64, batch_size::Int, learning_rate::Float64, buffer::Int, v0::Float64, v1::Float64, order::Int, latent_space::Int, Lambda::Function)

  pars = Dict()
  pars['Y'] = copy(Y)
  pars['m'] = size(pars['Y'], 1)
  pars['n'] = latent_space
  pars["na_inds"] = hcat(getindex.(findall(ismissing, Y), 1), getindex.(findall(ismissing, Y), 2))

  tmp_ind = copy(pars["na_inds"])
  if tmp_ind[1, 2] == 1
    tmp_ind[1, 2] = 3
  end
  tmp_ind[:, 2] = tmp_ind[:, 2] .- 1
  pars['Y'][pars["na_inds"][:, 1], pars["na_inds"][:, 2]] = pars['Y'][tmp_ind[:, 1], tmp_ind[:, 2]]

  still_missing = hcat(getindex.(findall(ismissing, pars['Y']), 1), getindex.(findall(ismissing, pars['Y']), 2))
  while size(still_missing, 1) > 1
    tmp_ind = copy(still_missing)
    tmp_ind[:, 2] = tmp_ind[:, 2] .- 1
    pars['Y'][still_missing[:, 1], still_missing[:, 2]] = pars['Y'][tmp_ind[:, 1], tmp_ind[:, 2]]
    still_missing = hcat(getindex.(findall(ismissing, pars['Y']), 1), getindex.(findall(ismissing, pars['Y']), 2))
  end

  # data parameters
  pars["Time"] = size(pars['Y'], 2)
  pars['h'] = h
  pars["nbasis"] = nbasis
  pars["batch_size"] = batch_size
  pars["learning_rate"] = learning_rate
  pars["buffer"] = buffer

  x = [pars['h']:pars['h']:(pars['h']*pars["Time"]);]
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
  if (size(pars["na_inds"])[1] != 0)
    H[pars["na_inds"]] .= 0
  end
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

  return pars
end

function create_pars(Y::Matrix{Union{Missing,Float64}}, nbasis::Int, h::Float64, batch_size::Int, learning_rate::Vector{Float64}, buffer::Int, v0::Float64, v1::Float64, order::Int, latent_space::Int, Lambda::Function)

  pars = Dict()
  pars['Y'] = copy(Y)
  pars['m'] = size(pars['Y'], 1)
  pars['n'] = latent_space
  pars["na_inds"] = hcat(getindex.(findall(ismissing, Y), 1), getindex.(findall(ismissing, Y), 2))

  tmp_ind = copy(pars["na_inds"])
  if tmp_ind[1, 2] == 1
    tmp_ind[1, 2] = 3
  end
  tmp_ind[:, 2] = tmp_ind[:, 2] .- 1
  pars['Y'][pars["na_inds"][:, 1], pars["na_inds"][:, 2]] = pars['Y'][tmp_ind[:, 1], tmp_ind[:, 2]]

  still_missing = hcat(getindex.(findall(ismissing, pars['Y']), 1), getindex.(findall(ismissing, pars['Y']), 2))
  while size(still_missing, 1) > 1
    tmp_ind = copy(still_missing)
    tmp_ind[:, 2] = tmp_ind[:, 2] .- 1
    pars['Y'][still_missing[:, 1], still_missing[:, 2]] = pars['Y'][tmp_ind[:, 1], tmp_ind[:, 2]]
    still_missing = hcat(getindex.(findall(ismissing, pars['Y']), 1), getindex.(findall(ismissing, pars['Y']), 2))
  end

  # data parameters
  pars["Time"] = size(pars['Y'], 2)
  pars['h'] = h
  pars["nbasis"] = nbasis
  pars["batch_size"] = batch_size
  pars["learning_rate"] = learning_rate
  pars["buffer"] = buffer

  x = [pars['h']:pars['h']:(pars['h']*pars["Time"]);]
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
  if (size(pars["na_inds"])[1] != 0)
    H[pars["na_inds"]] .= 0
  end
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

  return pars
end


function update_M!(pars)

  X = copy(pars['X'])
  X_prime = copy(pars["X_prime"])
  Q = copy(pars['Q'])
  n = copy(pars['n'])
  P = copy(pars['P'])
  Sigma_M = copy(pars["Sigma_M"])
  Time = copy(pars["Time"])
  buffer = copy(pars["buffer"])
  Lambda = pars["Lambda"]

  X = X[:, (buffer):(Time-buffer-1)]
  X_prime = X_prime[:, (buffer):(Time-buffer-1)]
  Time = size(X, 2)

  f = mapslices(Lambda, X, dims = 1)

  F_tilde = kronecker(f', I(n))
  Q_tilde = kronecker(I(Time), inv(Q))

  V = F_tilde' * Q_tilde * F_tilde + inv(Sigma_M)
  a = F_tilde' * Q_tilde * vec(X_prime)
  C = cholesky(Hermitian(V))
  b = rand(Normal(0.0, 1.0), n * P)

  # permutedims(reshape(C.U \ ((C.L \ a) + b), (P, n)), [2,1])
  pars['M'] = reshape(C.U \ ((C.L \ a) + b), (n, P))

  return pars
end

function update_gamma!(pars)

  n = copy(pars['n'])
  P = copy(pars['P'])
  M = copy(pars['M'])
  v0 = copy(pars["v0"])
  v1 = copy(pars["v1"])
  gamma = copy(pars["gamma"])

  p1 = pdf.(Normal(0, sqrt(v1)), M)
  p0 = pdf.(Normal(0, sqrt(v0)), M)

  p = p1./(p1+p0)
  p = replace!(p, NaN=>0)

  samp = [rand(Binomial(1, p[i,j])) for i in 1:n, j in 1:P]

  pars["gamma"] = [samp[i,j] == 1 ? 1 : 0 for i in 1:n, j in 1:P]
  pars["Sigma_M"] = Diagonal([(vec(pars["gamma"])[i] == 1 ? pars["v1"] : pars["v0"]) for i in 1:(pars['n']*pars['P'])])

  return pars
end

function update_R!(pars)

  Y = copy(pars['Y'])
  X = copy(pars['X'])
  H = copy(pars['H'])
  m = copy(pars['m'])
  a_R = copy(pars["a_R"])
  A_R = copy(pars["A_R"])
  nu_R = copy(pars["nu_R"])
  # na_inds = copy(pars["na_inds"])
  Time_na = pars["Time_na"]
  Time = copy(pars["Time"])
  buffer = copy(pars["buffer"])

  sse_tmp = [Y[:,j] - H[j] * X[:,j] for j in buffer:(Time-buffer-1)]
  sse_tmp = reduce(hcat, sse_tmp)
  Time = size(sse_tmp)[2]
  n_time_points = Time_na

  a_hat = (n_time_points + nu_R)/2
  for i in 1:m
    b_hat = (nu_R/a_R[i]) + 0.5*sum(sse_tmp[i,:].^2)
    pars['R'][i,i] = rand(InverseGamma(a_hat, b_hat))
  end

  a_hat = (nu_R + 1)/2
  for i in 1:m
    b_hat = (nu_R/pars['R'][i,i]) + (1/A_R[i]^2)
    pars["a_R"][i] = rand(InverseGamma(a_hat, b_hat))
  end

  return pars
end

function update_Q!(pars)

  X = copy(pars['X'])
  X_prime = copy(pars["X_prime"])
  M = copy(pars['M'])
  n = copy(pars['n'])
  a_Q = copy(pars["a_Q"])
  A_Q = copy(pars["A_Q"])
  nu_Q = copy(pars["nu_Q"])
  Time = copy(pars["Time"])
  buffer = copy(pars["buffer"])
  Lambda = pars["Lambda"]

  X = X[:, (buffer):(Time-buffer-1)]
  X_prime = X_prime[:, (buffer):(Time-buffer-1)]
  Time = size(X, 2)

  f = mapslices(Lambda, X, dims = 1)

  nu_hat = nu_Q + n + Time - 1
  Phi_hat = 2 * nu_Q * Diagonal(1 ./ a_Q) + (X_prime - M * f) * (X_prime - M * f)'

  pars['Q'] = rand(InverseWishart(nu_hat, cholesky(Hermitian(Phi_hat))))

  a_hat = (nu_Q + n) / 2
  for i in 1:n
    b_hat = (nu_Q / pars['Q'][i, i]) + (1 / A_Q[i]^2)
    pars["a_Q"][i] = rand(InverseGamma(a_hat, b_hat))
  end

  return pars
end

function Q_grad(y, H, phi, phi_prime, A, M, R_inv, Q_inv, fprime, f_curr)

  y = reshape(y, (size(y)[1], 1))
  phi = reshape(phi, (1, size(phi)[1]))
  phi_prime = reshape(phi_prime, (1, size(phi_prime)[1]))
  f_curr = reshape(f_curr, (size(f_curr)[1], 1))

  Q_new = - phi' * y' * R_inv * H +
      phi' * phi * A * H' * R_inv * H +
      phi_prime' * phi_prime * A * Q_inv -
      phi_prime' * f_curr' * M' * Q_inv -
      phi' * phi_prime * A * Q_inv * fprime +
      phi' * f_curr' * M' * Q_inv * fprime

    return Q_new
end

function update_A!(pars)

  Y = copy(pars['Y'])
  X = copy(pars['X'])
  X_prime = copy(pars["X_prime"])
  A = copy(pars['A'])
  M = copy(pars['M'])
  Q = copy(pars['Q'])
  R = copy(pars['R'])
  H = copy(pars['H'])
  Phi = copy(pars["Phi"])
  Phi_prime = copy(pars["Phi_prime"])
  Lambda = pars["Lambda"]

  Time = copy(pars["Time"])
  # na_inds = copy(pars["na_inds"])
  n = copy(pars['n'])
  m = copy(pars['m'])
  lr = pars["learning_rate"]
  batch_size = pars["batch_size"]
  buffer = pars["buffer"]

  # inds = rand(1:Time, batch_size)
  inds = rand(buffer:(Time-buffer), batch_size)

  y = disallowmissing(Y[:, inds])
  x = disallowmissing(X[:, inds])
  H_sub = H[inds]
  phi = Phi[inds, :]
  phi_prime = Phi_prime[inds, :]
  R_inv = inv(R)
  Q_inv = inv(Q)

  f_curr = mapslices(Lambda, x, dims = 1)
  # fprime = [dLambda(x[:,i], M) for i in 1:size(inds)[1]]
  fprime = [M * ForwardDiff.jacobian(Lambda, x[:, i]) for i in 1:size(inds)[1]]

  dq = [Q_grad(y[:, i], H_sub[i], phi[i, :], phi_prime[i, :], A, M, R_inv, Q_inv, fprime[i], f_curr[:, i]) for i in 1:size(inds)[1]]
  dq = reduce(+, dq)

  scale_el = 1 / 100
  dq = (1 / batch_size) * (dq - (1 / Time) * (scale_el) * Matrix{Float64}(I, size(A)[1], size(A)[1]) * A - 2 * (scale_el) * sign.(A))

  if lr isa Vector
    pars['A'] = A - transpose(lr .* dq')
  else
    pars['A'] = A - lr * dq
  end
  # pars['A'] = A - lr * dq

  pars['X'] = (pars["Phi"] * pars['A'])'
  pars["X_prime"] = (pars["Phi_prime"] * pars['A'])'

  return pars
end

# base
function DEtection_sampler(Y::Matrix{Float64}, learning_rate::Float64, decay_end::Float64, Lambda::Function;
  nits::Int = 2000, burnin::Int = 1000, nbasis::Int = 400, h::Float64 = 0.1, batch_size::Int = 50, buffer::Int = 10, v0::Float64 = 1e-6, v1::Float64 = 1e4, order::Int = 1, latent_space::Int = 3)

  pars = create_pars(Y, nbasis, h, batch_size, learning_rate, buffer, v0, v1, order, latent_space, Lambda)

  keep_samps = nits - burnin
  n_change = log10(learning_rate / decay_end)
  if n_change == 0
    change_times = 1.0
  else
    change_times = floor.(collect(range(1, stop = burnin, length = Int((n_change + 1)))))
  end

  M_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)
  Q_post = Array{Float64}(undef, pars['n'], pars['n'], keep_samps)
  R_post = Array{Float64}(undef, pars['m'], pars['m'], keep_samps)
  A_post = Array{Float64}(undef, size(pars['A'])[1], pars['n'], keep_samps)
  gamma_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)

  @showprogress 1 "Burnin..." for i in 1:burnin
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    if i in change_times
      raise_pow = findall(change_times .== i)[1] - 1
      pars["learning_rate"] = learning_rate * 10^(-Float64(raise_pow))
    end

  end

  @showprogress 1 "Sampling..." for i in 1:keep_samps
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    M_post[:, :, i] = pars['M']
    Q_post[:, :, i] = pars['Q']
    R_post[:, :, i] = pars['R']
    A_post[:, :, i] = pars['A']
    gamma_post[:, :, i] = pars["gamma"]
  end

  out = Dict()
  out['M'] = M_post
  out['Q'] = Q_post
  out['R'] = R_post
  out['A'] = A_post
  out["gamma"] = gamma_post
  out["Phi"] = pars["Phi"]
  out["Phi_prime"] = pars["Phi_prime"]

  return out
end

function DEtection_sampler(Y::Matrix{Float64}, learning_rate::Vector{Float64}, decay_end::Vector{Float64}, Lambda::Function;
  nits::Int = 2000, burnin::Int = 1000, nbasis::Int = 400, h::Float64 = 0.1, batch_size::Int = 50, buffer::Int = 10, v0::Float64 = 1e-6, v1::Float64 = 1e4, order::Int = 1, latent_space::Int = 3)

  pars = create_pars(Y, nbasis, h, batch_size, learning_rate, buffer, v0, v1, order, latent_space, Lambda)

  keep_samps = nits - burnin
  n_learns = length(learning_rate)
  n_change = log10.(learning_rate ./ decay_end)
  if any(n_change .== 0)
    change_times = fill(1.0, n_learns)
  else
    change_times = map(x -> floor.(collect(range(1, stop = burnin, length = Int(x + 1)))), n_change)
  end


  M_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)
  Q_post = Array{Float64}(undef, pars['n'], pars['n'], keep_samps)
  R_post = Array{Float64}(undef, pars['m'], pars['m'], keep_samps)
  A_post = Array{Float64}(undef, size(pars['A'])[1], pars['n'], keep_samps)
  gamma_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)

  @showprogress 1 "Burnin..." for i in 1:burnin
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    if any(i .∈ change_times)
      which_vec = findall((i .∈ change_times) .== 1)
      raise_pow = map(x -> findall(change_times[x] .== i)[1] - 1, which_vec)
      learning_rate[which_vec] = learning_rate[which_vec] .* 10 .^ (-Float64.(raise_pow))
      pars["learning_rate"] = learning_rate
    end

  end

  @showprogress 1 "Sampling..." for i in 1:keep_samps
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    M_post[:, :, i] = pars['M']
    Q_post[:, :, i] = pars['Q']
    R_post[:, :, i] = pars['R']
    A_post[:, :, i] = pars['A']
    gamma_post[:, :, i] = pars["gamma"]
  end

  out = Dict()
  out['M'] = M_post
  out['Q'] = Q_post
  out['R'] = R_post
  out['A'] = A_post
  out["gamma"] = gamma_post
  out["Phi"] = pars["Phi"]
  out["Phi_prime"] = pars["Phi_prime"]

  return out
end

# missing
function DEtection_sampler(Y::Matrix{Union{Missing,Float64}}, learning_rate::Float64, decay_end::Float64, Lambda::Function;
  nits::Int = 2000, burnin::Int = 1000, nbasis::Int = 400, h::Float64 = 0.1, batch_size::Int = 50, buffer::Int = 10, v0::Float64 = 1e-6, v1::Float64 = 1e4, order::Int = 1, latent_space::Int = 3)

  pars = create_pars(Y, nbasis, h, batch_size, learning_rate, buffer, v0, v1, order, latent_space, Lambda)

  keep_samps = nits - burnin
  n_change = log10(learning_rate / decay_end)
  if n_change == 0
    change_times = 1.0
  else
    change_times = floor.(collect(range(1, stop = burnin, length = Int((n_change + 1)))))
  end

  M_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)
  Q_post = Array{Float64}(undef, pars['n'], pars['n'], keep_samps)
  R_post = Array{Float64}(undef, pars['m'], pars['m'], keep_samps)
  A_post = Array{Float64}(undef, size(pars['A'])[1], pars['n'], keep_samps)
  gamma_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)

  @showprogress 1 "Burnin..." for i in 1:burnin
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    if i in change_times
      raise_pow = findall(change_times .== i)[1] - 1
      pars["learning_rate"] = learning_rate * 10^(-Float64(raise_pow))
    end

  end

  @showprogress 1 "Sampling..." for i in 1:keep_samps
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    M_post[:, :, i] = pars['M']
    Q_post[:, :, i] = pars['Q']
    R_post[:, :, i] = pars['R']
    A_post[:, :, i] = pars['A']
    gamma_post[:, :, i] = pars["gamma"]
  end

  out = Dict()
  out['M'] = M_post
  out['Q'] = Q_post
  out['R'] = R_post
  out['A'] = A_post
  out["gamma"] = gamma_post
  out["Phi"] = pars["Phi"]
  out["Phi_prime"] = pars["Phi_prime"]

  return out
end

function DEtection_sampler(Y::Matrix{Union{Missing,Float64}}, learning_rate::Vector{Float64}, decay_end::Vector{Float64}, Lambda::Function;
  nits::Int = 2000, burnin::Int = 1000, nbasis::Int = 400, h::Float64 = 0.1, batch_size::Int = 50, buffer::Int = 10, v0::Float64 = 1e-6, v1::Float64 = 1e4, order::Int = 1, latent_space::Int = 3)

  pars = create_pars(Y, nbasis, h, batch_size, learning_rate, buffer, v0, v1, order, latent_space, Lambda)

  keep_samps = nits - burnin
  n_learns = length(learning_rate)
  n_change = log10.(learning_rate ./ decay_end)
  if any(n_change .== 0)
    change_times = fill(1.0, n_learns)
  else
    change_times = map(x -> floor.(collect(range(1, stop = burnin, length = Int(x + 1)))), n_change)
  end

  M_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)
  Q_post = Array{Float64}(undef, pars['n'], pars['n'], keep_samps)
  R_post = Array{Float64}(undef, pars['m'], pars['m'], keep_samps)
  A_post = Array{Float64}(undef, size(pars['A'])[1], pars['n'], keep_samps)
  gamma_post = Array{Float64}(undef, pars['n'], pars['P'], keep_samps)

  @showprogress 1 "Burnin..." for i in 1:burnin
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    if any(i .∈ change_times)
      which_vec = findall((i .∈ change_times) .== 1)
      raise_pow = map(x -> findall(change_times[x] .== i)[1] - 1, which_vec)
      learning_rate[which_vec] = learning_rate[which_vec] .* 10 .^ (-Float64.(raise_pow))
      pars["learning_rate"] = learning_rate
    end

  end

  @showprogress 1 "Sampling..." for i in 1:keep_samps
    pars = update_M!(pars)
    pars = update_gamma!(pars)
    pars = update_R!(pars)
    pars = update_Q!(pars)
    pars = update_A!(pars)

    M_post[:, :, i] = pars['M']
    Q_post[:, :, i] = pars['Q']
    R_post[:, :, i] = pars['R']
    A_post[:, :, i] = pars['A']
    gamma_post[:, :, i] = pars["gamma"]
  end

  out = Dict()
  out['M'] = M_post
  out['Q'] = Q_post
  out['R'] = R_post
  out['A'] = A_post
  out["gamma"] = gamma_post
  out["Phi"] = pars["Phi"]
  out["Phi_prime"] = pars["Phi_prime"]

  return out
end


# Y
# learning_rate = 1e0
# decay_end = 1e0
# nits = 10000
# burnin = 5000
# nbasis = 200
# h = 1/12
# batch_size = 20
# buffer = 5
# v0 = 1e-6
# v1 = 1e4
# order = 1
# latent_space = 10


# nits = 1000
# burnin = 500
# nbasis = 300
# h = 0.01
# batch_size = 50
# learning_rate = 1.0
# decay_end = 1e-5
# buffer = 20
# v0 = 1e-5
# v1 = 1e3
#
# run = sgmcmc(nits = 10000,
#               burnin = 5000,
#               Y = Y,
#               nbasis = 150,
#               h = 0.01,
#               batch_size = 50,
#               learning_rate = 1e-2,
#               decay_end = 1e-5,
#               buffer = 10,
#               v0 = 1e-5,
#               v1 = 1e2)
#
# mean(run['M'], dims = 3)
# mean(run["gamma"], dims = 3)
#
# X_rec = (run["Phi"] * reshape(mean(run['A'], dims = 3), 301, 3))'
# X_prime_rec = (run["Phi_prime"] * reshape(mean(run['A'], dims = 3), 301, 3))'
#
# plot([pars['h']:pars['h']:pars['h']*pars["Time"];], X_rec[:1,:])
# plot([pars['h']:pars['h']:pars['h']*pars["Time"];], X_prime_rec[:1,:])
#
#
# #### testing code
# X = copy(pars['X'])
# X_prime = copy(pars["X_prime"])
# Time = pars["Time"]
# buffer = pars["buffer"]
#
# X = X[:, buffer:(Time-buffer)]
# X_prime = X_prime[:, buffer:(Time-buffer)]
# X = disallowmissing(X)
#
# Lambda.(X)
# mapslices(Lambda, X, dims = 1)
#
#
# m = n = 3
# h = 0.01
# nbasis = 200
# batch_size = 20
# learning_rate = 1e-3
# buffer = 20
# v0 = 1e-5
# v1 = 1e2
# n_pred = 0
#
# pars = create_pars(Y = Z, nbasis = 400, h = 0.1, batch_size = 50, learning_rate = 1e1, buffer = 20, v0 = 1e-6, v1 = 1e4, order = 2, latent_space = 2)
# pars = create_pars(Z, 250, 0.01, 50, 1e-1, 20, 1e-6, 1e4, 2, 2)
# pars = update_M!(pars)
# pars = update_gamma!(pars)
# pars = update_R!(pars)
# pars = update_Q!(pars)
# pars = update_A!(pars)
#
#
# plot(20:1:980, pars["X_prime"][:1,20:1:980])
# plot!(20:1:980, Y_prime[:1,20:1:980])
# # plot([pars['h']:pars['h']:pars['h']*(pars["Time"]);], Y[:1,:])
# # plot!([pars['h']:pars['h']:pars['h']*(pars["Time"]);], pars['X'][:1,:])
# pars['M'] .* pars["gamma"]
#
#
# pars["learning_rate"] = 0.00001
