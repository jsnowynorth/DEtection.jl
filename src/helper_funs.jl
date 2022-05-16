

"""
DerivativeClass

Creates the derivative class. Includes:

- TimeDerivative: Derivatives w.r.t. time
- TimeNames: Names of the time derivatives
- orderTime: Order of temporal derivative

"""
struct DerivativeClass
  TimeDerivative::Dict
  TimeNames::Vector{String}
  orderTime::Int
end


"""
StateClass

Creates the state class. Includes:

- State: Current state
- TimeState: Derivatives w.r.t. time
- TimeNames: Names of the time derivatives
- SpaceState: Derivatives w.r.t. space
- SpaceNames: Names of the space derivatives
- orderSpace: Order of spatial derivative
- orderTime: Order of temporal derivative
- spatialDimension: 1 if [x] 2 if [x,y]
- componentDimension: number of components, [u] or [u,v] or [u,v,w] ...

"""
# struct StateClass
#   State::Array{Float64}
#   TimeState::Dict
#   TimeNames::Vector{String}
#   SpaceState::Dict
#   SpaceNames::Vector{String}
#   orderSpace::Int
#   orderTime::Int
#   spatialDimension::Int
#   componentDimension::Int
# end
struct StateClass
  State::Dict
  StateNames::Vector{String}
end

"""
    bspline(x, knot_locs, degree, derivative)

Creates a matrix of B-Splines evaluated at each x with specified knot locations and degree.
Dimension is (length(x), length(x_knot_locs)).
Returns the basis matrix and the derivative of the basis matrix.

# Arguments
- x: data
- knot_locs: knot locations
- degree: degree of the B-Spline
- derivative: order of the derivative for Psi_x

# Examples
```
x = 1:1:30
knot_locs = 5:5:25
degree = 3
derivative = 1

Psi, Psi_x = bspline(x, knot_locs, degree, derivative)
```
"""
function bspline(x, knot_locs, degree, derivative)

  @rput x
  @rput knot_locs
  @rput degree
  @rput derivative
  # bxtmp = R"splines2::bSpline(x, knots = knot_locs, degree = degree, intercept = TRUE)"
  # dbxtmp = R"splines2::dbs(x, knots = knot_locs, degree = degree, derivs = derivative, intercept = TRUE)"

  # if derivative == 0
  #   btmp = R"splines2::bSpline(x, knots = knot_locs, degree = degree, intercept = TRUE)"
  # else
  #   btmp = R"splines2::dbs(x, knots = knot_locs, degree = degree, derivs = derivative, intercept = TRUE)"
  # end
  if derivative == 0
    btmp = R"splines2::bSpline(x, df = knot_locs, degree = degree, intercept = TRUE)"
  else
    btmp = R"splines2::dbs(x, df = knot_locs, degree = degree, derivs = derivative, intercept = TRUE)"
  end

  Phi = rcopy(btmp)
  # Phi = rcopy(bxtmp)
  # Phi_x = rcopy(dbxtmp)

  # return Phi, Phi_x
  return Phi
end


"""
    derivatives_dictionary(;orderx::Int, orderTime::Int, νT::Int, νS::Int,
                                Δx::Float64, Δt::Float64, degree::Int,
                                x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}},
                                Time::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}})

Creates a dictionary of space and time derivatives and their names. Used only for 1 spactial dimension.

# Arguments
- orderx::Int order of the x derivatives (spatial)
- orderTime::Int order of the time derivatives
- νT::Int number of time basis functions
- νS::Int number of space basis functions
- degree::Int degree of bsplines
- x::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}} x (spatial) values
- Time::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}} time values

- orderSpace::Int: order of the spatial derivatives
- orderTime::Int: order of the time derivative
- νS::Int, νS::Array{Int}: either number of spatial basis functions for 1 dimension or [x,y] dimensions
- νT::Int: number of time basis functions
- SpaceStep::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}: in 1 dim spatial indexes or [x,y] spatial indexes
- TimeStep::StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}}: temporal indexes
- degree::Int = 4: degree of bspline

# Examples
```
x = 2:0.05:7.5
y = -2.5:0.05:2.5
Time = 0:0.08:16
Δt = 0.08

νSx = 10
νSy = 10
νT = 10
degree = 4
orderTime = 1
orderSpace = 3

BasisDerivative = derivatives_dictionary(orderSpace, orderTime, [νSx, νSy], νT, [x, y], Time)
```
"""
function derivatives_dictionary(orderTime::Int,
  νT::Int,
  TimeStep::Vector;
  degree::Int = 4)

  Time_derivative = Dict()
  Time_derivative["Phi"] = bspline(TimeStep, νT, degree, 0)
  Time_names = ["Phi"]
  for i in 1:orderTime
    Time_derivative["Phi_".*repeat('t', i)] = bspline(TimeStep, νT, degree, i)
    push!(Time_names, "Phi_" .* repeat('t', i))
  end

  Time_names = vcat("Phi", ["Phi_" .* repeat('t', i) for i in 1:orderTime])

  BasisDerivative = DerivativeClass(Time_derivative, Time_names, orderTime)

  return BasisDerivative
end

"""
construct_state(;A::Array{Float64, 2},
                        Time_derivative::Dict, Time_names::Vector{String},
                        Space_derivative::Dict, Space_names::Vector{String})

Creates a dictionary and corresponding names of the space and time derivatives. Used only for 1 spatial dimension.

# Arguments
- A::Array{Float64, 2}
- BasisDerivative::DerivativeClass
- Theta: Dimension of components diagonal matrix

# Examples
```
ProcessDerivative = construct_state(A, BasisDerivative, Theta)
```

"""
function construct_state(A::Array{Float64,2}, BasisDerivative::DerivativeClass)

  orderTime = BasisDerivative.orderTime

  # initialize state dictionary and names
  State_process = Dict()
  
  # save base state
  State_process["U"] = A * BasisDerivative.TimeDerivative[BasisDerivative.TimeNames[1]]'
  State_names = ["U"]

  # save time derivatives
  for i in 1:orderTime
    State_process["ΔU_".*repeat('t', i)] = A * BasisDerivative.TimeDerivative[BasisDerivative.TimeNames[i+1]]'
    push!(State_names, "ΔU_" .* repeat('t', i))
  end

  ProcessDerivative = StateClass(State_process, State_names)

  return ProcessDerivative
end



"""
    create_create_library(Λ, Λnames, locs, State)

Create a matrix of the function Λ evaluated at each location

# Arguments
- Λ: function
- Λnames: names of each component of Λ
- llocs: space-time locations to evaluate the function at
- State: StateClass current state of values

"""

function create_Λ(Λ::Function, A, Basis, ΛTimeNames, X=nothing)

  if X === nothing
    Φ = [Basis.TimeDerivative[Basis.TimeNames[i]] for i in 1:Basis.orderTime]
    Λeval = reduce(hcat, Λ(A, Φ))
  else
    Φ = [Basis.TimeDerivative[Basis.TimeNames[i]] for i in 1:Basis.orderTime]
    Λeval = reduce(hcat, Λ(A, Φ, X))
  end

  return Λeval

end

function create_∇Λ(Λ::Function, A, Basis, samp_inds, ΛNames, X=nothing)

    N = size(A,1)
    Lname = length(ΛNames)

    if Lname > 2
        if X === nothing
            Φ = [reshape(Basis.TimeDerivative[Basis.TimeNames[i]][j,:],1,:) for i in 1:Lname, j in samp_inds]
            dΛeval = [jacobian(A -> Λ(A, Φ[:,i]), reshape(A[n,:],1,:)) for i in 1:length(samp_inds), n in 1:N]
        else
            Φ = [reshape(Basis.TimeDerivative[Basis.TimeNames[i]][j,:],1,:) for i in 1:Lname, j in samp_inds]
            dΛeval = [jacobian(A -> Λ(A, Φ[:,i], X[:, samp_inds[i]]), reshape(A[n,:],1,:)) for i in 1:length(samp_inds), n in 1:N]
        end
    else
        if X === nothing
            Φ = [Basis.TimeDerivative[Basis.TimeNames[i]][j,:] for i in 1:Lname, j in samp_inds]
            dΛeval = [jacobian(A -> Λ(A, Φ[i]), A[n,:]) for i in 1:length(samp_inds), n in 1:N]
        else
            Φ = [Basis.TimeDerivative[Basis.TimeNames[i]][j,:] for i in 1:Lname, j in samp_inds]
            dΛeval = [jacobian(A -> Λ(A, Φ[i], X[:, samp_inds[i]]), A[n,:]) for i in 1:length(samp_inds), n in 1:N]
        end
    end


  return dΛeval

end

