module DEtection

using StatsBase, LinearAlgebra, Plots, StatsPlots, CSV
using Kronecker
using Missings
using Distributions, Random
using ProgressMeter
using ReverseDiff: JacobianTape, JacobianConfig, jacobian, jacobian!, compile
using ForwardDiff
using Statistics
using JLD2
using RCall
using TexTables, KernelDensity
using DataFrames, DataFramesMeta, Chain
# @rlibrary ggplot2


# include("helper_funs.jl")
include("hpd.jl")
include("print_equation.jl")
include("sgmcmc_sampler.jl")
include("test_funs.jl")

export DEtection_sampler
export print_equation


end # module
