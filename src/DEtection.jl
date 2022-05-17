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
using CodeTracking, Revise
# using Zygote



include("helper_funs.jl")
include("DE_sampler.jl")
include("process_sampler.jl")


export DEtection_sampler
export print_equation
export posterior_surface
export posterior_summary
export hpd


end # module
