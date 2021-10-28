module DimensionalityReduction

using DataFrames
# using ManifoldLearning, Gadfly
using TSne, MultivariateStats
using Distances

export join_columns, col_types_dict, transform_in_matrix
export dimensionality_reduction
export dimensionality_reduction_model

include("Functions.jl")

end # module
