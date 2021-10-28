function common_noncommon_indexes(collectionA, collectionB)
    index_common = findin(collectionA, collectionB)
    index_noncommon = find_out(collectionA, collectionB)
    return index_common, index_noncommon

end

function common_noncommon(collectionA, collectionB)
    index_com_noncom = common_noncommon_indexes(collectionA, collectionB)
    things_in_common = collectionA[index_com_noncom[1]]
    things_notin_common = collectionA[index_com_noncom[2]]
    return things_in_common, things_notin_common

end


function join_columns(d1::DataFrame, d2::DataFrame; on = :feature_name)

    f_common = common_noncommon(d1[:feature_name], d2[:feature_name])[1]

    d1_index = findin(d1[:feature_name], f_common)
    d2_index = findin(d2[:feature_name], f_common)

    d1 = sort(d1[d1_index, :], cols = on)
    d2 = sort(d2[d2_index, :], cols = on)

    cols2 = names(d2)

    if d1[:feature_name] == d2[:, :feature_name]
        for col in cols2
            d1[col] = d2[col]
        end
        return d1
    else
        println("Oh... no equal, find the problem")
    end

end

function col_types_dict(dataframe::DataFrame)
    Dict(names(dataframe), eltypes(dataframe))
end

function transform_in_matrix(dataframe::DataFrame)
    typecols = col_types_dict(dataframe)
    data_allow = [Float64, Float32, Int, Int64, Int32]
    new_cols = Array{Symbol, 1}()

    for col in keys(typecols)
        if in(typecols[col], data_allow)
            push!(new_cols, col)
        end
    end

    sort!(new_cols)

    dataframe = dataframe[:, new_cols]

    matrix = Array(dataframe)

    return [matrix, new_cols]
end

function euclidian_node_array(node_vector_array)               # Calculates eucledian distances between an array nodes x dimensions and returns a matrix of euclidian distances between nodes.
    count_nodes = size(node_vector_array,1)
    euclidean_array = zeros(count_nodes, count_nodes)            # Empty array to store euclidian distances between nodes. Its columns and raws numbers should be equal the number of nodes
    for node1 in 1:count_nodes
        node1_dimension_values = node_vector_array[node1, :]
        for node2 in 1:count_nodes
            node2_dimension_values = node_vector_array[node2, :]
            euclidean_value = euclidean(node1_dimension_values, node2_dimension_values)            # Calculate euclidean distance between 2 nodes
            euclidean_array[node1, node2] = euclidean_value    # Add the euclidean value into the array
        end
    end
    return euclidean_array
end


function dimensionality_reduction(method::String, array::Matrix, outdim; k=100, t::Int=1, ɛ::Float64=1.0, noise=0.001:0.00001:0.002, initial_dims = size(array)[2], iterations=1000, perplexity=20)
    if method == "PCA"
        pca_obtention = fit(PCA, transpose(array), maxoutdim=outdim)
        pca_network = transpose( MultivariateStats.transform(pca_obtention, transpose(array)))
        println("PCA obtained; $outdim dimensions")
        return pca_network

        elseif method == "ICA"
        ica_obtention = fit(ICA, transpose(array), outdim)
        ica_network = transpose( MultivariateStats.transform(ica_obtention, transpose(array)))
        println("ICA obtained; $outdim dimensions")
        return ica_network

        elseif method == "MDS"
        mds_network = transpose(classical_mds(euclidian_node_array(array), outdim))
        println("MDS obtained; $outdim dimensions")
        return mds_network

        elseif method == "Isomap"
        isomap_network_map = MultivariateStats.transform(Isomap, transpose(array), k=k, d=outdim) #2D
        isomap_network = transpose(projection(isomap_network_map))
        println("Isomap obtained; $outdim dimensions")
        return isomap_network

        elseif method == "DiffMap"
        diffmap_network_map = MultivariateStats.transform(DiffMap, full(transpose(array)), d=outdim, t=t, ɛ=ɛ)
        diffmap_network = transpose(projection(diffmap_network_map))
        println("Diffusion Map obtained; $outdim dimensions")
        return diffmap_network

        elseif method == "noise+DiffMap"    #this bit is for when Diffuion Maps is giving a lapack error. This is because you cannot have to many equal values to create a diffusion map
        noise_network = similar(array)

        for i in eachindex(array)
            noise_network[i] = array[i] + rand(noise, 1)[1]
        end

        diffmap_network_map = MultivariateStats.transform(DiffMap, full(transpose(noise_network)), d=outdim, t=t, ɛ=ɛ)
        diffmap_network = transpose(projection(diffmap_network_map))
        println("Diffusion Map obtained; $outdim dimensions")
        return diffmap_network

        elseif method =="tSNE"
        tsne_network = tsne(array, outdim, initial_dims, iterations, perplexity, progress = false)
        println("tSNE obtained; $outdim dimensions")
        return tsne_network

    else
        println("This method $method is not implemented in this function")
    end
end

function dimensionality_reduction(method::String, df::DataFrame, outdim; k=100, t::Int=1, ɛ::Float64=1.0, noise=0.001:0.00001:0.002,
    initial_dims = size(df)[2], iterations=1000, perplexity=20, transpose = true)

    if transpose == true
        array = transform_in_matrix(df)
        d = array[1]
        d = d'
    else
        array = transform_in_matrix(df)
        d = array[1]
    end

    reduction =  dimensionality_reduction(method, d, outdim,
    k=k, t=t, ɛ=ɛ, noise=noise, initial_dims = initial_dims, iterations=iterations, perplexity=perplexity)
    samples = array[2]

    return reduction, samples
end

function dimensionality_reduction(method::String, df::DataFrame, outdim, colData::DataFrame; k=100, t::Int=1, ɛ::Float64=1.0, noise=0.001:0.00001:0.002,
    initial_dims = size(df)[2], iterations=1000, perplexity=20, transpose = true)

    reduction =  dimensionality_reduction(method, df, outdim,
    k=k, t=t, ɛ=ɛ, noise=noise, initial_dims = initial_dims, iterations=iterations, perplexity=perplexity, transpose = transpose)

    samples = [string(i) for i in reduction[2]]

    dataframe = DataFrame()

    dataframe[:name] = samples

    for i in 1:outdim
        comp = Symbol(string("Component_", i))
        dataframe[comp] = reduction[1][:, i]
    end

    dataframe = join(dataframe, colData, on = :name, kind = :inner)

    return dataframe
end



function dimensionality_reduction_model(method::String, array::Array, outdim::Int)
    if method == "PCA"
        pca_model = fit(PCA, transpose(array), maxoutdim=outdim)
        return pca_model

        elseif method == "ICA"
        ica_model = fit(ICA, transpose(array), outdim)
        return ica_model
    end
end

function dimensionality_reduction_model(method::String, df::DataFrame, outdim::Int; transpose = true)

    if transpose == true
        array = transform_in_matrix(df)
        d = array[1]
        d = d'
    else
        array = transform_in_matrix(df)
        d = array[1]
    end

    reduction =  dimensionality_reduction_model(method, d, outdim)

    return reduction
end
"""
export plot_component_variance

function plot_component_variance(PCA_model::PCA)
    variances_components = principalvars(PCA_model)
    total_variance = tvar(PCA_model)
    perc_variance_component = [i/total_variance for i in variances_components]
    data = DataFrame()
    data[:Percentage_Of_Variance_Preserved] = perc_variance_component
    data[:Principal_Component] = 1:length(perc_variance_component)
    plot(data, y="Percentage_Of_Variance_Preserved", x="Principal_Component", Geom.bar)
end

export PCA_projection_features

function PCA_projection_features(PCA_Model::PCA, initial_data::DataFrame)
    projections = projection(PCA_Model)
    n, dims = size(projections)

    dataframe = select_string_columns(initial_data)

    for i in 1:dims
        comp = Symbol(string("Component_", i))
        dataframe[comp] = projections[:, i]
    end

    return dataframe

end

"""

