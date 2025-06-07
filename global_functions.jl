using SpecialFunctions
using LoopVectorization

# Funzione per la permutazione degli array in ordine inverso
function reverse_pos(a::AbstractArray)
    return permutedims(a, reverse(1:ndims(a)))
end

# Funzione per il reshape python-like
function sam_reshape(a, dims)
    b = PermutedDimsArray(a, reverse(1:ndims(a)))
    a = reshape(b, reverse(dims))
    a = permutedims(a, reverse(1:ndims(a)))
    return a
end

# Funzione per il calcolo gelu non approssimato
function gelu_exact(x)
    return 0.5f0 * x * (1f0 + erf(x / sqrt(2f0)))
end


function safe_imresize(arr, new_size; method=Linear())
    # Trova dimensioni non-singleton
    non_singleton_dims = findall(size(arr) .> 1)
    
    if length(non_singleton_dims) == length(new_size)
        # Estrai solo le dimensioni spaziali
        spatial_data = 
            dropdims(arr, dims = tuple(findall(size(arr) .== 1)...))
    
        # Ridimensiona
        resized = imresize(spatial_data, new_size, method=method)
        
        # Ricostruisci con dimensioni originali
        original_shape = collect(size(arr))
        original_shape[non_singleton_dims] .= new_size
        return reshape(resized, original_shape...)
    else
        error("Mismatch between non-singleton dimensions and target size")
    end
end


###########################################################
# Assegnazione pesi e bias dal checkpoint
###########################################################

function assign_weight!(model, path_str::String, value)
    parts = split(path_str, '.')
    obj = model

    for i = 1:length(parts)-1
        part = parts[i]

        # Gestione dell'indice (es: layers_ps[2])
        if occursin('[', part)
            base, idx_str = match(r"(\w+)\[(\d+)\]", part).captures
            idx = parse(Int, idx_str)
            obj = getproperty(obj, Symbol(base))[idx]
        else
            obj = getproperty(obj, Symbol(part))
        end
    end

    # Ultimo campo: potrebbe essere .weight, .bias, ecc.
    last_part = parts[end]
    if occursin('[', last_part)
        base, idx_str = match(r"(\w+)\[(\d+)\]", last_part).captures
        idx = parse(Int, idx_str)
        getproperty(obj, Symbol(base))[idx] .= value
    else
        getproperty(obj, Symbol(last_part)) .= value
    end
end



function load_model_weights!(
    model::Any, 
    state_dict::Dict{String, Array{Float32}}
    )

    for (k, v) in state_dict
        try
            assign_weight!(model, k, v)
        catch e
            @warn "Errore assegnando $k" exception = e
        end
    end
end


##################################################
# Funzione per facilitare i test
##################################################

function is_correct(string::String, x::AbstractArray, y::AbstractArray)
    println(
        string,
        all(isapprox.(
            x,
            y;
            atol = 0.4,
            rtol = 0)),
        "	 Differenza massima: ",
        maximum(abs.(x .- y)),
        )
end