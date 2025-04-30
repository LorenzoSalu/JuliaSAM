using SpecialFunctions

# Funzione per la permutazione degli array in ordine inverso
function reverse_pos(a::AbstractArray)
    return permutedims(a, Tuple(ndims(a):-1:1))
end

# Funzione per il reshape python-like
function sam_reshape(a, dims)
    a = reverse_pos(a)
    a = reshape(a, reverse(dims))
    a = reverse_pos(a)
    return a
end

# Funzione per il calcolo gelu non approssimato
function gelu_exact(x)
    return 0.5f0 * x * (1f0 + erf(x / sqrt(2f0)))
end