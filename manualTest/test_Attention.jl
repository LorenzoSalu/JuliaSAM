using Lux
using CUDA
using Interpolations
using TensorOperations
using Einsum
using NPZ
using Random, Random123
using Combinatorics


#include("../modeling/image_encoder.jl")

#=
########################################################
# Recupero dati:
########################################################
=#

# Recupero i dati di input
integer_parameters = 
    npzread("./manualTest/testFiles/Attention_integer_parameters.npy")

dim = integer_parameters[1]
num_heads = integer_parameters[2]

boolean_parameters = 
    npzread("./manualTest/testFiles/Attention_boolean_parameters.npy")

qkv_bias = boolean_parameters[1]
use_rel_pos = boolean_parameters[2]
rel_pos_zero_init = boolean_parameters[3]

input_size = 
    tuple(npzread("./manualTest/testFiles/Attention_input_size.npy")...)

x = 
    npzread("./manualTest/testFiles/Attention_input_x.npy")

# Recupero i dati expected

expected_qkv_weights = 
    npzread("./manualTest/testFiles/Attention_qkv_weights.npy")

expected_qkv_bias = 
    npzread("./manualTest/testFiles/Attention_qkv_bias.npy")

expected_qkv_x = 
    npzread("./manualTest/testFiles/Attention_qkv_x.npy")

expected_qkv_x_flatten = 
    npzread("./manualTest/testFiles/Attention_qkv_x_flatten.npy")

expected_qkv_x_reshaped = 
    npzread("./manualTest/testFiles/Attention_qkv_x_reshaped.npy")

expected_qkv_x_reshaped_perm = 
    npzread("./manualTest/testFiles/Attention_qkv_x_reshaped_perm.npy")

expected_second_qkv_reshaped = 
    npzread("./manualTest/testFiles/Attention_second_qkv_reshaped.npy")

expected_q = 
    npzread("./manualTest/testFiles/Attention_expected_q.npy")

expected_k = 
    npzread("./manualTest/testFiles/Attention_expected_k.npy")

expected_v = 
    npzread("./manualTest/testFiles/Attention_expected_v.npy")



########################################################
# Funzione:
########################################################

struct Attention
    dim::Int
    num_heads::Int
    scale::Float64
    qkv::Dense
    proj::Dense
    use_rel_pos::Bool
    rel_pos_h::Union{Nothing, Matrix{Float32}}
    rel_pos_w::Union{Nothing, Matrix{Float32}}
end

function Attention(;
    dim::Int,
    num_heads::Int = 8,
    qkv_bias::Bool = true,
    use_rel_pos::Bool = false,
    rel_pos_zero_init::Bool = true,
    input_size::Union{Nothing, Tuple{Int, Int}} = nothing
    )
    
    head_dim = dim ÷ num_heads
    scale = 1 / sqrt(head_dim)

    qkv = Dense(dim, dim * 3, use_bias=qkv_bias, init_weight = kaiming_uniform)

    proj = Dense(dim, dim, init_weight = kaiming_uniform)

    rel_pos_h, rel_pos_w = nothing, nothing

    if use_rel_pos
        @assert input_size !== nothing 
        "Input size must be provided if using relative positional encoding."

        rel_pos_h = 
            zeros(Float32, 2 * input_size[1] - 1, head_dim)
        rel_pos_w = 
            zeros(Float32, 2 * input_size[2] - 1, head_dim)
    end
    
    return Attention(
        dim,
        num_heads, 
        scale, 
        qkv, 
        proj, 
        use_rel_pos, 
        rel_pos_h, 
        rel_pos_w
        )
end


function (m::Attention)(x::AbstractArray)
    B, H, W, _ = size(x)

    #qkv 

    return x
end



########################################################
# Test sulla funzione:
########################################################


attention = Attention(
    dim = dim,
    num_heads = num_heads,
    qkv_bias = qkv_bias,
    use_rel_pos = use_rel_pos,
    rel_pos_zero_init = rel_pos_zero_init,
    input_size = input_size,
)

self_qkv = attention.qkv
self_num_heads = attention.num_heads
self_scale = attention.scale
self_use_rel_pos = attention.use_rel_pos
self_rel_pos_h = attention.rel_pos_h
self_rel_pos_w = attention.rel_pos_w
self_proj = attention.proj

B, H, W, C = size(x)
head_dim = div(C, self_num_heads)

rng = Random.MersenneTwister()

ps, st = Lux.setup(rng, self_qkv)

ps.weight .= expected_qkv_weights # Questa è solo per fare i test
ps.bias .= expected_qkv_bias # Questa è solo per fare i test

# Applicazione layer lineares
x_flat = reshape(x, C, :)'  # (C, B*H*W)
qkv_out, _ = self_qkv(x_flat, ps, st) # Output (3*num_heads*head_dim, B*H*W)

#qkv_out = qkv_out' # (B*H*W, 3*num_heads*head_dim)

#display(vec(qkv_out))
#display(expected_qkv_x_flatten)

display(vec(permutedims(x, (4, 3, 2, 1))))



println(
    "qkv(x) corretto? ",
    isapprox(
        vec(permutedims(expected_qkv_x, (4, 3, 2, 1))), 
        expected_qkv_x_flatten;
        rtol=1e-6)
    )

println(
    "qkv(x) corretto? ",
    isapprox(vec(qkv_out), vec(expected_qkv_x); rtol=1e-6)
    )

println(
    "qkv(x) flat corretto? ",
    isapprox(vec(qkv_out), expected_qkv_x_flatten; rtol=1e-6)
    )

qkv_out_reshaped = reshape(vec(qkv_out), (:, self_num_heads, 3, H*W, B))
qkv_out_reshaped = permutedims(qkv_out_reshaped, (5, 4, 3, 2, 1))


println(
    "qkv_out reshaped corretto? ",
    isapprox(vec(qkv_out_reshaped), vec(expected_qkv_x_reshaped); rtol=1e-6)
    )

#=
println(
    "q corretti? ",
    isapprox(q, expected_q, atol=1e-6)
    )


function find_elements(arr1, arr2; atol=1e-6)
    results = Bool[]
    for elem in arr1
        found = any(x -> isapprox(x, elem; atol=atol), arr2)
        push!(results, found)
    end
    return results
end
    
println(all(find_elements(expected_q, k)))


println(
    "weights corretti? ",
    isapprox(ps.weight, expected_qkv_weights, atol=1e-6)
    )

println(
    "Bias corretti? ",
    isapprox(ps.bias, expected_qkv_bias, atol=1e-6)
    )


y, _ = self_qkv(permutedims(x, (4, 1, 2, 3)), ps, st)
y = permutedims(y, (2, 3, 4, 1))

println(
    "qkv(x) corretto? ",
    isapprox(y, expected_qkv_x; rtol=1e-6)
    )

y_reshaped = permutedims(reshape(
    permutedims(y, (1, 2, 3, 4)),
    B,
    H * W,
    3,
    self_num_heads,
    :
    ), (1, 2, 3, 4, 5))


size(y_reshaped)
size(expected_qkv_x_reshaped)


println(
    "y reshaped corretto? ",
    isapprox(y_reshaped, expected_qkv_x_reshaped, atol=1e-6)
    )

println(
    "y reshaped vec corretto? ",
    isapprox(vec(y_reshaped), vec(expected_qkv_x_reshaped), atol=1e-6)
    )


qkv = permutedims(reshape(
    permutedims(y_reshaped, (1, 2, 3, 4, 5)),
    3, 
    B * self_num_heads,
    H * W,
    :
    ), (1, 2, 3, 4))



function find_elements(arr1, arr2; atol=1e-6)
    results = Bool[]
    for elem in arr1
        found = any(x -> isapprox(x, elem; atol=atol), arr2)
        push!(results, found)
    end
    return results
end

#println(all(find_elements(expected_qkv_x_reshaped_perm, y_reshaped)))
#println(all(find_elements(expected_second_qkv_reshaped, qkv)))

counter = 0
for (a, b) in zip(vec(y_reshaped), vec(expected_qkv_x_reshaped))
    global counter
    if !isapprox(a, b; rtol=1e-6)
        counter += 1
    end
end

println(counter)

counter = 0
for (a, b) in zip(vec(y_reshaped), vec(expected_qkv_x_reshaped_perm))
    global counter
    if !isapprox(a, b; rtol=1e-6)
        counter += 1
    end
end

println(counter)


counter = 0
for (a, b) in zip(vec(qkv), vec(expected_second_qkv_reshaped))
    global counter
    if !isapprox(a, b; rtol=1e-6)
        counter += 1
    end
end

println(counter)

=#