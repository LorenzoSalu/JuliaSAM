using Lux
using NPZ
using Random

include("../global_functions.jl")


######################################################
# MLPBlock
######################################################

# Viene definita la struttura NLPBlock
# viene utilizzata per la gestione di un percettrone multilayer
struct MLPBlock
    lin1::Dense
    lin1_ps::NamedTuple
    lin1_st::NamedTuple
    lin2::Dense
    lin2_ps::NamedTuple
    lin2_st::NamedTuple
    act::Function
end

function MLPBlock(
    embedding_dim::Int,
    mlp_dim::Int;
    act::Function = gelu_exact
    )

    rng = Random.MersenneTwister()

    lin1 = Dense(embedding_dim => mlp_dim, init_weight = kaiming_uniform)
    lin1_ps, lin1_st = Lux.setup(rng, lin1)

    lin2 = Dense(mlp_dim => embedding_dim, init_weight = kaiming_uniform)
    lin2_ps, lin2_st = Lux.setup(rng, lin2)

    # Per effettuare test decommentare
    ####################################################################
    #lin1_ps.weight .= test_lin1_weight
    #lin1_ps.bias .= test_lin1_bias

    #lin2_ps.weight .= test_lin2_weight 
    #lin2_ps.bias .= test_lin2_bias
    ####################################################################

    return MLPBlock(
        lin1,
        lin1_ps, lin1_st,
        lin2,
        lin2_ps, lin2_st,
        act
        )
end

function (self::MLPBlock)(x::AbstractArray)::AbstractArray
    x, _ = self.lin1(x', self.lin1_ps, self.lin1_st)
    x = x'

    x = self.act.(x)

    y, = self.lin2(x', self.lin2_ps, self.lin2_st)
    y = y'

    return y
end



######################################################
# LayerNorm2d
######################################################

# Viene definita la struttura LayerNorm2d
struct LayerNorm2d
    weight::Union{Nothing, Vector{Float32}}
    bias::Union{Nothing, Vector{Float32}}
    eps::Float32
end

function LayerNorm2d(
    num_channels::Int,
    eps::Float32 = 1e-6f0
    )

    weight = ones(Float32, num_channels)
    bias = zeros(Float32, num_channels)

    # Per effettuare test decommentare
    #####################################
    #weight = test_weight
    #bias = test_bias
    #####################################

    return LayerNorm2d(
        weight,
        bias,
        eps    
    )
end

function(self::LayerNorm2d)(x::AbstractArray)::AbstractArray
    u = mean(x, dims=2)
    s = mean(((x .- u) .^ 2), dims=2)
    x = (x .- u) ./ sqrt.(s .+ eps)

    weight_expanded = reshape(self.weight, (1, :, 1, 1))
    bias_expanded = reshape(self.bias, (1, :, 1, 1))

    x = weight_expanded .* x .+ bias_expanded
    return x
end
