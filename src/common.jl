using Lux
using NPZ
using Random
using Statistics

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

function MLPBlock(;
    embedding_dim::Int,
    mlp_dim::Int,
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

    if ndims(x) == 4
        x = permutedims(x, (4, 1, 2, 3))
        x, _ = self.lin1(x, self.lin1_ps, self.lin1_st)
        x = permutedims(x, (2, 3, 4, 1))

        x = self.act.(x)

        x = permutedims(x, (4, 1, 2, 3))
        y, = self.lin2(x, self.lin2_ps, self.lin2_st)

        y = permutedims(y, (2, 3, 4, 1))
    end

    if ndims(x) == 3
        x = permutedims(x, (3, 1, 2))
        x, _ = self.lin1(x, self.lin1_ps, self.lin1_st)
        x = permutedims(x, (2, 3, 1))
        x = self.act.(x)
        x = permutedims(x, (3, 1, 2))
        y, = self.lin2(x, self.lin2_ps, self.lin2_st)
        y = permutedims(y, (2, 3, 1))
    end

    return y
end



######################################################
# LayerNorm2d
######################################################

# Viene definita la struttura LayerNorm2d
struct LayerNorm2d <: LuxCore.AbstractLuxLayer
    num_channels::Int
    eps::Float32
end


LayerNorm2d(num_channels::Int) = LayerNorm2d(num_channels, 1f-6)

function LuxCore.initialparameters(rng::AbstractRNG, l::LayerNorm2d)
    return (weight=ones(Float32, l.num_channels),
            bias=zeros(Float32, l.num_channels))
end

function Lux.apply(ln::LayerNorm2d, x, ps, st)
    x = permutedims(x, (4, 3, 1, 2))
    
    u = mean(x, dims=2)
    s = mean(((x .- u) .^ 2), dims=2)
    x = (x .- u) ./ sqrt.(s .+ ln.eps)

    weight_expanded = reshape(ps.weight, (1, :, 1, 1))
    bias_expanded = reshape(ps.bias, (1, :, 1, 1))
    
    x = weight_expanded .* x .+ bias_expanded
    x = permutedims(x, (3, 4, 2, 1))
    
    return x, st
end
