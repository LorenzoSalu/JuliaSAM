using Lux
using NPZ
using Random
using Statistics

include("../global_functions.jl")



"""
######################################################
# MLPBlock
######################################################

    struct MLPBlock

Feedforward MLP block composed of two fully connected layers with an activation function in between.

# Fields
- `lin1::Dense`: First dense (fully connected) layer projecting from embedding dimension to hidden MLP dimension.
- `lin1_ps::NamedTuple`: Parameters associated with `lin1`.
- `lin1_st::NamedTuple`: State associated with `lin1`.
- `lin2::Dense`: Second dense layer projecting back from MLP dimension to embedding dimension.
- `lin2_ps::NamedTuple`: Parameters associated with `lin2`.
- `lin2_st::NamedTuple`: State associated with `lin2`.
- `act::Function`: Activation function applied between the two dense layers (default: `gelu_exact`).

"""
struct MLPBlock
    lin1::Dense
    lin1_ps::NamedTuple
    lin1_st::NamedTuple
    lin2::Dense
    lin2_ps::NamedTuple
    lin2_st::NamedTuple
    act::Function
end


"""
    MLPBlock(; embedding_dim::Int, mlp_dim::Int, act::Function=gelu_exact)

Constructor for `MLPBlock`.

# Arguments
- `embedding_dim`: Dimensionality of the input embedding space.
- `mlp_dim`: Dimensionality of the intermediate hidden layer.
- `act`: Activation function to apply between the two layers (default: `gelu_exact`).

# Returns
- A new `MLPBlock` instance with initialized layers and parameters.
"""
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

    return MLPBlock(
        lin1,
        lin1_ps, lin1_st,
        lin2,
        lin2_ps, lin2_st,
        act
        )
end



"""
    (self::MLPBlock)(x::AbstractArray)::AbstractArray

Forward pass for the `MLPBlock`.

# Arguments
- `x::AbstractArray`: Input tensor, can be 3D or 4D:
    - 4D tensor shape assumed as (Batch, Height, Width, Channels).
    - 3D tensor shape assumed as (Batch, SequenceLength, Channels).

# Returns
- Output tensor of the same shape as input with features transformed by the two-layer MLP and activation.

# Description
- Permutes dimensions to channel-first format for dense layer compatibility.
- Applies the first dense layer (`lin1`), activation, then second dense layer (`lin2`).
- Restores original dimension ordering before returning.
"""
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




"""
######################################################
# LayerNorm2d
######################################################

    struct LayerNorm2d <: LuxCore.AbstractLuxLayer

2D Layer Normalization module applied over the channel dimension of a 4D input tensor.

# Fields
- `num_channels::Int`: Number of channels in the input tensor.
- `eps::Float32`: Small epsilon added for numerical stability during normalization.

"""
struct LayerNorm2d <: LuxCore.AbstractLuxLayer
    num_channels::Int
    eps::Float32
end

"""
    LayerNorm2d(num_channels::Int)

Constructor for `LayerNorm2d` with default epsilon value of `1e-6`.

# Arguments
- `num_channels`: Number of input channels.

# Returns
- A new `LayerNorm2d` instance.

"""
LayerNorm2d(num_channels::Int) = LayerNorm2d(num_channels, 1f-6)



"""
    LuxCore.initialparameters(rng::AbstractRNG, l::LayerNorm2d)

Initializes the parameters for the `LayerNorm2d` layer.

# Returns
- A named tuple with:
  - `weight`: Scale parameter initialized to ones of shape `(num_channels,)`.
  - `bias`: Bias parameter initialized to zeros of shape `(num_channels,)`.

"""
function LuxCore.initialparameters(rng::AbstractRNG, l::LayerNorm2d)
    return (weight=ones(Float32, l.num_channels),
            bias=zeros(Float32, l.num_channels))
end


"""
    Lux.apply(ln::LayerNorm2d, x, ps, st)

Applies layer normalization on a 4D input tensor `x` with shape `(Batch, Height, Width, Channels)`.

# Arguments
- `ln`: `LayerNorm2d` instance.
- `x`: Input tensor `(B, H, W, C)`.
- `ps`: Parameters named tuple containing `weight` and `bias`.
- `st`: State (not used here, returned unchanged).

# Returns
- Normalized tensor of the same shape as `x`.
- Unchanged state `st`.

# Description
- Permutes `x` to `(C, W, B, H)` to normalize over channels.
- Computes mean and variance over the spatial and batch dimensions.
- Normalizes and scales the input by `weight` and `bias` parameters.
- Permutes the tensor back to original `(B, H, W, C)` shape.

"""
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
