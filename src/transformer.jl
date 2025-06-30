using Lux
using Lux: Dense
using Lux: Conv
using CUDA
using Interpolations
using TensorOperations
using Einsum
using Random
using NNlib

include("../global_functions.jl")
include("../src/prompt_encoder.jl")
include("../src/common.jl")


"""
########################################################
# _Attention:
########################################################

	struct _Attention

Multi-head attention module with learned linear projections for queries, keys, 
	values, and outputs.

This struct implements a scaled dot-product multi-head attention mechanism. It 
	contains four linear projections:
	- `q_proj` projects the input to the query space,
	- `k_proj` projects the input to the key space,
	- `v_proj` projects the input to the value space,
	- `out_proj` projects the result of attention back to the original 
		embedding dimension.

# Fields
- `embedding_dim::Int`: The dimension of the input and output embeddings.
- `internal_dim::Int`: The total dimension of all attention heads combined 
	(`embedding_dim ÷ downsample_rate`).
- `num_heads::Int`: The number of attention heads.
- `q_proj::Dense`: Linear layer for projecting queries.
- `q_proj_ps::NamedTuple`: Parameters of `q_proj`.
- `q_proj_st::NamedTuple`: State of `q_proj`.
- `k_proj::Dense`: Linear layer for projecting keys.
- `k_proj_ps::NamedTuple`: Parameters of `k_proj`.
- `k_proj_st::NamedTuple`: State of `k_proj`.
- `v_proj::Dense`: Linear layer for projecting values.
- `v_proj_ps::NamedTuple`: Parameters of `v_proj`.
- `v_proj_st::NamedTuple`: State of `v_proj`.
- `out_proj::Dense`: Linear layer for projecting the attention output back to 
	the embedding dimension.
- `out_proj_ps::NamedTuple`: Parameters of `out_proj`.
- `out_proj_st::NamedTuple`: State of `out_proj`.
"""
struct _Attention
	embedding_dim::Int
	internal_dim::Int
	num_heads::Int
	q_proj::Dense
	q_proj_ps::NamedTuple
	q_proj_st::NamedTuple
	k_proj::Dense
	k_proj_ps::NamedTuple
	k_proj_st::NamedTuple
	v_proj::Dense
	v_proj_ps::NamedTuple
	v_proj_st::NamedTuple
	out_proj::Dense
	out_proj_ps::NamedTuple
	out_proj_st::NamedTuple
end

"""
	_Attention(;
		embedding_dim::Int,
		num_heads::Int,
		downsample_rate::Int = 1,
	)
		
Multi-head self-attention module with query, key, and value projections, 
	implemented using `Lux.Dense`.

This module allows attention across sequences by projecting inputs into 
	multi-head query, key, and value spaces, computing scaled dot-product 
	attention, and projecting the output back to the embedding dimension.

# Arguments
- `embedding_dim::Int`: Dimension of the input and output embeddings.
- `num_heads::Int`: Number of attention heads. Must divide `
	embedding_dim / downsample_rate`.
- `downsample_rate::Int=1`: Optional factor to reduce internal attention 
	dimensionality (e.g., for efficiency).

# Returns
- `_Attention`: A struct containing the attention projections, weights, and 
	internal parameters.
"""
function _Attention(;
	embedding_dim::Int,
	num_heads::Int,
	downsample_rate::Int = 1,
)

	embedding_dim = embedding_dim
	internal_dim = embedding_dim ÷ downsample_rate
	num_heads = num_heads

	@assert internal_dim % num_heads == 0 "num_heads must divide embedding_dim."

	rng = Random.MersenneTwister()

	q_proj = Dense(embedding_dim => internal_dim, init_weight = kaiming_uniform)
	q_proj_ps, q_proj_st = Lux.setup(rng, q_proj)

	k_proj = Dense(embedding_dim => internal_dim, init_weight = kaiming_uniform)
	k_proj_ps, k_proj_st = Lux.setup(rng, k_proj)

	v_proj = Dense(embedding_dim => internal_dim, init_weight = kaiming_uniform)
	v_proj_ps, v_proj_st = Lux.setup(rng, v_proj)

	out_proj = Dense(internal_dim => embedding_dim, init_weight = kaiming_uniform)
	out_proj_ps, out_proj_st = Lux.setup(rng, out_proj)

	return _Attention(
		embedding_dim,
		internal_dim,
		num_heads,
		q_proj,
		q_proj_ps, q_proj_st,
		k_proj,
		k_proj_ps, k_proj_st,
		v_proj,
		v_proj_ps, v_proj_st,
		out_proj,
		out_proj_ps, out_proj_st,
	)
end

"""
	(self::_Attention)(;
		q::AbstractArray,
		k::AbstractArray,
		v::AbstractArray,
	)::AbstractArray

Performs a forward pass of multi-head attention.

# Arguments
- `self::_Attention`: The attention module instance.
- `q::AbstractArray`: Query tensor of shape `(L, B, C)` — sequence length, 
	batch size, channels.
- `k::AbstractArray`: Key tensor of the same shape as `q`.
- `v::AbstractArray`: Value tensor of the same shape as `q`.

# Returns
- `AbstractArray`: Output tensor of shape `(L, B, embedding_dim)` after 
	attention computation.
"""
function (self::_Attention)(;
	q::AbstractArray,
	k::AbstractArray,
	v::AbstractArray,
)::AbstractArray

	q = reverse_pos(q)
	k = reverse_pos(k)
	v = reverse_pos(v)

	q, _ = self.q_proj(q, self.q_proj_ps, self.q_proj_st)
	k, _ = self.k_proj(k, self.k_proj_ps, self.k_proj_st)
	v, _ = self.v_proj(v, self.v_proj_ps, self.v_proj_st)

	q = reverse_pos(q)
	k = reverse_pos(k)
	v = reverse_pos(v)

	q = _separate_heads(q, self.num_heads)
	k = _separate_heads(k, self.num_heads)
	v = _separate_heads(v, self.num_heads)

	_, _, _, c_per_head = size(q)

	attn = permutedims(batched_mul(
			permutedims(q, (3, 4, 1, 2)),
			permutedims(k, (4, 3, 1, 2)),
		), (3, 4, 1, 2))

	attn = attn / sqrt(c_per_head)

	attn = softmax(attn, dims = ndims(attn))

	out = permutedims(batched_mul(
		permutedims(attn, (3, 4, 1, 2)),
		permutedims(v, (3, 4, 1, 2)),
	), (3, 4, 1, 2))

	out = _recombine_heads(out)

	out = reverse_pos(out)

	out, _ = self.out_proj(Float32.(out), self.out_proj_ps, self.out_proj_st)

	out = reverse_pos(out)

	return out
end

"""
	_separate_heads(
		x::AbstractArray,
		num_heads::Int,
	)::AbstractArray

Separates the last dimension of the tensor into multiple attention heads.

This function reshapes the input tensor so that the channel dimension is split 
	into `num_heads` separate attention heads, and rearranges the dimensions 
	for compatibility with batched attention computation.

# Arguments
- `x::AbstractArray`: Input tensor of shape `(B, N, C)`, where:
  - `B` is the batch size,
  - `N` is the number of tokens,
  - `C` is the embedding dimension.
- `num_heads::Int`: Number of attention heads to split the channel dimension 
	into. Must divide `C`.

# Returns
- `AbstractArray`: Output tensor of shape `(B, num_heads, N, C ÷ num_heads)`.
"""
function _separate_heads(
	x::AbstractArray,
	num_heads::Int,
)::AbstractArray

	b, n, c = size(x)
	x = sam_reshape(x, (b, n, num_heads, c ÷ num_heads))
	return permutedims(x, (1, 3, 2, 4))
end


"""

	_recombine_heads(x::AbstractArray)::AbstractArray

Recombines the attention heads into a single embedding dimension.

This function reverses the effect of `_separate_heads` by merging the per-head
channels and restoring the original order of tokens.

# Arguments
- `x::AbstractArray`: Input tensor of shape `(B, num_heads, N, C_per_head)`, 
	where:
	- `B` is the batch size,
	- `num_heads` is the number of attention heads,
	- `N` is the number of tokens,
	- `C_per_head` is the per-head channel size.

# Returns
- `AbstractArray`: Output tensor of shape `(B, N, num_heads * C_per_head)`.
"""
function _recombine_heads(x::AbstractArray)::AbstractArray

	b, n_heads, n_tokens, c_per_head = size(x)
	x = permutedims(x, (1, 3, 2, 4))
	return sam_reshape(x, (b, n_tokens, n_heads * c_per_head))
end


"""
########################################################
# TwoWayAttentionBlock:
########################################################

    struct TwoWayAttentionBlock

A transformer-style module that alternates self-attention, cross-attention, and 
	MLP operations to fuse and refine two interacting sets of tokens—typically 
	query and key/value embeddings (e.g. text and image).

# Fields
- `self_attn::_Attention`  
  Self-attention layer operating on the query tokens.
- `norm1::LayerNorm`, `norm1_ps`, `norm1_st`  
  LayerNorm and its parameter/state for post self-attention normalization.
- `cross_attn_token_to_image::_Attention`  
  Cross-attention where query tokens attend to image (key) embeddings.
- `norm2::LayerNorm`, `norm2_ps`, `norm2_st`  
  LayerNorm and its parameter/state for post cross-attention (tokens→image).
- `mlp::MLPBlock`  
  MLP block applied to refined query tokens.
- `norm3::LayerNorm`, `norm3_ps`, `norm3_st`  
  LayerNorm and its parameter/state for post-MLP normalization.
- `norm4::LayerNorm`, `norm4_ps`, `norm4_st`  
  LayerNorm and its parameter/state for post cross-attention (image→tokens).
- `cross_attn_image_to_token::_Attention`  
  Cross-attention where image embeddings attend to tokens.
- `skip_first_layer_pe::Bool`  
  If `true`, skips adding positional encoding in the first self-attention block.
"""
struct TwoWayAttentionBlock
	self_attn::_Attention
    norm1::LayerNorm
    norm1_ps::NamedTuple
    norm1_st::NamedTuple
    cross_attn_token_to_image::_Attention
    norm2::LayerNorm
    norm2_ps::NamedTuple
    norm2_st::NamedTuple
    mlp::MLPBlock
    norm3::LayerNorm
    norm3_ps::NamedTuple
    norm3_st::NamedTuple
    norm4::LayerNorm
    norm4_ps::NamedTuple
    norm4_st::NamedTuple
    cross_attn_image_to_token::_Attention
    skip_first_layer_pe::Bool
end

"""
    TwoWayAttentionBlock(; 
		embedding_dim, 
		num_heads, 
		mlp_dim=2048, 
		activation=relu, 
		attention_downsample_rate=2, 
		skip_first_layer_pe=false
	)

# Arguments
- `embedding_dim::Int`  
  Dimensionality of input embeddings.
- `num_heads::Int`  
  Number of attention heads.
- `mlp_dim::Int=2048`  
  Hidden dimensionality of the MLP block.
- `activation::Function=relu`  
  Activation function used in the MLP.
- `attention_downsample_rate::Int=2`  
  Downsampling rate for cross-attention projections.
- `skip_first_layer_pe::Bool=false`  
  If true, skips adding positional encodings in the first self-attention block.

# Returns
- `TwoWayAttentionBlock` instance.
"""
function TwoWayAttentionBlock(;
	embedding_dim::Int,
    num_heads::Int,
    mlp_dim::Int = 2048,
    activation::Function = relu,
    attention_downsample_rate::Int = 2,
    skip_first_layer_pe::Bool = false
)

    rng = Random.MersenneTwister()

	self_attn = _Attention(
        embedding_dim = embedding_dim, 
        num_heads = num_heads)

    norm1 = LayerNorm(embedding_dim, dims=1)
    norm1_ps, norm1_st = Lux.setup(rng, norm1)

    cross_attn_token_to_image = _Attention(
        embedding_dim = embedding_dim,
        num_heads = num_heads,
        downsample_rate = attention_downsample_rate
    )

    norm2 = LayerNorm(embedding_dim, dims=1)
    norm2_ps, norm2_st = Lux.setup(rng, norm2)

    mlp = MLPBlock(
        embedding_dim = embedding_dim, 
        mlp_dim = mlp_dim, 
        act = activation
        )

    norm3 = LayerNorm(embedding_dim, dims=1)
    norm3_ps, norm3_st = Lux.setup(rng, norm3)

    norm4 = LayerNorm(embedding_dim, dims=1)
    norm4_ps, norm4_st = Lux.setup(rng, norm4)

    cross_attn_image_to_token = _Attention(
        embedding_dim = embedding_dim,
        num_heads = num_heads,
        downsample_rate = attention_downsample_rate
    )

    skip_first_layer_pe = skip_first_layer_pe

	return TwoWayAttentionBlock(
		self_attn,
        norm1,
        norm1_ps,
        norm1_st,
        cross_attn_token_to_image,
        norm2,
        norm2_ps,
        norm2_st,
        mlp,
        norm3,
        norm3_ps,
        norm3_st,
        norm4,
        norm4_ps,
        norm4_st,
        cross_attn_image_to_token,
        skip_first_layer_pe
	)
end

"""
    (block::TwoWayAttentionBlock)(;
		queries, 
		keys, 
		query_pe, 
		key_pe) -> (updated_queries, updated_keys)

Applies the forward pass of the two-way attention block.

# Arguments
- `queries::AbstractArray`  
  Input query sequence, shape `(B, Nq, D)`.
- `keys::AbstractArray`  
  Input key/value sequence (e.g., image embeddings), shape `(B, Nk, D)`.
- `query_pe::AbstractArray`  
  Positional encoding to be added to `queries`, same shape.
- `key_pe::AbstractArray`  
  Positional encoding to be added to `keys`, same shape.

# Returns
- `(updated_queries, updated_keys)::Tuple{AbstractArray, AbstractArray}`  
  Refined embeddings for both `queries` and `keys` after attention and MLP.
"""
function (self::TwoWayAttentionBlock)(;
	queries::AbstractArray,
    keys::AbstractArray,
    query_pe::AbstractArray,
    key_pe::AbstractArray
)::Tuple{AbstractArray, AbstractArray}

    # Self attention block
    if self.skip_first_layer_pe
        queries = self.self_attn(q=queries, k=queries, v=queries)
    else
        q = queries + query_pe
        attn_out = self.self_attn(q=q, k=q, v=queries)
        queries = queries + attn_out
    end

    queries_dim = size(queries)
    queries = sam_reshape(queries, (:, queries_dim[end]))'
    queries, _ = self.norm1(queries, self.norm1_ps, self.norm1_st)
    queries = sam_reshape(queries', queries_dim)

    # Cross attention block, tokens attending to image embedding
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
    queries = queries + attn_out

    queries_dim = size(queries)
    queries = sam_reshape(queries, (:, queries_dim[end]))'
    queries, _ = self.norm2(queries, self.norm2_ps, self.norm2_st)
    queries = sam_reshape(queries', queries_dim)

    # MLP block
    mlp_out = self.mlp(queries)

    queries = queries + mlp_out

    queries_dim = size(queries)
    queries = sam_reshape(queries, (:, queries_dim[end]))'
    queries, _ = self.norm3(queries, self.norm3_ps, self.norm3_st)
    queries = sam_reshape(queries', queries_dim)

    # Cross attention block, image embedding attending to tokens 
    q = queries + query_pe
    k = keys + key_pe
    attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)

    keys = keys + attn_out

    keys_dim = size(keys)
    keys = sam_reshape(keys, (:, keys_dim[end]))'
    keys, _ = self.norm4(keys, self.norm4_ps, self.norm4_st)
    keys = sam_reshape(keys', keys_dim)


    return queries, keys
end



"""
########################################################
# TwoWayTransformer:
######################################################## 

    TwoWayTransformer

A transformer architecture composed of multiple stacked `TwoWayAttentionBlock`s 
	for bi-directional fusion between two sets of embeddings (e.g., image and 
	text). A final attention block reinforces the token-to-image alignment.

# Fields
- `depth::Int`  
  Number of stacked `TwoWayAttentionBlock` layers.
- `embedding_dim::Int`  
  Dimensionality of input and intermediate embeddings.
- `num_heads::Int`  
  Number of attention heads in each attention block.
- `mlp_dim::Int`  
  Hidden layer size of the MLP blocks inside each `TwoWayAttentionBlock`.
- `layers::Vector{TwoWayAttentionBlock}`  
  Sequence of stacked attention blocks composing the core of the transformer.
- `final_attn_token_to_image::_Attention`  
  Final cross-attention layer where tokens attend to image features.
- `norm_final_attn::LayerNorm`, `norm_final_attn_ps::NamedTuple`, 
  `norm_final_attn_st::NamedTuple`  
  LayerNorm and its parameters/state for post-final-attention normalization.
"""
struct TwoWayTransformer
	depth::Int
	embedding_dim::Int
	num_heads::Int
	mlp_dim::Int
	layers::Vector{TwoWayAttentionBlock}
	final_attn_token_to_image::_Attention
	norm_final_attn::LayerNorm
	norm_final_attn_ps::NamedTuple
	norm_final_attn_st::NamedTuple
end


"""
    TwoWayTransformer(; 
		depth, 
		embedding_dim, 
		num_heads, 
		mlp_dim,
        activation = relu, 
		attention_downsample_rate = 2)

Creates a `TwoWayTransformer` composed of multiple `TwoWayAttentionBlock`s
	for deep bidirectional communication between two modalities (e.g., image 
	and token embeddings).

# Arguments
- `depth::Int`  
  Number of stacked `TwoWayAttentionBlock` layers.
- `embedding_dim::Int`  
  Dimensionality of the token and image embeddings.
- `num_heads::Int`  
  Number of attention heads used in each attention block.
- `mlp_dim::Int`  
  Dimensionality of the hidden layer inside the MLP sub-blocks.
- `activation::Function = relu`  
  Activation function used in the MLPs (e.g., `relu`, `gelu`, ...).
- `attention_downsample_rate::Int = 2`  
  Downsampling ratio used in the key/value projections for cross-attention 
  layers.

# Returns
A `TwoWayTransformer` instance with initialized attention layers and 
	normalization.
"""
function TwoWayTransformer(;
	depth::Int,
	embedding_dim::Int,
	num_heads::Int,
	mlp_dim::Int,
	activation::Function = relu,
	attention_downsample_rate::Int = 2,
)

	rng = Random.MersenneTwister()

	depth = depth
	num_heads = num_heads
	mlp_dim = mlp_dim
	layers = Vector{TwoWayAttentionBlock}(undef, depth)

	for i in 1:depth
		layers[i] = TwoWayAttentionBlock(
			embedding_dim = embedding_dim,
			num_heads = num_heads,
			mlp_dim = mlp_dim,
			activation = activation,
			attention_downsample_rate = attention_downsample_rate,
			skip_first_layer_pe = (i == 1),
		)
	end

	final_attn_token_to_image = _Attention(
		embedding_dim = embedding_dim,
		num_heads = num_heads,
		downsample_rate = attention_downsample_rate,
	)

	norm_final_attn = LayerNorm(embedding_dim, dims = 1)

	norm_final_attn_ps, norm_final_attn_st =
		Lux.setup(rng, norm_final_attn)

	return TwoWayTransformer(
		depth,
		embedding_dim,
		num_heads,
		mlp_dim,
		layers,
		final_attn_token_to_image,
		norm_final_attn,
		norm_final_attn_ps,
		norm_final_attn_st,
	)
end

"""
    (transformer::TwoWayTransformer)(;
		image_embedding, 
		image_pe, 
		point_embedding) -> (queries, keys)

Applies the full two-way transformer to perform iterative communication between 
	`point_embedding` (queries)
and `image_embedding` (keys). The process includes self-attention on the 
	points, bidirectional cross-attention,
and a final attention layer from tokens to image features.

# Arguments
- `image_embedding::AbstractArray`  
  Tensor of shape `(B, C, H, W)` representing spatial image features.
- `image_pe::AbstractArray`  
  Positional encoding of the image features, of the same shape as 
  `image_embedding`.
- `point_embedding::AbstractArray`  
  Token-level embedding representing points, typically of shape `(B, N, C)`.

# Returns
- `queries::AbstractArray`  
  Refined token features after bidirectional interaction with image features.
- `keys::AbstractArray`  
  Refined image features updated via image-to-token attention.
"""
function (self::TwoWayTransformer)(;
	image_embedding::AbstractArray,
	image_pe::AbstractArray,
	point_embedding::AbstractArray,
)::Tuple{AbstractArray, AbstractArray}

	bs, c, h, w = size(image_embedding)
	image_embedding = sam_reshape(image_embedding, (bs, c, h * w))
	image_embedding = permutedims(image_embedding, (1, 3, 2))

	image_pe = sam_reshape(image_pe, (bs, c, h * w))
	image_pe = permutedims(image_pe, (1, 3, 2))

	queries = point_embedding
	keys = image_embedding

	for layer in self.layers
		queries, keys = layer(
			queries = queries,
			keys = keys,
			query_pe = point_embedding,
			key_pe = image_pe,
		)
	end

	q = queries + point_embedding
	k = keys + image_pe

	attn_out = self.final_attn_token_to_image(q = q, k = k, v = keys)

	queries = queries + attn_out

    queries_dim = size(queries)
    queries = sam_reshape(queries, (:, queries_dim[end]))'
	queries, _ = self.norm_final_attn(
		queries,
		self.norm_final_attn_ps,
		self.norm_final_attn_st,
	)
    queries = sam_reshape(queries', queries_dim)

	return queries, keys
end


