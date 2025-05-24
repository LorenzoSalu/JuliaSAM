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

########################################################
# _Attention:
########################################################

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


function _separate_heads(
	x::AbstractArray,
	num_heads::Int,
)::AbstractArray

	b, n, c = size(x)
	x = sam_reshape(x, (b, n, num_heads, c ÷ num_heads))
	return permutedims(x, (1, 3, 2, 4))
end

function _recombine_heads(x::AbstractArray)::AbstractArray

	b, n_heads, n_tokens, c_per_head = size(x)
	x = permutedims(x, (1, 3, 2, 4))
	return sam_reshape(x, (b, n_tokens, n_heads * c_per_head))
end



########################################################
# TwoWayAttentionBlock:
########################################################


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


########################################################
# TwoWayTransformer:
######################################################## 

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


