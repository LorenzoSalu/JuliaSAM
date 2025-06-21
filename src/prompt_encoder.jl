using Lux
using Lux: Dense
using Lux: Conv
using CUDA
using Interpolations
using TensorOperations
using Einsum
using Random
using NNlib



"""
########################################################
# PositionalEmbeddingRandom:
########################################################

	struct PositionEmbeddingRandom

This layer produces fixed positional encodings for an input of shape `(H, W)`, based on a randomly initialized
Gaussian matrix. It is typically used to inject spatial information into transformer-based models.

# Fields
- `positional_encoding_gaussian_matrix::AbstractArray`: A learnable matrix of shape `(2, num_pos_feats)` used to project positions.
- `scale::Union{Nothing, Float32}`: A scaling factor applied to the Gaussian matrix during initialization. Defaults to `1.0` if not provided or non-positive.
"""
struct PositionEmbeddingRandom
	positional_encoding_gaussian_matrix::AbstractArray
	scale::Union{Nothing, Float32}
end


"""
    PositionEmbeddingRandom(
		num_pos_feats::Int = 64; 
		scale::Union{Nothing, Float32} = nothing)

A positional encoding layer that generates 2D random Fourier features based on a learned Gaussian matrix.

# Arguments
- `num_pos_feats::Int`: Number of random Fourier features per axis. The final output will have shape `(2 * num_pos_feats, H, W)`.
- `scale::Union{Nothing, Float32}`: Optional scaling factor for the Gaussian matrix. Defaults to `1.0` if not provided or non-positive.

# Returns
A `PositionEmbeddingRandom` layer that can be called on a spatial input size `(H, W)` to generate positional embeddings.
"""
function PositionEmbeddingRandom(
	num_pos_feats::Int = 64;
	scale::Union{Nothing, Float32} = nothing,
)

	if isnothing(scale) || scale <= 0.0
		scale = 1.0
	end

	positional_encoding_gaussian_matrix =
		scale .* randn(Float32, 2, num_pos_feats)

	return PositionEmbeddingRandom(
		positional_encoding_gaussian_matrix,
		scale,
	)
end


"""
    (layer::PositionEmbeddingRandom)(
		input_size::Tuple{Int, Int})

Generates positional encodings for a given 2D input size using random Fourier features.

# Arguments
- `input_size::Tuple{Int, Int}`: A tuple `(H, W)` representing the spatial resolution of the input.

# Returns
An array of shape `(2 * num_pos_feats, H, W)` containing the positional encodings.
"""
function (self::PositionEmbeddingRandom)(input_size::Tuple{Int, Int})
	h, w = input_size

	# Qui potrebbe andare una funzione per il calcolo su gpu
	grid = ones(Float32, h, w)

	y_embed = cumsum(grid, dims = 1) .- 0.5
	x_embed = cumsum(grid, dims = 2) .- 0.5

	y_embed = y_embed ./ h
	x_embed = x_embed ./ w

	coords = cat(x_embed, y_embed; dims = 3)

	pe = _pe_encoding(self, coords)

	return permutedims(pe, (3, 1, 2))
end


"""
	forward_with_coords(
		self::PositionEmbeddingRandom,
		coords_input::AbstractArray,
		image_size::Tuple{Int, Int},
		)

Applies the positional embedding to explicitly provided (x, y) coordinates.

# Arguments
- `coords_input::AbstractArray`: An array of shape `(H, W, 2)` with normalized or unnormalized spatial coordinates.
- `image_size::Tuple{Int, Int}`: A tuple `(H, W)` used to normalize the coordinates in the `[0, 1]` range.

# Returns
A tensor of shape `(H, W, 2 x num_pos_feats)` representing the positional encoding for the given coordinates.
"""
function forward_with_coords(
	self::PositionEmbeddingRandom,
    coords_input::AbstractArray,
	image_size::Tuple{Int, Int},
    )

	coords = copy(coords_input)

    coords[:, :, 1] .= coords[:, :, 1] ./ image_size[2]
    coords[:, :, 2] .= coords[:, :, 2] ./ image_size[1]

    return _pe_encoding(self, coords)
end

"""
	_pe_encoding(self::PositionEmbeddingRandom, coords::AbstractArray)

Projects 2D coordinates into a higher-dimensional random Fourier space and applies sine and cosine functions.

# Arguments
- `coords::AbstractArray`: A tensor of shape `(H, W, 2)` with normalized coordinates in the range `[0, 1]`.

# Returns
A tensor of shape `(H, W, 2 x num_pos_feats)` containing sinusoidal encodings.
"""
function _pe_encoding(self::PositionEmbeddingRandom, coords::AbstractArray)

	h, w = size(coords, 1), size(coords, 2)

	coords = 2 .* coords .- 1
	coords = reshape(coords, (h * w, 2))
	coords = coords * self.positional_encoding_gaussian_matrix
	coords = reshape(
		coords,
		(h, w, size(self.positional_encoding_gaussian_matrix, 2)),
	)

	coords = 2π .* coords

	return cat(sin.(coords), cos.(coords); dims = ndims(coords))
end



"""
############################################################
# PromptEncoder
############################################################
	
	struct PromptEncoder

A module that encodes user prompts (points, boxes, and masks) into embeddings for use in segmentation transformer models.

# Fields
- `embed_dim::Int`: Dimensionality of the output embeddings.
- `input_image_size::Tuple{Int, Int}`: Size of the original input image, used for normalization.
- `image_embedding_size::Tuple{Int, Int}`: Size of the image embedding to align prompt encodings spatially.
- `pe_layer::PositionEmbeddingRandom`: Module generating random Fourier positional encodings.
- `num_point_embedding::Int`: Number of distinct point embedding types (e.g., positive, negative, etc.).
- `point_embeddings::Chain`: A small sequence of `Embedding` layers for point prompts.
- `point_embeddings_ps::NamedTuple`: Parameters of the point embedding chain.
- `point_embeddings_st::NamedTuple`: States of the point embedding chain.
- `not_a_point_embed::Embedding`: Embedding used when a point is absent.
- `not_a_point_embed_ps::NamedTuple`: Parameters of the "not a point" embedding.
- `not_a_point_embed_st::NamedTuple`: States of the "not a point" embedding.
- `mask_input_size::Tuple{Int, Int}`: Expected size of the input mask before downscaling.
- `mask_downscaling::Chain`: CNN used to compress the binary mask into an embedding.
- `mask_downscaling_ps::NamedTuple`: Parameters of the mask downscaling network.
- `mask_downscaling_st::NamedTuple`: States of the mask downscaling network.
- `no_mask_embed::Embedding`: Learned embedding used when no mask is provided.
- `no_mask_embed_ps::NamedTuple`: Parameters of the "no mask" embedding.
- `no_mask_embed_st::NamedTuple`: States of the "no mask" embedding.
"""
struct PromptEncoder
	embed_dim::Int
	input_image_size::Tuple{Int, Int}
	image_embedding_size::Tuple{Int, Int}
	pe_layer::PositionEmbeddingRandom
	num_point_embedding::Int
	point_embeddings::Chain
	point_embeddings_ps::NamedTuple
	point_embeddings_st::NamedTuple
	not_a_point_embed::Embedding
	not_a_point_embed_ps::NamedTuple
	not_a_point_embed_st::NamedTuple
	mask_input_size::Tuple{Int, Int}
	mask_downscaling::Chain
	mask_downscaling_ps::NamedTuple
	mask_downscaling_st::NamedTuple
	no_mask_embed::Embedding
	no_mask_embed_ps::NamedTuple
	no_mask_embed_st::NamedTuple
end


"""
	function PromptEncoder(;
		embed_dim::Int,
		image_embedding_size::Tuple{Int, Int},
		input_image_size::Tuple{Int, Int},
		mask_in_chans::Int,
		activation::Function = gelu_exact,
	)

Constructs a `PromptEncoder` for embedding point, box, and mask prompts.

# Arguments
- `embed_dim::Int`: Dimension of the output embeddings.
- `image_embedding_size::Tuple{Int, Int}`: Spatial size of the transformer input tokens.
- `input_image_size::Tuple{Int, Int}`: Original image size, used to normalize coordinates.
- `mask_in_chans::Int`: Number of channels in the input mask.
- `activation::Function`: Activation function to use in the mask encoder CNN (default = `gelu_exact`).

# Returns
A `PromptEncoder` instance with initialized embedding layers and CNN modules.
"""
function PromptEncoder(;
	embed_dim::Int,
	image_embedding_size::Tuple{Int, Int},
	input_image_size::Tuple{Int, Int},
	mask_in_chans::Int,
	activation::Function = gelu_exact,
)

	rng = Random.MersenneTwister()

	embed_dim = embed_dim
	input_image_size = input_image_size
	image_embedding_size = image_embedding_size

	pe_layer = PositionEmbeddingRandom(embed_dim ÷ 2)

	num_point_embedding = 4

	point_embeddings =
		[Lux.Embedding(1 => embed_dim) for i in 1:num_point_embedding]

	point_embeddings = Chain(point_embeddings...)
	point_embeddings_ps, point_embeddings_st = Lux.setup(rng, point_embeddings)

	not_a_point_embed = Lux.Embedding(1 => embed_dim)
	not_a_point_embed_ps, not_a_point_embed_st =
		Lux.setup(rng, not_a_point_embed)

	mask_input_size = (4 * image_embedding_size[1], 4 * image_embedding_size[2])

	mask_downscaling = Chain(
		Conv(
			(2, 2),
			1 => mask_in_chans ÷ 4;
			stride = 2,
			cross_correlation = true,
			init_weight = kaiming_uniform,
		),
		LayerNorm2d(mask_in_chans ÷ 4),
		x -> activation.(x),
		Conv(
			(2, 2),
			mask_in_chans ÷ 4 => mask_in_chans;
			stride = 2,
			cross_correlation = true,
			init_weight = kaiming_uniform,
		),
		LayerNorm2d(mask_in_chans),
		x -> activation.(x),
		Conv(
			(1, 1),
			mask_in_chans => embed_dim;
			cross_correlation = true,
			init_weight = kaiming_uniform,
		),
	)

	mask_downscaling_ps, mask_downscaling_st = Lux.setup(rng, mask_downscaling)

	no_mask_embed = Lux.Embedding(1 => embed_dim)
	no_mask_embed_ps, no_mask_embed_st = Lux.setup(rng, no_mask_embed)

	return PromptEncoder(
		embed_dim,
		input_image_size,
		image_embedding_size,
		pe_layer,
		num_point_embedding,
		point_embeddings,
		point_embeddings_ps,
		point_embeddings_st,
		not_a_point_embed,
		not_a_point_embed_ps,
		not_a_point_embed_st,
		mask_input_size,
		mask_downscaling,
		mask_downscaling_ps,
		mask_downscaling_st,
		no_mask_embed,
		no_mask_embed_ps,
		no_mask_embed_st,
	)
end



"""
	get_dense_pe(self::PromptEncoder)::AbstractArray

Returns the dense positional encoding for the image embedding grid.

# Returns
- `pe::AbstractArray`: A tensor of shape `(1, C, H, W)` containing Fourier positional encodings.
"""
function get_dense_pe(self::PromptEncoder)::AbstractArray
	tmp = self.pe_layer(self.image_embedding_size)
	return reshape(
		tmp,
		(1, size(tmp)...),
	)
end


"""
	function _embed_points(
		self::PromptEncoder,
		points::AbstractArray,
		labels::AbstractArray;
		pad::Bool,
	)::AbstractArray

Embeds point prompts using a combination of positional encoding and learned point-type embeddings.

# Arguments
- `points::AbstractArray`: Coordinates of the points with shape `(B, N, 2)`, where `B` is batch size and `N` is the number of points.
- `labels::AbstractArray`: Labels for each point (1 = foreground, 0 = background, -1 = padding).
- `pad::Bool`: Whether to append a padding point with label -1 to the input.

# Returns
- `point_embedding::AbstractArray`: Embedded point features with shape `(B, N, D)`, where `D` is the embedding dimension.
"""
function _embed_points(
	self::PromptEncoder,
	points::AbstractArray,
	labels::AbstractArray;
	pad::Bool,
)::AbstractArray

	points = points .+ 0.5

	if pad
		device = isa(points, CuArray) ? CuArray : Array
		padding_point = device(zeros(Float32, size(points, 1), 1, 2))

		device = isa(labels, CuArray) ? CuArray : Array
		padding_label = device(Float32.(fill(-1, size(labels, 1), 1)))

		points = cat(points, padding_point; dims = 2)
		labels = cat(labels, padding_label; dims = 2)
	end

	point_embedding = forward_with_coords(
		self.pe_layer,
		points,
		self.input_image_size,
	)

	mask = labels .== -1
	indices = findall(mask)
	for idx in indices
		point_embedding[idx, :] .= 0.0
	end

	mask = labels .== -1
	indices = findall(mask)
	for idx in indices
		point_embedding[idx, :] .+= self.not_a_point_embed_ps.weight
	end

	mask = labels .== 0
	indices = findall(mask)
	for idx in indices
		point_embedding[idx, :] .+= self.point_embeddings_ps[1].weight
	end

    mask = labels .== 1
	indices = findall(mask)
	for idx in indices
		point_embedding[idx, :] .+= self.point_embeddings_ps[2].weight
	end

	return point_embedding
end


"""
	_embed_boxes(
		self::PromptEncoder,
		boxes::AbstractArray)::AbstractArray

Embeds box prompts using the positional encoding of their corners, each enhanced with a learned token type embedding.

# Arguments
- `boxes::AbstractArray`: Bounding boxes with shape `(B, 4)` in `(x0, y0, x1, y1)` format.

# Returns
- `corner_embedding::AbstractArray`: Embedded corner features with shape `(B, 2, D)`, where `D` is the embedding dimension.
"""
function _embed_boxes(
	self::PromptEncoder,
	boxes::AbstractArray)::AbstractArray

	boxes = boxes .+ 0.5
	coords = sam_reshape(boxes, (:, 2, 2))

	corner_embedding = forward_with_coords(
		self.pe_layer,
		coords,
		self.input_image_size,
	)

	corner_embedding[:, 1, :] .+= self.point_embeddings_ps[3].weight'
	corner_embedding[:, 2, :] .+= self.point_embeddings_ps[4].weight'

	return corner_embedding
end


"""
	_embed_masks(
		self::PromptEncoder,
		masks::AbstractArray)::AbstractArray
		
Embeds binary mask prompts using a small CNN encoder.

# Arguments
- `masks::AbstractArray`: Input masks with shape `(B, 1, H, W)`.

# Returns
- `mask_embedding::AbstractArray`: Embedded masks with shape `(B, D, H', W')`, where `D` is the embedding dimension and `(H', W')` is the downscaled size.
"""
function _embed_masks(
	self::PromptEncoder,
	masks::AbstractArray)::AbstractArray

	mask_embedding, _ = self.mask_downscaling(
		permutedims(masks, (3, 4, 2, 1)),
		self.mask_downscaling_ps,
		self.mask_downscaling_st,
	)

    mask_embedding = permutedims(mask_embedding, (4, 3, 1, 2))

	return mask_embedding
end

"""
	_get_batch_size(
		self::PromptEncoder,
		points::Union{Nothing, Tuple{AbstractArray, AbstractArray}},
		boxes::Union{Nothing, AbstractArray},
		masks::Union{Nothing, AbstractArray},
	)::Int

Determines the batch size from the available prompt inputs.

# Arguments
- `points::Union{Nothing, Tuple{AbstractArray, AbstractArray}}`: Optional tuple of point coordinates and labels.
- `boxes::Union{Nothing, AbstractArray}`: Optional bounding boxes.
- `masks::Union{Nothing, AbstractArray}`: Optional masks.

# Returns
- `batch_size::Int`: The batch size inferred from the first available prompt input. Defaults to 1 if none are provided.
"""
function _get_batch_size(
	self::PromptEncoder,
	points::Union{Nothing, Tuple{AbstractArray, AbstractArray}},
	boxes::Union{Nothing, AbstractArray},
	masks::Union{Nothing, AbstractArray},
)::Int

	if !isnothing(points)
		return size(points[1], 1)
	elseif !isnothing(boxes)
		return size(boxes, 1)
	elseif !isnothing(masks)
		return size(masks, 1)
	else
		return 1
	end
end


"""
	_get_device(self::PromptEncoder)

Returns the device type used by the prompt encoder's embeddings.

# Returns
- `device`: Either `CuArray` if embeddings are on GPU, or `Array` for CPU.
"""
function _get_device(self::PromptEncoder)
	return isa(self.point_embeddings_ps[1].weight, CuArray) ? CuArray : Array
end


"""

	(self::PromptEncoder)(;
		points::Union{Nothing, Tuple{AbstractArray, AbstractArray}},
		boxes::Union{Nothing, AbstractArray},
		masks::Union{Nothing, AbstractArray},
	)::Tuple{AbstractArray, AbstractArray}

Encodes given prompt inputs (points, boxes, masks) into sparse and dense embeddings.

# Arguments
- `points::Union{Nothing, Tuple{AbstractArray, AbstractArray}}`: Optional tuple of point coordinates and labels.
- `boxes::Union{Nothing, AbstractArray}`: Optional bounding boxes.
- `masks::Union{Nothing, AbstractArray}`: Optional binary masks.

# Returns
- `sparse_embeddings::AbstractArray`: Concatenated embeddings for sparse prompts (points and boxes), shape `(B, N, D)`.
- `dense_embeddings::AbstractArray`: Dense embeddings for masks or a learned no-mask embedding, shape `(B, D, H, W)`.

The method automatically pads points if boxes are not provided.
"""
function (self::PromptEncoder)(;
	points::Union{Nothing, Tuple{AbstractArray, AbstractArray}},
	boxes::Union{Nothing, AbstractArray},
	masks::Union{Nothing, AbstractArray},
)::Tuple{AbstractArray, AbstractArray}

	bs = _get_batch_size(self, points, boxes, masks)
	device = _get_device(self)
	sparse_embeddings = device(Array{Float32}(undef, bs, 0, self.embed_dim))

	if !isnothing(points)
		coords, labels = points
		pad = isnothing(boxes)
		point_embeddings = _embed_points(self, coords, labels, pad = pad)
		sparse_embeddings = cat(sparse_embeddings, point_embeddings; dims = 2)
	end

	if !isnothing(boxes)
		box_embeddings = _embed_boxes(self, boxes)
		sparse_embeddings = cat(sparse_embeddings, box_embeddings; dims = 2)
	end

	if !isnothing(masks)
		dense_embeddings = _embed_masks(self, masks)
	else
		dense_embeddings =
			sam_reshape(self.no_mask_embed_ps.weight, (1, :, 1, 1))

		dense_embeddings =
			repeat(
				dense_embeddings,
				bs,
				1,
				self.image_embedding_size[1],
				self.image_embedding_size[2],
			)
	end

	return sparse_embeddings, dense_embeddings
end