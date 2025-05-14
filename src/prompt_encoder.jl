using Lux
using Lux: Dense
using Lux: Conv
using CUDA
using Interpolations
using TensorOperations
using Einsum
using Random
using NNlib


########################################################
# # PositionalEmbeddingRandom:
########################################################

struct PositionEmbeddingRandom
	positional_encoding_gaussian_matrix::AbstractArray
	scale::Union{Nothing, Float32}
end


function PositionEmbeddingRandom(;
	num_pos_feats::Int = 64,
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