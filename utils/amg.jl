using Lux
using CUDA
using Interpolations
using TensorOperations
using Einsum
using NPZ
using NNlib
using ImageTransformations
using ImageComponentAnalysis

using PyCall

include("../global_functions.jl")


########################################################
########################################################
# MaskDataTest
########################################################
########################################################

struct MaskData
	_stats::Dict{String, Any}
end

function MaskData(; kwargs::Dict{String, <:Any})
	for v in values(kwargs)
		@assert v isa AbstractArray || v isa Vector || v isa CuArray
		"MaskData only supports Vectors, AbstractArrays, or CuArrays."
	end
	return MaskData(kwargs)
end


# Aggiunta elemento al dizionario
function Base.setindex!(md::MaskData, item::Any, key::String)
	@assert item isa AbstractArray || item isa Vector || item isa CuArray
	"MaskData only supports Vectors, AbstractArrays, or CuArrays."
	md._stats[key] = item
end

# Ottieni un elemento dal dizionario
function Base.getindex(md::MaskData, key::String)
	return md._stats[key]
end

# Elimina un elemento dal dizionario
function Base.delete!(md::MaskData, key::String)
	delete!(md._stats, key)
end

# Itera sugli elementi del dizionario
function pairs(md::MaskData)
	return pairs(md._stats)
end

# Filtra i dati in base a una maschera booleana
function filter!(md::MaskData, keep::AbstractArray)
	for (k, v) in md._stats
		if v === nothing
			md._stats[k] = nothing
		elseif v isa CuArray || v isa AbstractArray
			md._stats[k] = v[keep, ntuple(_ -> :, ndims(v) - 1)...]
		elseif v isa Vector && type(keep) == Bool
			md._stats[k] = [v[i] for i in eachindex(keep) if keep[i]]
		elseif v isa Vector
			md._stats[k] = [v[i] for i in eachindex(keep)]
		else
			throw(ArgumentError(
				"MaskData key $k has an unsupported type $(typeof(v))."))
		end
	end
end


# Concatena nuovi dati al dizionario esistente
function cat!(md::MaskData, new_stats::MaskData)
	for (k, v) in new_stats._stats
		if !haskey(md._stats, k) || md._stats[k] === nothing
			md._stats[k] = deepcopy(v)
		elseif v isa CuArray || v isa AbstractArray
			md._stats[k] = vcat(md._stats[k], v)
		elseif v isa Vector
			md._stats[k] = vcat(md._stats[k], deepcopy(v))
		else
			throw(ArgumentError(
				"MaskData key $k has an unsupported type $(typeof(v))."))
		end
	end
end


# Converte tutti i tensori in Array
function to_array!(md::MaskData)
	for (k, v) in md._stats
		if v isa CuArray
			md._stats[k] = AbstractArray(v)
		end
	end
end


function batched_mask_to_box(masks::AbstractArray)::AbstractArray
	"""
	Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
	an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
	"""

	if isempty(masks)
		if masks isa CuArray
			return CuArray(zeros(eltype(masks), size(masks)[1:end-2]..., 4))
		else
			return zeros(eltype(masks), size(masks)[1:end-2]..., 4)
		end
	end

	shape = size(masks)
	h, w = shape[end-1:end]

	if length(masks) > 2
		if size(masks)[end-2:end] != size(masks)
			masks = reshape(masks, (:, size(masks)[end-2:end]...))
		end
	else
		masks = reshape(masks, (1, size(masks)...))
	end

	in_height = maximum(masks, dims = ndims(masks))
	in_height = dropdims(in_height, dims = ndims(in_height))
	in_height_coords = in_height .* (0:h-1)'
	bottom_edges = maximum(in_height_coords, dims = ndims(in_height_coords))
	bottom_edges = dropdims(bottom_edges, dims = ndims(bottom_edges))
	in_height_coords .= in_height_coords .+ h .* .!in_height
	top_edges = minimum(in_height_coords, dims = ndims(in_height_coords))
	top_edges = dropdims(top_edges, dims = ndims(top_edges))

	in_width = maximum(masks, dims = ndims(masks) - 1)
	in_width = dropdims(in_width, dims = ndims(in_width) - 1)
	in_width_coords = in_width .* (0:w-1)'
	right_edges = maximum(in_width_coords, dims = ndims(in_width_coords))
	right_edges = dropdims(right_edges, dims = ndims(right_edges))
	in_width_coords .= in_width_coords .+ w .* .!in_width
	left_edges = minimum(in_width_coords, dims = ndims(in_width_coords))
	left_edges = dropdims(left_edges, dims = ndims(left_edges))

	empty_filter = (right_edges .< left_edges) .| (bottom_edges .< top_edges)
	out = hcat(left_edges, top_edges, right_edges, bottom_edges)
	out .= out .* .!reshape(empty_filter, (size(empty_filter)..., 1))

	if length(shape) > 2
		out = sam_reshape(out, (shape[1:end-2]..., 4))
	else
		out = out[0]
	end

	return out
end

function coco_encode_rle(
	uncompressed_rle::Dict{String, <:Any},
)
	h, w = uncompressed_rle["size"]
	mask = pyimport("pycocotools.mask")
	rle = mask.frPyObjects(uncompressed_rle, h, w)
	rle["counts"] = String(rle["counts"])
	return rle
end

function remove_small_regions(
	mask::AbstractArray;
	area_thresh::Float32,
	mode::String,
)
	"""
	Removes small disconnected regions and holes in a mask. Returns the
	mask and an indicator of if the mask has been modified.
	"""

	@assert mode in ["holes", "islands"] "mode must be 'holes' or 'islands'."

	correct_holes = mode == "holes"
	working_mask = Int.(xor.(correct_holes, mask))

	n_labels, regions, stats = connected_components_with_stats(working_mask)

	sizes = stats[:, end][2:end]

	small_regions = [i for (i, s) in enumerate(sizes) if s < area_thresh]

	if length(small_regions) == 0
		return mask, false
	end

	fill_labels = vcat(0, small_regions)

	if !correct_holes
		fill_labels = [i for i in 1:n_labels if i ∉ fill_labels]

		if length(fill_labels) == 0
			fill_labels = [trunc(Int, argmax(sizes))]
		end
	end

	mask = broadcast(∈(fill_labels), regions)

	return mask, true
end


function connected_components_with_stats(mask::Matrix{Int})
	# Get connected components
	components = label_components(mask)
	n_labels = maximum(components)

	# Initialize stats array (left, top, width, height, area)
	stats = zeros(Int, n_labels + 1, 5)

	# Calculate stats for each component
	for label in 0:n_labels
		# Get component pixels
		component = components .== label
		if any(component)
			# Find bounding box
			indices = findall(component)
			rows = getindex.(indices, 1)
			cols = getindex.(indices, 2)

			top = minimum(rows)
			bottom = maximum(rows)
			left = minimum(cols)
			right = maximum(cols)

			# Calculate stats
			stats[label+1, :] = [
				left,              # x
				top,               # y
				right - left + 1,  # width
				bottom - top + 1,  # height
				count(component),  # area
			]
		end
	end

	return n_labels, components, stats
end



function uncrop_masks(
	masks::AbstractArray, crop_box::Vector{Int}, orig_h::Int, orig_w::Int,
)::AbstractArray

	x0, y0, x1, y1 = crop_box
	if x0 == 0 && y0 == 0 && x1 == orig_w && y1 == orig_h
		return masks
	end

	pad_x = orig_w - (x1 - x0)
	pad_y = orig_h - (y1 - y0)
	pad = (0, 0, y0, pad_y - y0, x0, pad_x - x0)

	#sopra, sotto, sinistra, destra
	padded_masks = NNlib.pad_zeros(masks, pad)
	return padded_masks
end

function uncrop_points(
	points::AbstractArray, crop_box::Vector{Int},
)::AbstractArray

	x0, y0, _, _ = crop_box

	if points isa CuArray
		offset = CuArray(reshape([x0, y0], (1, 2)))  # Crea l'offset su GPU
	else
		offset = reshape([x0, y0], (1, 2))  # Crea l'offset su CPU
	end

	if length(size(points)) == 3
		offset = reshape(offset, (size(offset, 1), 1, size(offset)[2:end]...))
	end

	return offset .+ points
end

function uncrop_boxes_xyxy(
	boxes::AbstractArray,
	crop_box::Vector{Int})::AbstractArray

	x0, y0, _, _ = crop_box

	offset =
		boxes isa CuArray ? CuArray(reshape([x0, y0, x0, y0], (1, 4))) :
		reshape([x0, y0, x0, y0], (1, 4))

	if ndims(boxes) == 3
		offset = reshape(offset, (size(offset, 1), 1, size(offset)[2:end]...))
	end

	return boxes .+ offset
end

function generate_crop_boxes(
	im_size::Tuple{Int, Int},
	n_layers::Int,
	overlap_ratio::Float32,
)::Tuple{Vector{Vector{Int}}, Vector{Int}}
	"""
	Generates a list of crop boxes of different sizes. Each layer
	has (2**i)**2 boxes for the ith layer.
	"""
	crop_boxes = []
	layer_idxs = []
	im_h, im_w = im_size
	short_side = min(im_h, im_w)

	push!(crop_boxes, [0, 0, im_w, im_h])
	push!(layer_idxs, 0)

	for i_layer in 0:n_layers-1
		n_crops_per_side = 2^(i_layer + 1)
		overlap =
			trunc(Int, overlap_ratio * short_side * (2 / n_crops_per_side))

		crop_w = crop_len(im_w, n_crops_per_side, overlap)
		crop_h = crop_len(im_h, n_crops_per_side, overlap)

		crop_box_x0 =
			[trunc(Int, (crop_w - overlap) * i) for i in 0:n_crops_per_side-1]
		crop_box_y0 =
			[trunc(Int, (crop_h - overlap) * i) for i in 0:n_crops_per_side-1]

		for x0 in crop_box_x0, y0 in crop_box_y0
			box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
			push!(crop_boxes, box)
			push!(layer_idxs, i_layer + 1)
		end
	end

	return crop_boxes, layer_idxs
end

function crop_len(orig_len, n_crops, overlap)
	return Int(ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
end

function build_all_layer_point_grids(
	n_per_side::Int, n_layers::Int, scale_per_layer::Int,
)::Vector{AbstractArray}

	points_by_layer = Vector{AbstractArray}(undef, n_layers + 1)
	for i in 0:n_layers
		n_points = Int(floor(n_per_side / (scale_per_layer^i)))
		points_by_layer[i+1] = build_point_grid(n_points)
	end

	return points_by_layer
end

function build_point_grid(n_per_side::Int)::AbstractArray
	offset = 1 / (2 * n_per_side)
	points_one_side = range(offset, stop = 1 - offset, length = n_per_side)
	points_x = repeat(points_one_side', n_per_side, 1)
	points_y = repeat(points_one_side, 1, n_per_side)
	points = hcat(vec(points_y), vec(points_x))
	return points
end

function calculate_stability_score(
	masks::AbstractArray, mask_threshold::Float32, threshold_offset::Float32,
)::AbstractArray

	high_masks = masks .> (mask_threshold + threshold_offset)
	high_masks = sum(high_masks, dims = ndims(high_masks))
	intersections = vec(sum(high_masks, dims = ndims(high_masks) - 1))

	low_masks = masks .> (mask_threshold - threshold_offset)
	low_masks = sum(low_masks, dims = ndims(low_masks))
	unions = vec(sum(low_masks, dims = ndims(low_masks) - 1))

	return intersections ./ unions
end

function area_from_rle(rle::Dict{String, <:Any})::Int
	return sum(rle["counts"][1:2:end])
end

function rle_to_mask(rle::Dict{String, <:Any})::AbstractArray
	"""Compute a binary mask from an uncompressed RLE."""
	h, w = rle["size"]

	mask = Array{Bool}(undef, h * w)
	idx = 1
	parity = false

	for count in rle["counts"]
		mask[idx:min(idx + count, length(mask))] .= parity
		idx += count
		parity = !parity
	end

	mask = sam_reshape(mask, (w, h))
	return mask'
end

function mask_to_rle(tensor::AbstractArray)::Vector{Dict{String, <:Any}}
	b, h, w, = size(tensor)

	tensor = permutedims(tensor, (1, 3, 2))
	tensor = sam_reshape(tensor, (b, :))

	diff = tensor[:, 2:end] .⊻ tensor[:, 1:end-1]
	change_indices = findall(diff)

	out = []

	for i in 1:b
		cur_idxs = [ci[2] for ci in change_indices if ci[1] == i]
		cur_idxs = vcat([1], cur_idxs .+ 1, [(h * w) + 1])
		btw_idxs = cur_idxs[2:end] .- cur_idxs[1:end-1]

		counts = tensor[i, 1] == 0 ? Int[] : [0]
		append!(counts, btw_idxs)
		push!(out, Dict("size" => [h, w], "counts" => counts))
	end

	return out
end

function batch_iterator(batch_size::Int, args::AbstractArray...)
	error =
		length(args) > 0 && all([length(a) == length(args[1]) for a in args])

	@assert error
	"Batched iteration must have inputs of all the same size."

	n_batches =
		length(args[1]) ÷ batch_size + Int(length(args[1]) % batch_size != 0)

	return (
		[[a[b*batch_size+1:min((b + 1) * batch_size, end)] for a in args]
		 for b in 0:n_batches-1])
end

function box_xyxy_to_xywh(box_xyxy::AbstractArray)::AbstractArray
	box_xywh = deepcopy(box_xyxy)
	box_xywh[3] -= box_xywh[1]
	box_xywh[4] -= box_xywh[2]
	return box_xywh
end

function is_box_near_crop_edge(
	boxes::AbstractArray,
	crop_box::Vector{Int},
	orig_box::Vector{Int},
	atol::Float32 = 20.0f0,
)

	"""is_box_near_crop_edge: Filter masks at the edge of a crop, 
	but not at the edge of the original image."""

	crop_box_lux =
		boxes isa CuArray ? CuArray(Float32.(crop_box)) : Float32.(crop_box)

	orig_box_lux =
		boxes isa CuArray ? CuArray(Float32.(orig_box)) : Float32.(orig_box)

	boxes = Float32.(uncrop_boxes_xyxy(boxes, crop_box))

	near_crop_edge = isapprox.(boxes, crop_box_lux', atol = atol, rtol = 0)
	near_image_edge = isapprox.(boxes, orig_box_lux', atol = atol, rtol = 0)

	near_crop_edge = near_crop_edge .& .!near_image_edge

	return any(near_crop_edge, dims = 2)
end