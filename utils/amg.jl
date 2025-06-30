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
# MaskDataTest
########################################################

"""
    struct MaskData

Container that stores heterogeneous mask–related tensors in a single
dictionary‐like structure.

# Fields
- `_stats::Dict{String, Any}`  
  Internal dictionary that maps string keys to data arrays
  (`AbstractArray`, `Vector`, or `CuArray`).
"""
struct MaskData
	_stats::Dict{String, Any}
end


"""
    MaskData(; kwargs::Dict{String, <:Any})

Constructs a `MaskData` object from a dictionary of keyword–style pairs.

# Arguments
- `kwargs::Dict{String, <:Any}`  
  Dictionary whose values must be `AbstractArray`, `Vector`,
  or `CuArray`. Each key represents a mask attribute
  (e.g. `"masks"`, `"labels"`, `"scores"`).

# Returns
- A `MaskData` instance wrapping the provided dictionary.
"""
function MaskData(; kwargs::Dict{String, <:Any})
	for v in values(kwargs)
		@assert v isa AbstractArray || v isa Vector || v isa CuArray
		"MaskData only supports Vectors, AbstractArrays, or CuArrays."
	end
	return MaskData(kwargs)
end


"""
    Base.setindex!(md::MaskData, item::Any, key::String)

Inserts or updates a key–value pair in the `MaskData` container.

# Arguments
- `md::MaskData`: The `MaskData` object to be modified.
- `item::Any`: The value to assign. Must be an `AbstractArray`, `Vector`, or `CuArray`.
- `key::String`: The string key under which to store the value.

# Behavior
- Asserts that `item` is a valid array-like object.
- Updates the internal dictionary `_stats` to associate `key` with `item`.
"""
function Base.setindex!(md::MaskData, item::Any, key::String)
	@assert item isa AbstractArray || item isa Vector || item isa CuArray
	"MaskData only supports Vectors, AbstractArrays, or CuArrays."
	md._stats[key] = item
end



"""
    Base.getindex(md::MaskData, key::String) -> Any

Retrieves the value associated with a given key in the `MaskData` object.

# Arguments
- `md::MaskData`: The `MaskData` container.
- `key::String`: The key whose associated value is to be returned.

# Returns
- The array-like object (e.g., `Vector`, `AbstractArray`, or `CuArray`) stored under `key`.
"""
function Base.getindex(md::MaskData, key::String)
	return md._stats[key]
end


"""
    Base.delete!(md::MaskData, key::String)

Deletes the entry associated with `key` from the `MaskData` object.

# Arguments
- `md::MaskData`: The `MaskData` container.
- `key::String`: The key to remove from the internal dictionary.

# Behavior
- Removes the key–value pair from the internal `_stats` dictionary if it exists.
"""
function Base.delete!(md::MaskData, key::String)
	delete!(md._stats, key)
end


"""
    pairs(md::MaskData) -> Base.Iterators.Pairs

Returns an iterator over the key–value pairs in the `MaskData` container.

# Arguments
- `md::MaskData`: The container to iterate over.

# Returns
- An iterator yielding `(key::String, value::Any)` pairs.

"""
function pairs(md::MaskData)
	return pairs(md._stats)
end


"""
    filter!(md::MaskData, keep::AbstractArray)

Filters the contents of the `MaskData` container in-place based on the boolean or index array `keep`.

# Arguments
- `md::MaskData`: The container to be filtered.
- `keep::AbstractArray`: An array specifying which elements to retain. Can be a boolean mask or an array of indices.

# Behavior
- If the stored value is `nothing`, it is preserved as `nothing`.
- If the value is an `AbstractArray` or `CuArray`, the array is indexed along its first dimension using `keep`.
- If the value is a `Vector` and `keep` is a boolean mask, only the elements at positions where `keep[i]` is `true` are retained.
- If the value is a `Vector` and `keep` is an index array, the corresponding elements are selected.
- Any unsupported type raises an `ArgumentError`.
"""
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


"""
    cat!(md::MaskData, new_stats::MaskData)

Concatenates the contents of another `MaskData` instance into the current one, modifying it in-place.

# Arguments
- `md::MaskData`: The target `MaskData` container to which data will be appended.
- `new_stats::MaskData`: The source `MaskData` whose values will be concatenated.

# Behavior
- For each key in `new_stats`:
  - If the key is not present in `md` or its value is `nothing`, the value is deep-copied from `new_stats`.
  - If the value is an `AbstractArray` or `CuArray`, it is concatenated using `vcat`.
  - If the value is a `Vector`, it is deep-copied and concatenated using `vcat`.
  - If the value type is unsupported, an `ArgumentError` is raised.
"""
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


"""
    to_array!(md::MaskData)

Converts all values stored in the `MaskData` instance from `CuArray` to standard `Array`, modifying the structure in-place.

# Arguments
- `md::MaskData`: The `MaskData` container to be modified.

# Behavior
- Iterates over all key-value pairs.
- If a value is a `CuArray`, it is converted to a standard `Array` using `Array(v)`.
- Other types are left unchanged.
"""
function to_array!(md::MaskData)
	for (k, v) in md._stats
		if v isa CuArray
			md._stats[k] = AbstractArray(v)
		end
	end
end

"""
    batched_mask_to_box(masks::AbstractArray) -> AbstractArray

Computes bounding boxes in XYXY format for each mask in a batch of masks.

# Arguments
- `masks::AbstractArray`: An N-dimensional tensor representing a batch of binary masks.
  The last two dimensions represent the height (H) and width (W) of each mask.
  Masks are expected to have values indicating the mask presence (e.g., 0 or 1).

# Returns
- An array of shape `C1 × C2 × ... × 4` where `C1, C2, ...` represent any leading batch dimensions of `masks`.
- Each bounding box is in the format `[x_min, y_min, x_max, y_max]`.
- For empty masks (no positive pixels), the bounding box is `[0, 0, 0, 0]`.
"""
function batched_mask_to_box(masks::AbstractArray)::AbstractArray

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


"""
    coco_encode_rle(uncompressed_rle::Dict{String, <:Any}) -> Dict{String, Any}

Encodes an uncompressed Run-Length Encoding (RLE) mask into the COCO RLE format.

# Arguments
- `uncompressed_rle::Dict{String, <:Any}`: A dictionary representing the uncompressed RLE mask.
  Expected to contain at least:
  - `"size"`: a tuple or array `(height, width)` representing the mask dimensions.
  - `"counts"`: the uncompressed RLE counts.

# Returns
- A dictionary representing the compressed RLE mask in COCO format.
  The `"counts"` field is converted to a `String` type as required by COCO.
"""
function coco_encode_rle(
	uncompressed_rle::Dict{String, <:Any},
)
	h, w = uncompressed_rle["size"]
	mask = pyimport("pycocotools.mask")
	rle = mask.frPyObjects(uncompressed_rle, h, w)
	rle["counts"] = String(rle["counts"])
	return rle
end

"""
    remove_small_regions(
		mask::AbstractArray; 
		area_thresh::Float32, 
		mode::String) -> Tuple{AbstractArray, Bool}

Removes small connected regions from a binary mask based on an area threshold.

# Arguments
- `mask::AbstractArray`: Binary mask array where the regions to process are defined.
- `area_thresh::Float32`: Minimum area threshold. Regions smaller than this are removed.
- `mode::String`: Either `"holes"` or `"islands"`.
    - `"holes"`: Removes small holes (background regions inside foreground).
    - `"islands"`: Removes small islands (foreground regions inside background).

# Returns
- A tuple `(filtered_mask, changed)` where:
    - `filtered_mask::AbstractArray`: The binary mask after removing small regions.
    - `changed::Bool`: `true` if any region was removed, `false` otherwise.
"""
function remove_small_regions(
	mask::AbstractArray;
	area_thresh::Float32,
	mode::String,
)
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

"""
    connected_components_with_stats(
		mask::Matrix{Int}) -> Tuple{Int, Matrix{Int}, Matrix{Int}}

Computes connected components of a binary mask and returns statistics for each component.

# Arguments
- `mask::Matrix{Int}`: Binary mask (integer matrix) where connected components are identified.

# Returns
- `n_labels::Int`: Number of connected components found (excluding background label 0).
- `components::Matrix{Int}`: Matrix of the same size as `mask`, where each pixel is labeled with its component index (0 for background).
- `stats::Matrix{Int}`: Matrix of size `(n_labels+1, 5)` where each row corresponds to a component and columns represent:
    1. `left` (x coordinate of bounding box)
    2. `top` (y coordinate of bounding box)
    3. `width` of bounding box
    4. `height` of bounding box
    5. `area` (number of pixels in the component)

The first row corresponds to the background (label 0).
"""
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


"""
    uncrop_masks(masks::AbstractArray, crop_box::Vector{Int}, orig_h::Int, orig_w::Int) -> AbstractArray

Restores masks cropped to a bounding box back to their original spatial dimensions by padding zeros around them.

# Arguments
- `masks::AbstractArray`: The cropped masks array.
- `crop_box::Vector{Int}`: Bounding box `[x0, y0, x1, y1]` that was used for cropping.
- `orig_h::Int`: Original height of the image before cropping.
- `orig_w::Int`: Original width of the image before cropping.

# Returns
- `AbstractArray`: The masks padded back to the original size `(orig_h, orig_w)`.

# Behavior
- If the crop box corresponds to the full original image (no cropping), the function returns the input masks unchanged.
- Otherwise, it pads the cropped masks with zeros to restore them to the original dimensions.
"""
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

"""
    uncrop_points(points::AbstractArray, crop_box::Vector{Int}) -> AbstractArray

Adjusts coordinates of points that were cropped within a bounding box, restoring them to the original coordinate system by adding the crop offset.

# Arguments
- `points::AbstractArray`: Array of points coordinates, typically of shape `(N, 2)` or `(batch_size, N, 2)`.
- `crop_box::Vector{Int}`: Bounding box `[x0, y0, x1, y1]` used for cropping.

# Returns
- `AbstractArray`: Points coordinates shifted by `(x0, y0)` offset, mapping them back to the original image space.
"""
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


"""
    uncrop_boxes_xyxy(
		boxes::AbstractArray, 
		crop_box::Vector{Int}) -> AbstractArray

Restores bounding boxes, cropped within a specified box, to the original coordinate space by adding the crop offset. The boxes are expected in XYXY format `[x_min, y_min, x_max, y_max]`.

# Arguments
- `boxes::AbstractArray`: Array of bounding boxes, shape `(N, 4)` or `(batch_size, N, 4)`.
- `crop_box::Vector{Int}`: Crop bounding box `[x0, y0, x1, y1]` applied previously.

# Returns
- `AbstractArray`: Bounding boxes shifted by `(x0, y0, x0, y0)` offset, mapping them back to original image coordinates.
"""
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


"""
    generate_crop_boxes(
		im_size::Tuple{Int, Int}, 
		n_layers::Int, 
		overlap_ratio::Float32) -> Tuple{Vector{Vector{Int}}, Vector{Int}}

Generates a list of crop boxes of varying sizes for multi-scale image cropping.

# Arguments
- `im_size::Tuple{Int, Int}`: Tuple `(height, width)` representing the original image dimensions.
- `n_layers::Int`: Number of crop layers. Each layer subdivides the image into progressively smaller crop boxes.
- `overlap_ratio::Float32`: Fraction of the crop box size that crops should overlap.

# Returns
- `Tuple{Vector{Vector{Int}}, Vector{Int}}`:
  - A vector of crop boxes, each defined as `[x0, y0, x1, y1]`.
  - A vector of integers indicating the layer index for each crop box (0 for the full image, then 1 to `n_layers` for the subdivided layers).
"""
function generate_crop_boxes(
	im_size::Tuple{Int, Int},
	n_layers::Int,
	overlap_ratio::Float32,
)::Tuple{Vector{Vector{Int}}, Vector{Int}}

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

"""
    crop_len(orig_len::Int, n_crops::Int, overlap::Int) -> Int

Calculates the length of each crop box dimension, given the original dimension,
the number of crops along that dimension, and the overlap in pixels between adjacent crops.

# Arguments
- `orig_len::Int`: Original length (height or width) of the image dimension.
- `n_crops::Int`: Number of crops along this dimension.
- `overlap::Int`: Number of pixels overlap between adjacent crops.

# Returns
- `Int`: The computed crop box length for this dimension.
"""
function crop_len(orig_len, n_crops, overlap)
	return Int(ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
end


"""
    build_all_layer_point_grids(
		n_per_side::Int, 
		n_layers::Int, 
		scale_per_layer::Int) -> Vector{AbstractArray}

Generates a vector of point grids for multiple layers, where each successive layer has a scaled down
number of points per side.

# Arguments
- `n_per_side::Int`: Number of points per side in the base layer (layer 0).
- `n_layers::Int`: Total number of layers to generate (excluding layer 0).
- `scale_per_layer::Int`: Scale factor to reduce points per side for each subsequent layer.

# Returns
- `Vector{AbstractArray}`: A vector where each element is a grid of points for the corresponding layer.
  The vector length is `n_layers + 1` (including the base layer).
"""
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

"""
    build_point_grid(n_per_side::Int) -> AbstractArray

Generates a 2D grid of points evenly spaced in the unit square [0, 1] × [0, 1].

# Arguments
- `n_per_side::Int`: Number of points per side in the grid.

# Returns
- `AbstractArray`: An array of 2D points of size `(n_per_side^2, 2)`. Each row is a point `[y, x]` with coordinates
  normalized between 0 and 1, offset to avoid exact edges.
"""
function build_point_grid(n_per_side::Int)::AbstractArray
	offset = 1 / (2 * n_per_side)
	points_one_side = range(offset, stop = 1 - offset, length = n_per_side)
	points_x = repeat(points_one_side', n_per_side, 1)
	points_y = repeat(points_one_side, 1, n_per_side)
	points = hcat(vec(points_y), vec(points_x))
	return points
end


"""
    calculate_stability_score(
        masks::AbstractArray,
        mask_threshold::Float32,
        threshold_offset::Float32,
    ) -> AbstractArray

Computes a stability score for masks based on thresholded intersections and unions.

# Arguments
- `masks::AbstractArray`: Array containing mask probability values or scores.
- `mask_threshold::Float32`: Central threshold value for binarizing the masks.
- `threshold_offset::Float32`: Offset applied above and below the central threshold to define "high" and "low" masks.

# Returns
- `AbstractArray`: Stability scores calculated as the ratio of the intersection to the union across the last two dimensions of the mask array.
"""
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


"""
    area_from_rle(rle::Dict{String, <:Any}) -> Int

Computes the area (number of pixels) of a mask encoded in Run-Length Encoding (RLE) format.

# Arguments
- `rle::Dict{String, <:Any}`: A dictionary representing the RLE of the mask, expected to contain
  a key `"counts"` with the RLE counts array.

# Returns
- `Int`: The total area (number of pixels) covered by the mask.
"""
function area_from_rle(rle::Dict{String, <:Any})::Int
	return sum(rle["counts"][1:2:end])
end


"""
    rle_to_mask(rle::Dict{String, <:Any}) -> AbstractArray{Bool}

Computes a binary mask from an uncompressed Run-Length Encoding (RLE) dictionary.

# Arguments
- `rle::Dict{String, <:Any}`: A dictionary representing the uncompressed RLE mask,
  expected to have keys:
  - `"size"`: a tuple or array with the height and width of the mask `(h, w)`.
  - `"counts"`: an array of integers representing the RLE counts.

# Returns
- `AbstractArray{Bool}`: A binary mask of shape `(h, w)`, where `true` indicates
  mask pixels and `false` background.
"""
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

"""
    mask_to_rle(tensor::AbstractArray) -> Vector{Dict{String, <:Any}}

Encodes a batch of binary masks into run-length encoding (RLE) format.

# Arguments
- `tensor::AbstractArray`: A 3D array of shape `(B, H, W)` where `B` is the batch size,
  `H` and `W` are the height and width of each binary mask. Values are assumed to be
  binary (0 or 1).

# Returns
- `Vector{Dict{String, <:Any}}`: A vector of length `B`, each element being a dictionary
  with keys:
  - `"size"`: a vector `[H, W]` representing the mask dimensions.
  - `"counts"`: a vector of integers encoding the run-length counts of the mask.
"""
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


"""
    batch_iterator(batch_size::Int, args::AbstractArray...)

Split multiple arrays into batches of size `batch_size`.

# Arguments
- `batch_size::Int`: The size of each batch.
- `args::AbstractArray...`: One or more arrays, all of the same length, to be batched simultaneously.

# Returns
- `Vector{Vector{AbstractArray}}`: A vector where each element is a vector containing slices (batches) of the input arrays.
"""
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

"""
    box_xyxy_to_xywh(box_xyxy::AbstractArray) -> AbstractArray

Convert bounding boxes from XYXY format to XYWH format.

# Arguments
- `box_xyxy::AbstractArray`: An array representing bounding boxes in XYXY format,
  where each box is defined as [x_min, y_min, x_max, y_max].

# Returns
- `AbstractArray`: A new array with the bounding boxes in XYWH format,
  where each box is defined as [x_min, y_min, width, height].
"""
function box_xyxy_to_xywh(box_xyxy::AbstractArray)::AbstractArray
	box_xywh = deepcopy(box_xyxy)
	box_xywh[3] -= box_xywh[1]
	box_xywh[4] -= box_xywh[2]
	return box_xywh
end


"""
    is_box_near_crop_edge(
        boxes::AbstractArray,
        crop_box::Vector{Int},
        orig_box::Vector{Int},
        atol::Float32 = 20.0f0,
    ) -> AbstractArray

Check whether any bounding boxes are near the edges of a crop box but not near the edges of the original image.

# Arguments
- `boxes::AbstractArray`: An array of bounding boxes in XYXY format, shape `(N, 4)` or similar.
- `crop_box::Vector{Int}`: The crop box coordinates `[x0, y0, x1, y1]`.
- `orig_box::Vector{Int}`: The original image box coordinates `[x0, y0, x1, y1]`.
- `atol::Float32=20.0f0`: Absolute tolerance distance to consider "near" an edge.

# Returns
- `AbstractArray`: A boolean array indicating for each box if it is near the crop box edge but not near the original image edge.
"""
function is_box_near_crop_edge(
	boxes::AbstractArray,
	crop_box::Vector{Int},
	orig_box::Vector{Int},
	atol::Float32 = 20.0f0,
)

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