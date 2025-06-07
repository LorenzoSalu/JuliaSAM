using Lux
using CUDA
using Interpolations
using TensorOperations
using Einsum
using NNlib
using ImageTransformations

include("../global_functions.jl")
include("../utils/amg.jl")


struct ResizeLongestSize
	target_length::Int
end

function ResizeLongestSize(;
	target_length::Int,
)
	"""
	Resizes images to the longest side 'target_length', as well as provides
	methods for resizing coordinates and boxes. Provides methods for
	transforming both numpy array and batched torch tensors.
	"""
	return ResizeLongestSize(
		target_length,
	)
end



function apply_image(
	self::ResizeLongestSize,
	image::AbstractArray{UInt8, 3})
	"""
	Expects an array with shape HxWxC in uint8 format.
	"""
	target_size = get_preprocess_shape(
		size(image, 1),
		size(image, 2),
		self.target_length,
	)
	resized_image = imresize(image, target_size, method=Linear())
	return trunc.(Int, resized_image)
end


function apply_coords(
	self::ResizeLongestSize,
	coords::AbstractArray{<:Real},
	original_size::Tuple{Int, Int},
)::AbstractArray{Float64}
	"""
	Expects a array of length 2 in the final dimension. Requires the
	original image size in (H, W) format.
	"""
	old_h, old_w = original_size
	new_h, new_w = get_preprocess_shape(old_h, old_w, self.target_length)

	coords = Float64.(deepcopy(coords))

    colons = ntuple(_ -> :, ndims(coords) - 1)

	coords[colons..., 1] .= coords[colons..., 1] .* (new_w / old_w)
	coords[colons..., 2] .= coords[colons..., 2] .* (new_h / old_h)

	return coords
end



function apply_boxes(
    self::ResizeLongestSize,
    boxes::AbstractArray{<:Real},
    original_size::Tuple{Int, Int},
    )
    """
    Expects an array shape Bx4. Requires the original image size
    in (H, W) format.
    """
    reshaped_boxes = sam_reshape(boxes, (:, 2, 2))
    transformed_boxes = apply_coords(self, reshaped_boxes, original_size)
    final_boxes = sam_reshape(transformed_boxes, (:, 4))
    return final_boxes
end

function apply_image_(
    self::ResizeLongestSize,
    image::AbstractArray{<:Real, 4}
)
    """
    Expects batched images with shape BxCxHxW and float format. This
    transformation may not exactly match apply_image. apply_image is
    the transformation expected by the model.
    """
    target_size = 
        get_preprocess_shape(size(image, 3), size(image, 4), self.target_length)

    resized_image = 
        imresize(image, (size(image, 1), size(image, 2), target_size...))

    return Float32.(resized_image)
end


function get_preprocess_shape(
	oldh::Int,
	oldw::Int,
	long_side_length::Int,
)::Tuple{Int, Int}
	"""
	Compute the output size given input size and target long side length.
	"""
	scale = long_side_length * 1.0 / max(oldh, oldw)
	newh = oldh * scale
	neww = oldw * scale
	neww = trunc(Int, neww + 0.5)
	newh = trunc(Int, newh + 0.5)
	return (newh, neww)
end


# end of file: test_transform.jl
# -----------------------------