using Lux
using CUDA
using Interpolations
using TensorOperations
using Einsum
using NPZ
using Random, Random123
using NNlib
using ImageTransformations
using ChainRulesCore

include("../global_functions.jl")
include("../src/prompt_encoder.jl")
include("../src/transformer.jl")
include("../src/mask_decoder.jl")
include("../src/image_encoder.jl")


"""
########################################################
# Sam:
########################################################

    struct Sam

Top-level container for the SAM (Segment Anything Model) components.

This struct encapsulates all the core modules required for performing segmentation:
image encoder, prompt encoder, and mask decoder. It also includes preprocessing parameters
such as pixel normalization statistics and image format.

# Fields
- `image_encoder::ImageEncoderViT`: Vision Transformer-based encoder for input images.
- `prompt_encoder::PromptEncoder`: Module to encode point, box, and mask prompts.
- `mask_decoder::MaskDecoder`: Decoder module that produces segmentation masks and IOU scores.
- `pixel_mean::AbstractArray{Float32}`: Mean values for pixel-wise normalization (shape `(3, 1, 1)`).
- `pixel_std::AbstractArray{Float32}`: Standard deviations for normalization (shape `(3, 1, 1)`).
- `mask_threshold::Float32`: Threshold for converting predicted logits to binary masks.
- `image_format::String`: Input image format, e.g., `"RGB"` or `"BGR"`.
- `device::Union{Nothing, String}`: Hardware target, `"cpu"` or `"gpu"`, auto-detected on construction.
"""
struct Sam
	image_encoder::ImageEncoderViT
	prompt_encoder::PromptEncoder
	mask_decoder::MaskDecoder
	pixel_mean::AbstractArray{Float32}
	pixel_std::AbstractArray{Float32}
	mask_threshold::Float32
	image_format::String
    device::Union{Nothing, String}
end

"""
    Sam(;
        image_encoder::ImageEncoderViT,
        prompt_encoder::PromptEncoder,
        mask_decoder::MaskDecoder,
        pixel_mean::Vector{Float64} = [123.675, 116.28, 103.53],
        pixel_std::Vector{Float64} = [58.395, 57.12, 57.375],
        mask_threshold::Float32 = 0.0f0,
        image_format::String = "RGB",
    )
        
Creates a new instance of the `Sam` model with the given components and configuration.

# Arguments
- `image_encoder::ImageEncoderViT`: The vision transformer used to extract image features.
- `prompt_encoder::PromptEncoder`: The encoder for sparse and dense prompts.
- `mask_decoder::MaskDecoder`: The decoder that generates masks and IOU predictions.
- `pixel_mean::Vector{Float64}`: Pixel mean for image normalization. Defaults to ImageNet mean.
- `pixel_std::Vector{Float64}`: Pixel std for image normalization. Defaults to ImageNet std.
- `mask_threshold::Float32`: Threshold to binarize mask logits. Defaults to `0.0f0`.
- `image_format::String`: Color format of the image. Defaults to `"RGB"`.

# Returns
- `Sam`: A fully initialized SAM model, ready for encoding and decoding tasks.
"""
function Sam(;
	image_encoder::ImageEncoderViT,
	prompt_encoder::PromptEncoder,
	mask_decoder::MaskDecoder,
	pixel_mean::Vector{Float64} = [123.675, 116.28, 103.53],
	pixel_std::Vector{Float64} = [58.395, 57.12, 57.375],
	mask_threshold::Float32 = 0.0f0,
	image_format::String = "RGB",
)

    if CUDA.has_cuda()
        device = "gpu"
    else
        device = "cpu"
    end

    pixel_mean = reshape(pixel_mean, (:, 1, 1))
    pixel_std = reshape(pixel_std, (:, 1, 1))

	return Sam(
		image_encoder,
		prompt_encoder,
		mask_decoder,
		Float32.(pixel_mean),
		Float32.(pixel_std),
		mask_threshold,
		image_format,
        device
	)
end


"""

    preprocess(self::Sam, x::AbstractArray)

Preprocesses the input image tensor by normalizing and padding it to match the encoder's expected dimensions.

This function normalizes the input using the model's `pixel_mean` and `pixel_std`, and applies zero-padding
so that the final image size matches the resolution expected by the `ImageEncoderViT`.

# Arguments
- `self::Sam`: The `Sam` model instance.
- `x::AbstractArray`: Input image or batch of images, with shape `(C, H, W)` or `(N, C, H, W)`.

# Returns
- `x::AbstractArray`: The normalized and padded image tensor.

# Notes
- Assumes images are in the same format (`RGB` or `BGR`) specified by `self.image_format`.
- The returned image has shape compatible with the `image_encoder`.
"""
function preprocess(self::Sam, x::AbstractArray)
	# Normalize colors
    if ndims(x) == 4
    	x = 
            (x .- 
            reshape(self.pixel_mean, (1, size(self.pixel_mean)...))) ./ 
            reshape(self.pixel_std, (1, size(self.pixel_std)...))

    else
    	x = (x .- self.pixel_mean) ./ self.pixel_std
    end

	h, w = size(x)[end-1:end]
	padh = self.image_encoder.img_size .- h
	padw = self.image_encoder.img_size .- w

    dims = (ndims(x), ndims(x)-1)

    x = NNlib.pad_zeros(x, (0, padw, 0, padh), dims = dims)
	return x
end


"""
    postprocess_masks(
        self::Sam;
        masks::AbstractArray,
        input_size::Tuple{Int, Int},
        original_size::Tuple{Int, Int},
    )

Postprocesses the predicted masks by resizing them back to the original image resolution.

This function first crops the masks to match the preprocessed input size, and then resizes them
to the original image resolution using bilinear interpolation.

# Arguments
- `self::Sam`: The `Sam` model instance.
- `masks::AbstractArray`: The predicted masks, of shape `(B, 1, H, W)` or `(1, H, W)` depending on context.
- `input_size::Tuple{Int, Int}`: The spatial resolution of the input image before padding.
- `original_size::Tuple{Int, Int}`: The original resolution of the image before any resizing or preprocessing.

# Returns
- `masks::AbstractArray`: The resized masks with shape `(B, 1, H_orig, W_orig)`.
"""
function postprocess_masks(
	self::Sam;
	masks::AbstractArray,
	input_size::Tuple{Int, Int},
	original_size::Tuple{Int, Int},
)
    NoInt = size(masks, 1) == 1 

    if NoInt
        masks = dropdims(masks; dims = 1)
        masks = imresize(
            permutedims(masks, (2, 3, 1)),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            method=Linear(),
        )
        masks = permutedims(masks, (3, 1, 2))
        masks = reshape(masks, (1, size(masks)...))
    
    else
    
        masks = imresize(
            permutedims(masks, (3, 4, 1, 2)),
            (self.image_encoder.img_size, self.image_encoder.img_size),
            method=Linear(),
        )
        masks = permutedims(masks, (3, 4, 1, 2))
    end

    masks = masks[:, :, 1:input_size[1], 1:input_size[2]]

    NoInt = size(masks, 1) == 1 
    
    if NoInt

        masks = dropdims(masks; dims = 1)
        masks = imresize(
            permutedims(masks, (2, 3, 1)),
            original_size,
            method=Linear(),
        )
        masks = permutedims(masks, (3, 1, 2))
        masks = reshape(masks, (1, size(masks)...))
    
    else

        masks = imresize(
            permutedims(masks, (3, 4, 1, 2)),
            original_size,
            method=Linear(),
        )

        masks = permutedims(masks, (3, 4, 1, 2))
    end

	return masks

end


"""
    (self::Sam)(
        batched_input::Vector{Dict{String, Any}};
        multimask_output::Bool,
    )
        
Performs forward pass for a batch of image prompts using the SAM model.

This method takes a list of input prompts (points, boxes, masks) and their corresponding images,
runs the image encoder, prompt encoder, and mask decoder, and returns post-processed masks and 
intermediate outputs.

# Arguments
- `self::Sam`: The `Sam` model instance.
- `batched_input::Vector{Dict{String, Any}}`: A list of input dictionaries, each containing:
    - `"image"`: The input image tensor (H×W×3 or batched).
    - `"original_size"`: The original image size as a tuple `(H, W)`.
    - `"point_coords"` (optional): Coordinates of prompts.
    - `"point_labels"` (optional): Corresponding labels for the prompt points.
    - `"boxes"` (optional): Bounding box prompts.
    - `"mask_inputs"` (optional): Mask input prompts.
- `multimask_output::Bool`: Whether to produce multiple masks per input or not.

# Returns
- `Vector{Dict{String, AbstractArray}}`: A list of dictionaries for each input containing:
    - `"masks"`: The binary output masks, thresholded.
    - `"iou_predictions"`: The predicted mask quality scores.
    - `"low_res_logits"`: The raw mask logits before upscaling.
"""
function (self::Sam)(
	batched_input::Vector{Dict{String, Any}};
	multimask_output::Bool,
)::Vector{Dict{String, AbstractArray}}

    return ChainRulesCore.ignore_derivatives() do
        input_images =
            cat(
                [preprocess(self, x["image"]) for x in batched_input]...;
                dims = 4,
            )

        input_images = permutedims(input_images, (4, 1, 2, 3))

        image_embeddings = self.image_encoder(input_images)

        outputs = []

        image_embeddings_v = 
            [image_embeddings[i, :, :, :] for i in 1:size(image_embeddings, 1)]

        for (image_record, curr_embedding) in zip(batched_input, image_embeddings_v)
            if "point_coords" in keys(image_record)
                points =
                    (image_record["point_coords"], image_record["point_labels"])
            else
                points = nothing
            end

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points = points,
                boxes = get(image_record, "boxes", nothing),
                masks = get(image_record, "mask_inputs", nothing),
            )

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=
                    Float32.(reshape(curr_embedding, (1, size(curr_embedding)...))),
                image_pe=Float32.(get_dense_pe(prompt_encoder)),
                sparse_prompt_embeddings=Float32.(sparse_embeddings),
                dense_prompt_embeddings=Float32.(dense_embeddings),
                multimask_output=multimask_output,
            )

            input_size = 
            (size(image_record["image"])[end-1], size(image_record["image"])[end])

            masks = postprocess_masks(
                self=self, 
                masks=low_res_masks,
                input_size=input_size,
                original_size=image_record["original_size"]
            )

            final_masks = masks .> self.mask_threshold

            push!(outputs, Dict(
                "masks" => final_masks,
                "iou_predictions" => iou_predictions,
                "low_res_logits" => low_res_masks,
            ))
        end
        return outputs
    end
end