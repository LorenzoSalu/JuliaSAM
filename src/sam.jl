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

########################################################
# Sam:
########################################################
struct Sam
	image_encoder::ImageEncoderViT
	prompt_encoder::PromptEncoder
	mask_decoder::MaskDecoder
	pixel_mean::Vector{Float32}
	pixel_std::Vector{Float32}
	mask_threshold::Float32
	image_format::String
end

function Sam(;
	image_encoder::ImageEncoderViT,
	prompt_encoder::PromptEncoder,
	mask_decoder::MaskDecoder,
	pixel_mean::Vector{Float64} = [123.675, 116.28, 103.53],
	pixel_std::Vector{Float64} = [58.395, 57.12, 57.375],
	mask_threshold::Float32 = 0.0f0,
	image_format::String = "RGB",
)

	return Sam(
		image_encoder,
		prompt_encoder,
		mask_decoder,
		Float32.(pixel_mean),
		Float32.(pixel_std),
		mask_threshold,
		image_format,
	)
end


function preprocess(self::Sam, x::AbstractArray{Float32})

	# Normalize colors
	x = (x .- self.pixel_mean) ./ self.pixel_std

	h, w = size(x)[end-1:end]
	padh = self.image_encoder.img_size .- h
	padw = self.image_encoder.img_size .- w

	x = NNlib.pad_zeros(x, (0, 0, padh, 0, padw, 0))

	return x
end

function postprocess_masks(;
	self::Sam,
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