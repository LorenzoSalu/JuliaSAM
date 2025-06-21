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
# MLP
########################################################

    struct MLP

Multi-layer perceptron (MLP) implemented using a sequence of dense layers and optional sigmoid output.

# Fields
- `num_layers::Int`: Number of layers in the network.
- `layers::Chain`: Sequence of `Dense` layers.
- `layers_ps::NamedTuple`: Parameters for the layers.
- `layers_st::NamedTuple`: States for the layers.
- `sigmoid_output::Bool`: Whether to apply a sigmoid activation on the final output.

"""
struct MLP
	num_layers::Int
	layers::Chain
	layers_ps::NamedTuple
	layers_st::NamedTuple
	sigmoid_output::Bool
end


"""
    MLP(; 
		input_dim::Int, 
		hidden_dim::Int, 
		output_dim::Int, 
		num_layers::Int, 
		sigmoid_output::Bool = false)

Constructs an MLP with the specified architecture.

# Arguments
- `input_dim`: Dimension of the input features.
- `hidden_dim`: Number of units in the hidden layers.
- `output_dim`: Dimension of the output layer.
- `num_layers`: Total number of layers (including output layer).
- `sigmoid_output` (optional): If `true`, applies a `sigmoid` to the final output. Default is `false`.

# Returns
- An instance of the `MLP` struct with initialized layers and parameters.

"""
function MLP(;
	input_dim::Int,
	hidden_dim::Int,
	output_dim::Int,
	num_layers::Int,
	sigmoid_output::Bool = false,
)

	num_layers = num_layers

	h = fill(hidden_dim, num_layers - 1)

	in_dims = [input_dim; h]
	out_dims = [h; output_dim]

	layers = Chain(
		[Dense(in_dim => out_dim)
		 for (in_dim, out_dim) in zip(in_dims, out_dims)]
	)

	rng = Random.MersenneTwister()
	layers_ps, layers_st = Lux.setup(rng, layers)

	sigmoid_output = sigmoid_output

	return MLP(
		num_layers,
		layers,
		layers_ps,
		layers_st,
		sigmoid_output,
	)
end


"""
    (self::MLP)(x::AbstractArray)

Applies the MLP to an input tensor `x`, supporting both 2D and 4D inputs.

# Arguments
- `x`: Input tensor of shape `(B, D)` or `(H, W, C, B)`.

# Returns
- The output of the MLP with the same shape format as the input.

# Description
- For 4D inputs, reshapes the input to 2D, processes it through the layers, and reshapes it back.
- Applies `relu` activation after each layer except the final one.
- Applies `sigmoid` to the final output if `sigmoid_output` is `true`.

"""
function (self::MLP)(x::AbstractArray)
	if ndims(x) == 4
		A, B, C, D = size(x)
		x = sam_reshape(x, (:, D))'

		for i in 1:self.num_layers
			if i < self.num_layers
				tmp, _ = self.layers[i](x, self.layers_ps[i], self.layers_st[i])
				x = relu.(tmp)
			else
				x, _ = self.layers[i](x, self.layers_ps[i], self.layers_st[i])
			end

		end

		if self.sigmoid_output
			x = sigmoid.(x)
		end

		D = size(x, 1)
		x = sam_reshape(x', (A, B, C, D))
	end

	if ndims(x) == 2
		x = x'

		for i in 1:self.num_layers
			if i < self.num_layers
				tmp, _ = self.layers[i](x, self.layers_ps[i], self.layers_st[i])
				x = relu.(tmp)
			else
				x, _ = self.layers[i](x, self.layers_ps[i], self.layers_st[i])
			end

		end

		if self.sigmoid_output
			x = sigmoid.(x)
		end

		x = x'
	end
	
	return x
end


"""
########################################################
# MaskDecoder
########################################################

    struct MaskDecoder

Transformer-based mask decoder module used for segmentation tasks.  
Combines token embeddings, upscaling convolutions, hypernetworks, and an IoU prediction head.

# Fields
- `transformer_dim::Int`: Dimensionality of transformer embeddings.
- `transformer::Any`: Transformer module used for decoding.
- `num_multimask_outputs::Int`: Number of predicted masks (excluding the best one).
- `iou_token::Embedding`: Learnable token for IoU prediction.
- `iou_token_ps::NamedTuple`: Parameters for `iou_token`.
- `iou_token_st::NamedTuple`: State for `iou_token`.
- `num_mask_tokens::Int`: Total number of mask tokens (including the best mask).
- `mask_tokens::Embedding`: Learnable tokens used to predict segmentation masks.
- `mask_tokens_ps::NamedTuple`: Parameters for `mask_tokens`.
- `mask_tokens_st::NamedTuple`: State for `mask_tokens`.
- `output_upscaling::Chain`: Decoder upscaling module using transposed convolutions and activations.
- `output_upscaling_ps::NamedTuple`: Parameters for the upscaling module.
- `output_upscaling_st::NamedTuple`: State for the upscaling module.
- `output_hypernetworks_mlps::Vector{MLP}`: Per-mask MLPs for generating dynamic mask weights.
- `iou_prediction_head::MLP`: MLP that predicts mask quality scores (IoU) given the output tokens.

"""
struct MaskDecoder
	transformer_dim::Int
	transformer::Any
	num_multimask_outputs::Int
	iou_token::Embedding
	iou_token_ps::NamedTuple
	iou_token_st::NamedTuple
	num_mask_tokens::Int
	mask_tokens::Embedding
	mask_tokens_ps::NamedTuple
	mask_tokens_st::NamedTuple
	output_upscaling::Chain
	output_upscaling_ps::NamedTuple
	output_upscaling_st::NamedTuple
	output_hypernetworks_mlps::Vector{MLP}
	iou_prediction_head::MLP
end

"""
    MaskDecoder(; 
		transformer_dim::Int, 
		transformer::Any, 
		num_multimask_outputs::Int = 3,
		activation::Function = gelu_exact, 
		iou_head_depth::Int = 3,
		iou_head_hidden_dim::Int = 256)

Creates a new instance of the `MaskDecoder` struct.

# Arguments
- `transformer_dim`: Embedding dimension for the transformer input/output.
- `transformer`: The transformer module used for decoding.
- `num_multimask_outputs`: Number of auxiliary masks predicted (default = 3).
- `activation`: Activation function used after upscaling layers.
- `iou_head_depth`: Number of layers in the IoU prediction MLP.
- `iou_head_hidden_dim`: Hidden layer size of the IoU prediction MLP.

# Returns
- A fully initialized `MaskDecoder` instance with learnable tokens, transformer, upscaling, and prediction heads.

"""
function MaskDecoder(;
	transformer_dim::Int,
	transformer::Any,
	num_multimask_outputs::Int = 3,
	activation::Function = gelu_exact,
	iou_head_depth::Int = 3,
	iou_head_hidden_dim::Int = 256,
)

	rng = Random.MersenneTwister()
	transformer_dim = transformer_dim
	transformer = transformer
	num_multimask_outputs = num_multimask_outputs

	iou_token = Lux.Embedding(1 => transformer_dim)
	iou_token_ps, iou_token_st = Lux.setup(rng, iou_token)

	num_mask_tokens = num_multimask_outputs + 1
	mask_tokens = Lux.Embedding(num_mask_tokens => transformer_dim)
	mask_tokens_ps, mask_tokens_st = Lux.setup(rng, mask_tokens)

	output_upscaling = Chain(
		ConvTranspose(
			(2, 2),
			transformer_dim => transformer_dim ÷ 4;
			stride = (2, 2),
            cross_correlation = true
		),
		LayerNorm2d(transformer_dim ÷ 4),
		x -> activation.(x),
		ConvTranspose(
			(2, 2),
			transformer_dim ÷ 4 => transformer_dim ÷ 8;
			stride = (2, 2),
            cross_correlation = true
		),
		x -> activation.(x),
	)

	output_upscaling_ps, output_upscaling_st = Lux.setup(rng, output_upscaling)

	output_hypernetworks_mlps =
		[MLP(
			input_dim = transformer_dim,
			hidden_dim = transformer_dim,
			output_dim = transformer_dim ÷ 8,
			num_layers = 3,
		) for i in 1:num_mask_tokens]

	iou_prediction_head = MLP(
		input_dim = transformer_dim,
		hidden_dim = iou_head_hidden_dim,
		output_dim = num_mask_tokens,
		num_layers = iou_head_depth,
	)

	return MaskDecoder(
		transformer_dim,
		transformer,
		num_multimask_outputs,
		iou_token,
		iou_token_ps,
		iou_token_st,
		num_mask_tokens,
		mask_tokens,
		mask_tokens_ps,
		mask_tokens_st,
		output_upscaling,
		output_upscaling_ps,
		output_upscaling_st,
		output_hypernetworks_mlps,
		iou_prediction_head,
	)

end


"""
    (self::MaskDecoder)(; 
		image_embeddings, 
		image_pe, 
		sparse_prompt_embeddings, 
		dense_prompt_embeddings, 
		multimask_output)

Applies the mask decoder to image and prompt embeddings, producing segmentation masks and IoU predictions.

# Keyword Arguments
- `image_embeddings`: Feature map from the image encoder.
- `image_pe`: Positional encoding for the image tokens.
- `sparse_prompt_embeddings`: Sparse token embeddings (e.g., points, boxes).
- `dense_prompt_embeddings`: Dense mask embeddings.
- `multimask_output`: If `true`, returns multiple masks including the best one; otherwise, only the best.

# Returns
- A tuple `(masks, iou_pred)` where:
    - `masks`: Tensor of shape `(B, K, H, W)` with K = 1 or `num_multimask_outputs`.
    - `iou_pred`: Predicted IoU scores for each output mask.

"""
function (self::MaskDecoder)(;
	image_embeddings::AbstractArray,
	image_pe::AbstractArray,
	sparse_prompt_embeddings::AbstractArray,
	dense_prompt_embeddings::AbstractArray,
	multimask_output::Bool,
)::Tuple{AbstractArray, AbstractArray}

	masks, iou_pred = predict_mask(
		self,
		image_embeddings = image_embeddings,
		image_pe = image_pe,
		sparse_prompt_embeddings = sparse_prompt_embeddings,
		dense_prompt_embeddings = dense_prompt_embeddings,
	)

	# Select the correct mask or masks for output
	if multimask_output
		mask_slice = 2:size(masks, 2)
	else
		mask_slice = 1:1
	end

	masks = masks[:, mask_slice, :, :]
	iou_pred = iou_pred[:, mask_slice]

	# Prepare output
	return masks, iou_pred
end


"""
    predict_mask(
		self::MaskDecoder; 
		image_embeddings, 
		image_pe, 
		sparse_prompt_embeddings, 
		dense_prompt_embeddings) -> (masks, iou_pred)

Predicts segmentation masks and their quality scores using the `MaskDecoder`.

This function processes the provided image features and prompt embeddings to generate segmentation masks via a transformer and hypernetwork-based decoding. It also predicts mask quality scores (IoU) for ranking the outputs.

# Arguments
- `self::MaskDecoder`: The `MaskDecoder` instance.
- `image_embeddings::AbstractArray`: Feature map from the image encoder, shape `(B, C, H, W)`.
- `image_pe::AbstractArray`: Positional encodings for image features, shape `(B, C, H, W)`.
- `sparse_prompt_embeddings::AbstractArray`: Sparse token embeddings (e.g., points, boxes), shape `(B, N_tokens, D)`.
- `dense_prompt_embeddings::AbstractArray`: Dense embeddings added to the image features, shape `(B, C, H, W)`.

# Returns
- `masks::AbstractArray`: Predicted segmentation masks of shape `(B, num_masks, H, W)`, where `num_masks = num_multimask_outputs + 1`.
- `iou_pred::AbstractArray`: IoU quality scores for each predicted mask, shape `(B, num_masks)`.

"""
function predict_mask(
	self::MaskDecoder;
	image_embeddings::AbstractArray,
	image_pe::AbstractArray,
	sparse_prompt_embeddings::AbstractArray,
	dense_prompt_embeddings::AbstractArray,
)::Tuple{AbstractArray, AbstractArray}

	batch_size = size(image_embeddings, 1)
	
	output_tokens =
		cat(self.iou_token_ps.weight, self.mask_tokens_ps.weight; dims = 2)'

	output_tokens = reshape(output_tokens, (1, size(output_tokens)...))
	output_tokens = repeat(output_tokens, batch_size, 1, 1)

	tokens = cat(output_tokens, sparse_prompt_embeddings, dims = 2)

	src = repeat(image_embeddings, size(tokens, 1), 1, 1, 1)

	src = src + dense_prompt_embeddings

	pos_src = repeat(image_pe, size(tokens, 1), 1, 1, 1)

	b, c, h, w = size(src)

	hs, src = self.transformer(
		image_embedding = src,
		image_pe = pos_src,
		point_embedding = tokens,
	)

	iou_token_out = hs[:, 1, :]
	mask_tokens_out = hs[:, 2:(1+self.num_mask_tokens), :]

	src = permutedims(src, (1, 3, 2))
	src = sam_reshape(src, (b, c, h, w))

	upscaled_embedding, _ = self.output_upscaling(
		permutedims(src, (3, 4, 2, 1)), 
		self.output_upscaling_ps,
		self.output_upscaling_st
		)

	upscaled_embedding = permutedims(upscaled_embedding, (4, 3, 1, 2))

	hyper_in_list::Vector{AbstractArray} = []
	for i in 1:self.num_mask_tokens
		push!(
			hyper_in_list,
			self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]),
		)
	end

    hyper_in_list = 
        [reshape(tensor, size(tensor, 1), 1, size(tensor, 2)) 
        for tensor in hyper_in_list]

	hyper_in = cat(hyper_in_list...; dims = 2)

	b, c, h, w = size(upscaled_embedding)
	upscaled_embedding = sam_reshape(upscaled_embedding, (b, c, h * w))

	masks = batched_mul(
        permutedims(hyper_in, (2, 3, 1)), 
        permutedims(upscaled_embedding, (2, 3, 1))
        )

	masks = sam_reshape(masks, (b, :, h, w))

	iou_pred = self.iou_prediction_head(iou_token_out)

	return masks, iou_pred
end