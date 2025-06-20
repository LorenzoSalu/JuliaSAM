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
# MLP:
########################################################

struct MLP
	num_layers::Int
	layers::Chain
	layers_ps::NamedTuple
	layers_st::NamedTuple
	sigmoid_output::Bool
end

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


########################################################
# MaskDecoder:
########################################################

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