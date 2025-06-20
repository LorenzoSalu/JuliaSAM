using Lux
using CUDA

include("./src/prompt_encoder.jl")
include("./src/transformer.jl")
include("./src/mask_decoder.jl")
include("./src/image_encoder.jl")
include("./src/sam.jl")

function _build_sam(;
	encoder_embed_dim,
	encoder_depth,
	encoder_num_heads,
	encoder_global_attn_indexes,
	checkpoint = nothing,
)

	prompt_embed_dim = 256
	image_size = 1024
	vit_patch_size = 16
	image_embedding_size = image_size ÷ vit_patch_size

	image_encoder = ImageEncoderViT(
		depth = encoder_depth,
		embed_dim = encoder_embed_dim,
		img_size = image_size,
		mlp_ratio = 4.0f0,
		norm_layer = LayerNorm,
		num_heads = encoder_num_heads,
		patch_size = vit_patch_size,
		qkv_bias = true,
		use_rel_pos = true,
		global_attn_indexes = encoder_global_attn_indexes,
		window_size = 14,
		out_chans = prompt_embed_dim,
	)

	prompt_encoder = PromptEncoder(
		embed_dim = prompt_embed_dim,
		image_embedding_size = (image_embedding_size, image_embedding_size),
		input_image_size = (image_size, image_size),
		mask_in_chans = 16,
	)

	transformer = TwoWayTransformer(
		depth = 2,
		embedding_dim = prompt_embed_dim,
		mlp_dim = 2048,
		num_heads = 8,
	)

	mask_decoder = MaskDecoder(
		num_multimask_outputs = 3,
		transformer = transformer,
		transformer_dim = prompt_embed_dim,
		iou_head_depth = 3,
		iou_head_hidden_dim = 256,
	)

	sam = Sam(
		image_encoder = image_encoder,
		prompt_encoder = prompt_encoder,
		mask_decoder = mask_decoder,
		pixel_mean = [123.675, 116.28, 103.53],
		pixel_std = [58.395, 57.12, 57.375],
	)

	load_model_weights!(sam, checkpoint)

	return sam
end


function build_sam_vit_h(;checkpoint = nothing)
	return _build_sam(
		encoder_embed_dim = 1280,
		encoder_depth = 32,
		encoder_num_heads = 16,
		encoder_global_attn_indexes = (8, 16, 24, 32),
		checkpoint = checkpoint,
	)
end

build_sam = build_sam_vit_h








