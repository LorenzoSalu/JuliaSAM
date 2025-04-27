using Flux

include("../modeling/image_encoder.jl")


# Parametri di configurazione
kernel_size = (32, 32)
stride = (16, 16)
padding = (0, 0)
in_chans = 3
embed_dim = 1024


# Creazione di un'istanza di PatchEmbed
patch_embed = PatchEmbed(
    kernel_size = kernel_size,
    stride = stride,
    padding = padding,
    in_chans = in_chans,
    embed_dim = embed_dim
)

# Dimensioni dell'immagine di input
batch_size = 1
img_height = 64
img_width = 64

# Creazione di un tensore di input casuale
input_tensor = randn(Float32, img_height, img_width, in_chans, batch_size)

println("Dimensione input_tensor: ", size(input_tensor))
println("patch_embed.proj: ", patch_embed.proj)

# Esecuzione del forward pass
output_tensor = patch_embed(input_tensor)


expected_height = img_height ÷ stride[1]
expected_width = img_width ÷ stride[2]

# Stampa la forma dell'output
println("Forma dell'input: ", size(input_tensor))
println("Forma dell'output: ", size(output_tensor))

# Stampa la forma attesa dell'output
println(
    "Expected output: ", 
    (batch_size, expected_height, expected_width, embed_dim)
    )


@assert size(output_tensor) == (
    batch_size,
    expected_height,
    expected_width,
    embed_dim
    ) "Errore con la configurazione. 
    output shape: $(size(output_tensor))
    expected shape: $((batch_size, expected_height, expected_width, embed_dim))"


println("Test superato con successo!")