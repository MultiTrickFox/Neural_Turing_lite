using Knet: @diff, Param, value, grad, params
using Knet: sigm, tanh, softmax






const memory_size = 5

const in_size       = 52
const hidden_layers = (12, 20)
const out_size      = 52

const turing_layers = (8) # TODO

    # TODO implement a char2char with a neural-turing-layers for write attention



mutable struct Layer

    wri::Param
    wrs::Param
    wrm::Param

    wwi::Param
    wws::Param
    wwm::Param

    wki::Param
    wks::Param
    wkm::Param

    wfi::Param
    wfs::Param
    wfm::Param

    wii::Param
    wis::Param
    wim::Param

    wsi::Param
    wss::Param
    wsm::Param

    w::Param

end


Layer(in_size,layer_size) =
begin

    # read
    wri = Param(randn(in_size, memory_size))
    wrs = Param(randn(layer_size, memory_size))
    wrm = Param(randn(memory_size, memory_size))

    # write
    wwi = Param(randn(in_size, memory_size))
    wws = Param(randn(layer_size, memory_size))
    wwm = Param(randn(memory_size, memory_size))

    # keep
    wki = Param(randn(in_size, layer_size))
    wks = Param(randn(layer_size, layer_size))
    wkm = Param(randn(memory_size, layer_size))

    # forget
    wfi = Param(randn(in_size, layer_size))
    wfs = Param(randn(layer_size, layer_size))
    wfm = Param(randn(memory_size, layer_size))

    # intermediate
    wii = Param(randn(in_size, layer_size))
    wis = Param(randn(layer_size, layer_size))
    wim = Param(randn(memory_size, layer_size))

    # show
    wsi = Param(randn(in_size, layer_size))
    wss = Param(randn(layer_size, layer_size))
    wsm = Param(randn(memory_size, layer_size))

    # intermediate memory
    w = Param(randn(layer_size, memory_size))

Layer(wri,wrs,wrm,wwi,wws,wwm,wki,wks,wkm,wfi,wfs,wfm,wii,wis,wim,wsi,wss,wsm,w)
end


(layer::Layer)(in, state, memory) =
begin

    read   = softmax(in * layer.wri + state * layer.wrs + memory * layer.wrm)
    write  = softmax(in * layer.wwi + state * layer.wws + memory * layer.wwm)
    attn_m = read .* memory

    keep   = sigm.(in * layer.wki + state * layer.wks + attn_m * layer.wkm)
    forget = sigm.(in * layer.wfi + state * layer.wfs + attn_m * layer.wfm)
    interm = tanh.(in * layer.wii + state * layer.wis + attn_m * layer.wim)
    show   = sigm.(in * layer.wsi + state * layer.wss + attn_m * layer.wsm)

    interm_m = tanh.(interm * layer.w)
    state    = forget .* state + keep .* interm
    out      = show .* tanh.(state)

    memory += write .* interm_m

(out, state, memory)
end



make(in_size, hiddens, out_size) =
begin
    model = []
    for i in 1:hm_layers
        if     i == 1         i, o = in_size, hiddens[1]
        elseif i == hm_layers i, o = hiddens[end], out_size
        else                  i, o = hiddens[i-1], hiddens[i]
        end
        push!(model, layer(i,o))

model
end


zero_state = [zeros(1,l_size) for l_size in (hiddens..., out_size)]
zero_memory = zeros(1,memory_size)


prop(data, (enc, dec); enc_state=zero_state, dec_state=zero_state, memory=zero_memory) =
begin
    ... = enc(data, enc_state, memory)
    ... = dec(data, dec_state, memory)
end
