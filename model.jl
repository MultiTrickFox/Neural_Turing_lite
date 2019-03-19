using Knet: @diff, Param, value, grad
using Knet: sigm, tanh, softmax


const memory_size = 50



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

Layer(wri,wrs,wrm,wwi,wws,wwm,wki,wks,wkm,wfi,wfs,wfm,wii,wis,wim,wsi,wss,wsm)
end


(layer::Layer)(in, state, memory) =
begin

    read   = softmax(in * layer.wri + state * layer.wrs + memory * layer.wrm)
    write  = softmax(in * layer.wwi + state * layer.wws + memory * layer.wwm)
    attn_m = read .* tanh.(memory)

    keep   = sigm.(in * layer.wki + state * layer.wks + attn_m * layer.wkm)
    forget = sigm.(in * layer.wfi + state * layer.wfs + attn_m * layer.wfm)
    interm = tanh.(in * layer.wii + state * layer.wis + attn_m * layer.wim)
    show   = sigm.(in * layer.wsi + state * layer.wss + attn_m * layer.wsm)

    state  = forget .* state + keep .* interm
    out    = show .* state

    memory += write

(out, state, memory)
end



main() =
begin

    seq_len = 2

    in_size = 10
    l_size = 5

    lstm = Layer(in_size,l_size)

    in_data = [randn(1,in_size) for _ in 1:seq_len]

    state = zeros(1,l_size)

    memory = zeros(1,memory_size)

    for timestep in in_data

    out, state, memory = lstm(timestep, state, memory)
        @show out
        @show state
        @show memory

    end

end


main()
