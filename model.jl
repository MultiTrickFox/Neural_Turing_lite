using Knet: @diff, Param, value, grad, params
using Knet: sigm, tanh, softmax


const memory_size = 5



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
    out      = show .* state

    memory += write .* interm_m

(out, state, memory)
end



in_size = 10
l_size = 10


seq_len = 10
hm_data = 20



lstm = Layer(in_size,l_size)

state = zeros(1,l_size)

memory = zeros(1,memory_size)


data = [[randn(1,in_size) for _ in 1:seq_len] for __ in 1:hm_data]



main(model, state, memory, data) =
begin

    for datapoint in data

        in_data = datapoint[1:end-1]
        out_data = datapoint[2:end]


        g = @diff begin

            outs = []
            for timestep in in_data

                out, state, memory = lstm(timestep, state, memory)
                push!(outs, out)

            end

        sum(sum([(e1-e2).^2 for (e1,e2) in zip(outs, out_data)]))
        end

        for param in params(lstm)
            param += .01 .* grad(g, param)
        end

    end

end


test(model, state, memory, data) =

    for timestep in data[1]

        out, state, memory = lstm(timestep, state, memory)

        # @show out
        # @show state
        @show memory
        println(" ")

    end



test(lstm, state, memory, data)

main(lstm, state, memory, data)

test(lstm, state, memory, data)
