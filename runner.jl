include("model.jl")

using Knet: Param, @diff, value, grad


const seq_len   = 10
const hm_data   = 20
const hm_epochs = 10


model = make()
data = [[randn(1,in_size) for _ in 1:seq_len] for __ in 1:hm_data]


runner() =
for i in 1:hm_epochs
    l = 0
    for seq in data
        input = seq[1:end-1]
        label = seq[2:end]
        response, state, memory = prop(model, input)
        l += loss(response, label)
    end
    @show i, l
end ; runner()





# initial test code.

# in_size = 10
# l_size = 10
#
# seq_len = 10
# hm_data = 20
#
# lstm = Layer(in_size,l_size)
#
# state = zeros(1,l_size)
#
# memory = zeros(1,memory_size)
#
#
# data = [[randn(1,in_size) for _ in 1:seq_len] for __ in 1:hm_data]
#
#
#
# main(model, state, memory, data) =
# begin
#
#     for datapoint in data
#
#         in_data = datapoint[1:end-1]
#         out_data = datapoint[2:end]
#
#
#         g = @diff begin
#
#             outs = []
#             for timestep in in_data
#
#                 out, state, memory = lstm(timestep, state, memory)
#                 push!(outs, out)
#
#             end
#
#         sum(sum([(e1-e2).^2 for (e1,e2) in zip(outs, out_data)]))
#         end
#
#         # for param in params(lstm)
#         #     param -= .01 .* grad(g, param)
#         # end
#
#         for field in fieldnames(Layer)
#             setfield!(model, field, Param(getfield(model, field) - .01 .* grad(g, getfield(model, field))))
#         end
#
#     end
#
# end
#
#
# test(model, state, memory, data) =
#
#     for timestep in data[1]
#
#         out, state, memory = lstm(timestep, state, memory)
#
#         # @show out
#         # @show state
#         @show memory
#         println(" ")
#
#     end
#
#
#
# test(lstm, state, memory, data)
#
#
# println("---")
#
# main(lstm, state, memory, data)
#
# println("---")
#
#
# test(lstm, state, memory, data)
