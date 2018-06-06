using Yao
uniform_state(num_bit) = register(ones(Complex128, 1<<num_bit)/sqrt(1<<num_bit))
