include(joinpath("estimation", "indirect_inference.jl"))

using NLopt
using Main.IndirectInference

function myfunc(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 0
        grad[2] = 0.5/sqrt(x[2])
    end
    return sqrt(x[2])
end


function myconstraint(x::Vector, grad::Vector, a, b)
    if length(grad) > 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end


cons0 = [(x,g) -> myconstraint(x,g,2,0), (x,g) -> myconstraint(x,g,-1,1)]
a = NLopt_options(lb=[-Inf, 0], cons_ineq=cons0)


NLopt_wrapper(fn=myfunc, init_val=[1.234, 5.678], NLoptOptions=a)



