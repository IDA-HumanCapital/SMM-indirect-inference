include(joinpath("estimation", "indirect_inference.jl"))

using NLopt
using Main.IndirectInference

# Example 1
# nonlinear inequality constraints
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


(minf, minx, ret, numevals) = NLopt_wrapper(fn=myfunc, init_val=[1.234, 5.678], NLoptOptions=a)
println("got $minf at $minx after $numevals iterations (returned $ret)")



# Example 2
# unconstrained
function myfunc2(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 2 * x[1]
        grad[2] = 2 * x[2]
    end
    return x[1]^2 + x[2]^2
end

(minf, minx, ret, numevals) = NLopt_wrapper(fn=myfunc2, init_val=[1, 5])
println("got $minf at $minx after $numevals iterations (returned $ret)")


function myfunc3(x::Vector, grad::Vector)
    if length(grad) > 0
        grad[1] = 2 * x[1]
    end
    return x[1]^2
end

(minf, minx, ret, numevals) = NLopt_wrapper(fn=myfunc3, init_val=[1])
println("got $minf at $minx after $numevals iterations (returned $ret)")
