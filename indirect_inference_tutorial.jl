
include(joinpath("estimation", "indirect_inference.jl"))

using Statistics
using Distributions
using LinearAlgebra
using NLopt
using Main.IndirectInference



# Need DGP - f(structural parameters, covariates)
# Need Auxiliary Estimation routine (aux model is embedded in this) - f(lhs, rhs variables)

# True model: y = x^2 β + ϵ
# Auxiliary model: y = [1, x, x^2, x^3] * b + ν = Z b + ν
# binding fcn: β -> b = [b1, b2, b3, b4]': b(β) = [0, 0, β, 0]'
# Aux model: y = Z * b(β) + ν = [1, x, x^2, x^3] * [0, 0, β, 0]' + ν
                            # = β x^2 + ϵ
# W* = [(∂b/∂β)*(∂b/∂β)']^{-1} = [matrix of zeros with 1 at loc 3,3]

# Routine:
# 1. estimate the auxiliary model on the actual data -> b0
# 2. loop here
#  2.a. β -> simulate from the structural model (true model)
#  2.b. estimate the aux model parameters on the simulated data -> b
#  2.c. do this until you find the closest match b ≈ b0
# How do we find this match?
# MSE = (b(β) - b0(β0))' * W * (b(β) - b0(β0))
# The β that gives the lowest MSE is our structural estimate


function dgp(β, X)
    # β must be a vector/1-dimensional array
    # X is an array of covariates, etc.
    ϵ = rand(Normal(0,1), size(X)[1])
    y = X.^2 * β + ϵ
    return(y)
end

function aux(Y, X)
    # Y and X are Arrays/ matrices for the lhs and rhs variables
    N = size(X)[1]
    Z = [ones(N) X X.^2 X.^3]
    out = OLS(Y, Z)
    return(out)
end

N = 200
X0 = rand(Normal(0,1), (N,1))  # this was the problem - I didn't specify the size of X correctly (and this is an example of why julia is not user friendly)
β0 = 1.0
Y0 = dgp(β0, X0)

temp_b0 = aux(Y0, X0) # 4x1 vector of estimates for b

# First let's use a grid search
β_grid = [β0 .+ i for i in (-.5:.025:.5)]
est1 = indirect_inference(Y0=Y0, X0=X0, true_model=dgp, aux_estimation=aux, search="grid", β_grid=β_grid)
est1 # 1 x 4 tuple containing:
# 1. MSE value at the minimum
# 2. argmin of the MSE - this is the estimate of the structural parameter
# 3. error code - "FORCED_STOP" means something broke.
# 4. number of iterations

# Next, let's use a nonlinear optimizer.  We will first perform unconstrained optimization 
# using the gradient free Nelder-Mead optimization algorithm (this is the default).
# For the nonlinear optimizer NLopt, β must be passed as an array even for univariate problems.
# Also, note that the argument search="NL" is optional, as "NL" is the default
est2 = indirect_inference(Y0=Y0, X0=X0, true_model=dgp, aux_estimation=aux, β_init=[β0], search="NL")
est2[2] # structural estimate


# Now let's apply some constraints in the form of lower and upper bounds
NLoptOptions = NLopt_options(lb = [-10.0], ub = [10.0])
est3 = indirect_inference(Y0=Y0, X0=X0, true_model=dgp, aux_estimation=aux, β_init=[β0], NLoptOptions = NLoptOptions)
# Note that est3[2] will probably be different from est2[2], etc. due to the simulations implicit in the estimation procedure.


# Now let's add in a weighting matrix.  Since we know the optimal weighting matrix, let's use it.
W = kron([0, 0, 1, 0], [0, 0, 1, 0]')
est4 = indirect_inference(Y0=Y0, X0=X0, true_model=dgp, aux_estimation=aux, β_init=[β0], NLoptOptions = NLoptOptions, W=W)


# Finally, let's use a gradient-required optimization algorithm.
# Note that we need to specify ∂b/∂β in the aux estimation routine,
# and we need to specify the argument "gradient=true"

function aux_grad(Y, X)  
    a = aux(Y, X)
    ∂b = [0, 0, 1, 0]
    return a, ∂b
end

NLoptOptions = NLopt_options(lb = [-10.0], ub = [10.0], alg=:LD_LBFGS)
est5 = indirect_inference(Y0=Y0, X0=X0, true_model=dgp, aux_estimation=aux_grad, β_init=[β0], NLoptOptions=NLoptOptions, W=W, gradient=true)




#####################
# other notes:

function dgp2(β, X)
    # B = Dict(:α => β[1], :γ => β[2])
    # α = B[:α]
    α = β[1]
    γ = β[2]
    ϵ = 
    ν = 
    y1 = α * X[1] + ϵ
    y2 = β[2] * X[2] .^ β[3] + ν
    Y = [y1; y2]
    return(Y)
end
