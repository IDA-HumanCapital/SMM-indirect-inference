# NOTE:
# If you want to use structs, etc. please use a wrapper function.
# This module is in active development, so please talk to me before modifying it.


# TO DO:
# Allow weighting matrix to depend upon b or β
# Add in MCMC for bayesian estimation

###########################################
###########################################
# Implements Indirect Inference and SMM
#
# Indirect Inference Estimation
# Usage:
# indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, kwargs)
# 
# Requried key word arguments:
#   Y0 is the outcome variable from the data
#   X0 is the list of covariates from the data
#   true_model is a generic function Y=f(β, X) describing the structural model
#   aux_estimation is a generic function b_hat = g(Y, X) describing the auxiliary model and estimation routine
# 
# Semi-Optional key word arguments:
#   search ∈ {"NL", "grid"}, Default is "NL"
#   β_init: initial value for the "NL" estimation routine (required for search="NL")
#   β_grid: grid of βs for the "grid" search estimation routine (required for search="grid")
#   J: number of times to simulate from the true model
#   NLoptOptions: Options passed to the nonlinear optimizer NLopt (see NLopt_Options() below for details)
#   W: the weighting matrix (I may change this later to get it from aux_estimation)  Note that the
#       optimal weighting matrix depends upon the binding function in general.  When the binding
#       function is not known, results are not expected to be efficient.
#   gradient: Must be =true if using search="NL" and a gradient based search algorithm.
# 
# Output:
#   array of the structural parameter estimates
# 
# Examples:
#   ii2 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
#   ii2b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β_init, NLoptOptions=NLoptOptions)
#   See indirect_inference_examples.jl for complete examples
#
###########################################
###########################################
#
# Indirect Inference Inference via the parameteric bootstrap
# Usage:
#   iibootstrap(;β, X0, true_model, aux_estimation, J_bs=9, kwargs...)
#
# Note:
#   iibootstrap() should be called with the same arguments as indirect_inference()
#
# Arguments:
#   J_bs: the number of bootstrap samples
#   β: the parameter used to simulate the model, 
#       usually the null hypothesis value β_0 if testing against a null
#       or the estimates β_hat
#   all other arguments are the same as indirect_inference()
#
# Output:
#   Array of size J_bs x K for the J_bs estimates
#
# Examples:
#   ii2 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
#   ii2bs = iibootstrap(β=ii2, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, J_bs=99)
#
# Note:
#   Not knowing the binding function that links the structural and auxiliary model parameters makes inference complicated.
#   Hence, this implements a parameteric bootstrap.
#   The trade-off is that this is computationally intensive.
#
###########################################
###########################################
#
# Setting options for the NLopt Optimizer
# Usage:
#   NLoptions = NLopt_options(lb=Nothing, ub=Nothing, cons_ineq=Nothing, alg=:LN_NELDERMEAD, xtol=1e-4)
#
# Arguments:
#   lb: array of lower bounds
#   ub: array of upper bounds
#   cons_ineq: array of generic functions representing inequality constraints
#   alg: optimization algorithm (NelderMead is the default)
#   xtol: x tolerance
#
# Output:
#   a struct of options to pass to the NLopt wrapper function
#
# More details regarding NLopt can be found here:
#   https://github.com/JuliaOpt/NLopt.jl
#   https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
#
###########################################




module IndirectInference

# using DataFrames
# using Parameters
using Statistics
using Distributions
using LinearAlgebra
using GLM
# using Optim, NLSolversBase
using NLopt


export OLS, indirect_inference, iibootstrap, NLopt_wrapper, NLopt_options

OLS = ((Y,X) -> inv(transpose(X) * X) * (transpose(X) * Y))


function auxiliary_model_sim(βi, grad, b0, X0, true_model, aux_estimation; kwargs...)
    # fit the auxiliary model 
    if :gradient in keys(kwargs)
        gradient = kwargs[:gradient]
    else
        gradient = false
    end
    if length(grad) > 0
        if gradient == false
            throw(ArgumentError("gradient function required for this optimization algorithm"))
        end
        # just ignore gradient for now; it's a required arg for NLopt, 
        # but return an error if using an optimization method that requires a gradient
        # this appears to cause NLopt to force stop, so the message above is not displayed.
        # when implemented, gradient should supply ∂b/∂β (and ∂W/∂β in the future) to allow calculation of ∂MSE/∂β
    end
    K = size(b0)[1]
    if :J in keys(kwargs)
        J = kwargs[:J]
    else
        J = 500
    end
    XJ = repeat(X0, J)
    Y_star = true_model(βi, XJ) # simulate from true model
    if :W in keys(kwargs)
        W = kwargs[:W]
    else
        W = I(K)
    end
    # b = aux_estimation(Y_star, XJ) # estimate the auxiliary model on simulations
    if gradient == false
        b = aux_estimation(Y_star, XJ)
    elseif gradient == true
        b, ∂b = aux_estimation(Y_star, XJ)
        if length(grad) > 0
            grad[:] = 2 .* ∂b' * W * (b .- b0)
        end
    end
    MSE = (transpose(b - b0) * W * (b - b0))[1] # why does julia make assignment difficult??
    return MSE
end


function indirect_inference(;Y0, X0, true_model, aux_estimation, NLoptOptions=Nothing, kwargs...)
    # simple indirect inference using Optim
    # and a linearized model
    N = length(Y0)
    b0 = aux_estimation(Y0, X0) # estimate auxiliary model on data
    if :gradient in keys(kwargs)
        if kwargs[:gradient] == true
            b0 = b0[1]
        end
    end
    if :search in keys(kwargs)
        search = kwargs[:search]
    else
        search = "NL"
    end
    if search == "grid"
        MSE = (βi -> auxiliary_model_sim(βi, [], b0, X0, true_model, aux_estimation; kwargs...))
        β_grid = kwargs[:β_grid]
        MSE0 = Inf # something large
        for βi in β_grid
            MSE1 = MSE(βi)
            if (MSE1 < MSE0)
                MSE0 = MSE1
                if length(βi) == 1
                    β_hat = [βi]
                else
                    β_hat = βi
                end
            end
        end    
        minf = MSE0
        ret = "grid search"
        numevals = length(β_grid)
    elseif search == "NL"
        MSE = ((βi, g) -> auxiliary_model_sim(βi, g, b0, X0, true_model, aux_estimation; kwargs...))
        β_init = kwargs[:β_init]
        (minf, β_hat, ret, numevals) = NLopt_wrapper(; fn=MSE, init_val=β_init, NLoptOptions=NLoptOptions)
    end
    return(minf, β_hat, ret, numevals)
end


function iibootstrap(;β, X0, true_model, aux_estimation, NLoptOptions=Nothing, J_bs=9, kwargs...)
    # implements a parameteric bootstrap with the indirect inference estimation function
    # Note that this is computationally intensive
    K = length(β)
    storage = zeros(J_bs, K)
    for j in 1:J_bs
        Y = true_model(β, X0)
        res = indirect_inference(;Y0=Y, X0=X0, true_model=true_model, aux_estimation=aux_estimation, NLoptOptions=NLoptOptions, kwargs...)
        est = res[2]
        storage[j, :] .= [est...]
    end
    return(storage)
end


# function sim(β, N, x_fn, y_fn, estimation_fn, aux_estimator, β_grid, J_outer=500, J_inner=500)
#     # do stuff in here
#     K = length(β)
#     storage = zeros(J_outer, K)
#     for j in 1:J_outer
#         X = x_fn((N,K))
#         Y = y_fn(β,X)
#         estimates = estimation_fn(Y0=Y, X0=X, true_model=y_fn, aux_estimation=aux_estimator, J=J_inner, β_grid=β_grid)
#         storage[j, :] .= estimates
#     end
#     return(storage)
# end


function NLopt_wrapper(; fn, init_val, NLoptOptions=Nothing)
    if NLoptOptions == Nothing
        NLoptOptions = NLopt_options()
    end
    opt = Opt(NLoptOptions.alg, length(init_val))
    if NLoptOptions.lb != Nothing
        opt.lower_bounds = NLoptOptions.lb
    end
    if NLoptOptions.ub != Nothing
        opt.upper_bounds = NLoptOptions.ub
    end
    opt.xtol_rel = NLoptOptions.xtol

    opt.min_objective = fn
    if NLoptOptions.cons_ineq != Nothing
        for i in 1:length(NLoptOptions.cons_ineq)
            inequality_constraint!(opt, NLoptOptions.cons_ineq[i], 1e-8)
        end
    end
    
    (minf,minx,ret) = optimize(opt, init_val)
    numevals = opt.numevals # the number of function evaluations
    # println("got $minf at $minx after $numevals iterations (returned $ret)")
    return(minf, minx, ret, numevals)
end


struct NLopt_options_obj
    lb
    ub
    cons_ineq
    alg
    xtol
    NLopt_options_obj(lb, ub, cons_ineq, alg, xtol) = new(lb, ub, cons_ineq, alg, xtol)
end


function NLopt_options(; lb=Nothing, ub=Nothing, cons_ineq=Nothing, alg=:LN_NELDERMEAD, xtol=1e-4)
    out = NLopt_options_obj(lb, ub, cons_ineq, alg, xtol)
end


end # end module
