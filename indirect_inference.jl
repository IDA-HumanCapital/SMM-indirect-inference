# NOTE:
# If you want to use structs, etc. please use a wrapper function.
# Please do not modify this module without talking to me first.

###########################################
###########################################
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
# Optional key word arguments:
#   search ∈ {"NL", "grid"}, Default is "NL"
#   β_init: initial value for the "NL" estimation routine
#   β_grid: grid of βs for the "grid" search estimation routine
#   J: number of times to simulate from the true model
# 
# Output:
#   array of the structural parameter estimates
# 
# Examples:
#   ii2 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
#   ii2b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL")
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


module IndirectInference

# using DataFrames
using Statistics
using Distributions
# using Parameters
using LinearAlgebra
# using GLM
using Optim, NLSolversBase



export OLS, indirect_inference, iibootstrap

OLS = ((Y,X) -> inv(transpose(X) * X) * (transpose(X) * Y))


# function auxiliary_model_sim_old(βi, b0, X0, true_model, aux_estimation, J=500)
#     # fit the auxiliary model 
#     K = size(b0)[1]
#     b_star = zeros(K, J)
#     for j in 1:J
#         Y_star = true_model(βi, X0) # simulate from true model
#         estimates = aux_estimation(Y_star, X0) # estimate the auxiliary model on simulations
#         b_star[:, j] = estimates # collect results
#     end
#     #is b_star close to b0?
#     b = sum(b_star, dims=2) / J
#     MSE = (transpose(b - b0) * (b - b0))[1] # why does julia make assignment difficult??
#     return MSE
# end


function auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation; J=500)
    # fit the auxiliary model 
    K = size(b0)[1]
    XJ = repeat(X0, J)
    Y_star = true_model(βi, XJ) # simulate from true model
    b = aux_estimation(Y_star, XJ) # estimate the auxiliary model on simulations
    MSE = (transpose(b - b0) * (b - b0))[1] # why does julia make assignment difficult??
    return MSE
end


# function indirect_inference_grid(Y0, X0, β_grid, true_model, aux_estimation, J=500)
#     # simple indirect inference using a grid search
#     # and a linearized model
#     N = length(Y0)
#     K = size(X0,2)
#     b0 = aux_estimation(Y0, X0) # estimate auxiliary model on data
#     β_hat = zeros(K, 1)
#     MSE0 = Inf # something large
#     for βi in β_grid
#         MSE = auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation, J)
#         if (MSE < MSE0)
#             MSE0 = MSE
#             β_hat = βi
#         end
#     end
#     return(β_hat)
# end


# function indirect_inference_optim(Y0, X0, true_model, aux_estimation, J=500)
#     # simple indirect inference using Optim
#     # and a linearized model
#     N = length(Y0)
#     K = size(X0,2)
#     b0 = aux_estimation(Y0, X0) # estimate auxiliary model on data
#     β_hat = zeros(K, 1)
#     MSE = (βi -> auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation, J))
#     if (K==1)
#         print("Univariate functions not supported at this time")
#         β_hat_out = 0        
#     else
#         results = optimize(MSE, β_hat)
#         β_hat_out = Optim.minimizer(results)
#     end
#     return(β_hat_out)
# end


function indirect_inference(;Y0, X0, true_model, aux_estimation, kwargs...)
    # simple indirect inference using Optim
    # and a linearized model
    N = length(Y0)
    K = size(X0,2)
    b0 = aux_estimation(Y0, X0) # estimate auxiliary model on data
    if :J in keys(kwargs)
        J = kwargs[:J]
    else
        J = 500
    end
    MSE = (βi -> auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation, J=J))
    if :search in keys(kwargs)
        search = kwargs[:search]
    else
        search = "NL"
    end
    if search == "grid"
        β_grid = kwargs[:β_grid]
        MSE0 = Inf # something large
        for βi in β_grid
            MSE1 = MSE(βi)
            if (MSE1 < MSE0)
                MSE0 = MSE1
                β_hat = βi
            end
        end    
    elseif search == "NL"
        if :β_init in keys(kwargs)
            β_init = kwargs[:β_init]
        else
            β_init = zeros(K, 1)
        end    
        if (K==1)
            print("Univariate functions not supported at this time")
            β_hat = β_init        
        else
            results = optimize(MSE, β_init)
            β_hat = Optim.minimizer(results)
        end
    end
    return(β_hat)
end


function iibootstrap(;β, X0, true_model, aux_estimation, J_bs=9, kwargs...)
    # implements a parameteric bootstrap with the indirect inference estimation function
    # Note that this is computationally intensive
    K = length(β)
    storage = zeros(J_bs, K)
    for j in 1:J_bs
        Y = true_model(β, X0)
        est = indirect_inference(;Y0=Y, X0=X0, true_model=true_model, aux_estimation=aux_estimation, kwargs...)
        # estimates = indirect_inference(Y0=Y, X0=X0, true_model=true_model, aux_estimation=aux_estimation, kwargs...)
        storage[j, :] .= [est...]
    end
    return(storage)
end


function sim(β, N, x_fn, y_fn, estimation_fn, aux_estimator, β_grid, J_outer=500, J_inner=500)
    # do stuff in here
    K = length(β)
    storage = zeros(J_outer, K)
    for j in 1:J_outer
        X = x_fn((N,K))
        Y = y_fn(β,X)
        estimates = estimation_fn(Y0=Y, X0=X, true_model=y_fn, aux_estimation=aux_estimator, J=J_inner, β_grid=β_grid)
        storage[j, :] .= estimates
    end
    return(storage)
end


end # end module
