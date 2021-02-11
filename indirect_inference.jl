

# If you want to use structs, etc. please use a wrapper function.
# Please do not modify this module without talking to me first.

module IndirectInference

# using DataFrames
using Statistics
using Distributions
# using Parameters
using LinearAlgebra
# using GLM
using Optim, NLSolversBase



export OLS, indirect_inference_grid, indirect_inference_optim

OLS = ((Y,X) -> inv(transpose(X) * X) * (transpose(X) * Y))


function auxiliary_model_sim_old(βi, b0, X0, true_model, aux_estimation, J=500)
    # fit the auxiliary model 
    K = size(b0)[1]
    b_star = zeros(K, J)
    for j in 1:J
        Y_star = true_model(βi, X0) # simulate from true model
        estimates = aux_estimation(Y_star, X0) # estimate the auxiliary model on simulations
        b_star[:, j] = estimates # collect results
    end
    #is b_star close to b0?
    b = sum(b_star, dims=2) / J
    MSE = (transpose(b - b0) * (b - b0))[1] # why does julia make assignment difficult??
    return MSE
end


function auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation, J=500)
    # fit the auxiliary model 
    K = size(b0)[1]
    XJ = repeat(X0, J)
    Y_star = true_model(βi, XJ) # simulate from true model
    b = aux_estimation(Y_star, XJ) # estimate the auxiliary model on simulations
    MSE = (transpose(b - b0) * (b - b0))[1] # why does julia make assignment difficult??
    return MSE
end


function indirect_inference_grid(Y0, X0, β_grid, true_model, aux_estimation, J=500)
    # simple indirect inference using a grid search
    # and a linearized model
    N = length(Y0)
    K = size(X0,2)
    b0 = aux_estimation(Y0, X0) # estimate auxiliary model on data
    β_hat = zeros(K, 1)
    MSE0 = Inf # something large
    for βi in β_grid
        MSE = auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation, J)
        if (MSE < MSE0)
            MSE0 = MSE
            β_hat = βi
        end
    end
    return(β_hat)
end


function indirect_inference_optim(Y0, X0, true_model, aux_estimation, J=500)
    # simple indirect inference using Optim
    # and a linearized model
    N = length(Y0)
    K = size(X0,2)
    b0 = aux_estimation(Y0, X0) # estimate auxiliary model on data
    β_hat = zeros(K, 1)
    MSE = (βi -> auxiliary_model_sim(βi, b0, X0, true_model, aux_estimation, J))
    if (K==1)
        print("Univariate functions not supported at this time")
        β_hat_out = 0        
    else
        results = optimize(MSE, β_hat)
        β_hat_out = Optim.minimizer(results)
    end
    return(β_hat_out)
end


function sim(β, N, x_fn, y_fn, estimation_fn, aux_estimator, β_grid, J_outer=500, J_inner=500)
    # do stuff in here
    K = length(β)
    storage = zeros(J_outer, K)
    for j in 1:J_outer
        X = x_fn((N,K))
        Y = y_fn(β,X)
        estimates = estimation_fn(Y, X, β_grid, y_fn, aux_estimator, J_inner)
        storage[j, :] .= estimates
    end
    return(storage)
end

end # end module
