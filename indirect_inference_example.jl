
include(joinpath("estimation", "indirect_inference.jl"))

using Statistics
using Distributions
using LinearAlgebra
using Main.IndirectInference


#######################
#######################
#######################
#######################

# indirect inference example 1
# y = β x^2
# note that the binding function is b = 2 .* β

ϵ = (σ -> Normal(0, σ)) # function to generate errors

σ0=1.0
y_true = ((β, X) -> X.^2 * β + rand(ϵ(σ0),size(X)[1])) # true model

x = (Size -> rand(Normal(0,1), Size)) # function to generate exogenous variables

function setup_X_matrix(X)
    # let's add a constant to the auxiliary OLS regression 
    a = ones(size(X)[1]) 
    # let's add in square and cubic terms.  Note, this makes the problem trivial, but
    # we don't have to add in the square and cubic terms (take them out and try it).
    return transpose([transpose(a); transpose(X); transpose(X.^2); transpose(X.^3)])
end

function est_aux(Y,X)  
    z = setup_X_matrix(X)
    return OLS(Y, z)
end


N=200
β0=3.0
K= length(β0)
X0 = x((N, K))
Y0 = y_true(β0, X0) # true model is y = x^2 β + ...
β_grid = [β0 .+ i for i in (-.5:.01:.5)]

ii1 = indirect_inference_grid(Y0, X0, β_grid, y_true, est_aux)

    

#######################
#######################
#######################
#######################

# indirect inference example 2
# y = β_1 x_1 ^ 2 + β_2 x_2 ^ 2

ϵ = (σ -> Normal(0, σ)) # function to generate errors

σ0=1.0
y_true = ((β, X) -> X.^2 * β + rand(ϵ(σ0),size(X)[1])) # true model

x = (Size -> rand(Normal(0,1), Size)) # function to generate exogenous variables

function setup_X_matrix(X)
    # let's add a constant to the auxiliary OLS regression 
    a = ones(size(X)[1]) 
    # let's add in square and cubic terms.  Note, this makes the problem trivial, but
    # we don't have to add in the square and cubic terms (take them out and try it).
    return transpose([transpose(a); transpose(X); transpose(X.^2); transpose(X.^3)])
end

function est_aux(Y,X)  
    z = setup_X_matrix(X)
    return OLS(Y, z)
end

N=200
β0=[3,2]
K = length(β0)
X0 = x((N, K))
Y0 = y_true(β0, X0) # true model is y = x^2 β + ...
β_grid = [β0 .+ i for i in (-.5:.01:.5)]

ii2 = indirect_inference_grid(Y0, X0, β_grid, y_true, est_aux)

ii2b = indirect_inference_optim(Y0, X0, y_true, est_aux)


#######################
#######################
#######################
#######################

# indirect inference example 3
# y = x^β
ϵ = (σ -> Normal(0, σ))

y_aux = ((β, σ, X) -> 1 + log.(X) * β + rand(ϵ(σ),size(X)[1])) #not needed but for illustation

σ0=1
y_true = ((β, X) -> X .^ β + rand(ϵ(σ0),size(X)[1]))

x = (Size -> rand(Uniform(0,1), Size))

function setup_X_matrix(X)
    a = ones(size(X)[1])
    return transpose([transpose(a); transpose(X); transpose(X.^2); transpose(X.^3)])
end

function est_aux(Y,X)
    a = log.(X)
    z = setup_X_matrix(a)
    return OLS(Y, z)
end


N=250
β0=3
K = length(β0)
X0 = x((N, K))
Y0 = y_true(β0, X0) # true model is y = x^2 β + ...
β_grid = [β0 .+ i for i in (-.5:.01:.5)]

ii3 = indirect_inference_grid(Y0, X0, β_grid, y_true, est_aux)








#####
# Now let's run a simulation experiment

ϵ = (σ -> Normal(0, σ)) # function to generate errors

σ0=1
y_true = ((β, X) -> X.^2 * β + rand(ϵ(σ0),size(X)[1])) # true model

x = (Size -> rand(Normal(0,1), Size)) # function to generate exogenous variables

function setup_X_matrix(X)
    a = ones(size(X)[1]) 
    return transpose([transpose(a); transpose(X); transpose(X.^2); transpose(X.^3)])
end

function est_aux(Y,X)  
    z = setup_X_matrix(X)
    return OLS(Y, z)
end


J_outer=100
J_inner=500
N=200
β0=3
β_grid = [β0 .+ i for i in (-.5:.02:.5)]

β_hats = sim(β0, N, x, y_true, indirect_inference_grid, est_aux, β_grid, J_outer, J_inner)


using Plots
histogram(β_hats, bins=8)









####
# Scratch and extra stuff below




# X can be N x k where N and k are arbitrary integers
# β must be k x 1
# we could probably make this a more general function from

x = (Size -> rand(Normal(0,1), Size))

function add_constant(X)
    a = ones(size(X)[1])
    return transpose([transpose(a); transpose(X)])
end


y = ((β, σ, X) -> X * β + rand(ϵ(σ),size(X)[1]))


X = x(10)
Y = y(3,1,X)

X = x((10,2))
Y = y([3,1],1,X)


# simulation example
J=10000
N=5000
σ=1
temp1 = sim(3, σ, N, J, x, y, OLS)
temp2 = sim([3,2], σ, N, J, x, y, OLS)







