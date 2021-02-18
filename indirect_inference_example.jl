
include(joinpath("estimation", "indirect_inference.jl"))

using Statistics
using Distributions
using LinearAlgebra
using NLopt
using Main.IndirectInference


#######################
#######################
#######################
#######################

# indirect inference example 1 
# - A trivial example that is useful for illustration of the method
# True model: y = x^2 β + ϵ
# Auxiliary model: y = [1, x, x^2, x^3] * b + ν = Z b + ν
# note that the binding function is given by b = [0, 0, β, 0]'
# This implies that the optimal weighting matrix should take the value 1 at position 3,3 and 0 elsewhere
# Also, this implies that ∂b/∂β = [0, 0, 1, 0]'.  
# Note that this makes W* = (∂b/∂β)(∂b/∂β)' = [(∂b/∂β)(∂b/∂β)']^{-1} as desired.

ϵ = (σ -> Normal(0, σ)) # function to generate errors

function y_true(β, X)
    σ0=1.0
    return( X.^2 * β + rand(ϵ(σ0), size(X)[1]) ) # true model
end

x = (Size -> rand(Normal(0,1), Size)) # function to generate exogenous variables

function setup_X_matrix(X)
    a = ones(size(X)[1]) 
    return transpose([transpose(a); transpose(X); transpose(X.^2); transpose(X.^3)])
end

function est_aux(Y,X)  
    z = setup_X_matrix(X)
    return OLS(Y, z)
end

function est_aux2(Y,X)  
    z = setup_X_matrix(X)
    ∂b = [0, 0, 1, 0]
    return OLS(Y, z), ∂b
end


N=200
β0=3.0
K= length(β0)
X0 = x((N, K))
Y0 = y_true(β0, X0) # true model is y = x^2 β + ...
β_grid = [β0 .+ i for i in (-.5:.025:.5)]
W = kron([0, 0, 1, 0], [0, 0, 1, 0]')

ii1 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
ii1W = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, W=W)

ii1bs = iibootstrap(β=ii1[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, J_bs=9)
ii1Wbs = iibootstrap(β=ii1W[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, W=W, J_bs=9)
 
# for the nonlinear optimizer NLopt, β must be passed as an array even for univariate problems:
ii1b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, β_init=[β0]) # unconstrained first

NLoptOptions = NLopt_options(lb=[β0 - 0.5], ub=[β0 + 0.5]) # next, set the same bounds as the grid search for comparison
ii1c = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, β_init=[β0], NLoptOptions=NLoptOptions)
ii1cW = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, β_init=[β0], NLoptOptions=NLoptOptions, W=W)

ii1cbs = iibootstrap(β=ii1c[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=[β0], NLoptOptions=NLoptOptions, W=W, J_bs=9)
ii1cWbs = iibootstrap(β=ii1cW[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=[β0], NLoptOptions=NLoptOptions, W=W, J_bs=9)

# now use the gradient with a gradient-required algorithm
NLoptOptions = NLopt_options(alg=:LD_LBFGS)
ii1d = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux2, β_init=[β0], NLoptOptions=NLoptOptions, gradient=true)
ii1dW = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux2, β_init=[β0], NLoptOptions=NLoptOptions, W=W, gradient=true)

NLoptOptions = NLopt_options(alg=:LD_MMA)
ii1e = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux2, β_init=[β0], NLoptOptions=NLoptOptions, gradient=true)
ii1eW = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux2, β_init=[β0], NLoptOptions=NLoptOptions, W=W, gradient=true)



# using Plots
# histogram(ii1bs)

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
    # let's add in square and cubic terms.  Note, this makes the problem trivial (see example 1), 
    # but we don't have to add in the square and cubic terms (take them out and try it).
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

# grid search
ii2 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
# bootstrap
ii2bs = iibootstrap(β=ii2[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, J_bs=9)

# NL optimization
NLoptOptions = NLopt_options(lb=[2.5, 1.5], ub=[3.5,2.5], alg=:LN_NELDERMEAD)
ii2b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, NLoptOptions=NLoptOptions)

NLoptOptions = NLopt_options(lb=[2.5, 1.5], ub=[3.5,2.5], alg=:LN_SBPLX)
ii2c = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, NLoptOptions=NLoptOptions)

NLoptOptions = NLopt_options(lb=[2.5, 1.5], ub=[3.5,2.5], alg=:LD_SLSQP)
ii2d = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, NLoptOptions=NLoptOptions)
# forces stop

NLoptOptions = NLopt_options(lb=[2.5, 1.5], ub=[3.5,2.5], alg=:LD_MMA)
ii2e = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, NLoptOptions=NLoptOptions)
#forces stop

# bootstrap the NelderMead based estimates
NLoptOptions = NLopt_options(lb=[2.5, 1.5], ub=[3.5,2.5], alg=:LN_NELDERMEAD)
ii2bbs = iibootstrap(β=ii2b[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, NLoptOptions=NLoptOptions, J_bs=9)

#######################
#######################
#######################
#######################

# indirect inference example 3
# This is an example that does NOT work well.
# y = x^β
ϵ = (σ -> Normal(0, σ))

σ0=1
y_true = ((β, X) -> X .^ β + rand(ϵ(σ0),size(X)[1]))

x = (Size -> rand(Uniform(0,1), Size))

# Note that the McLaurin expansion about β=0 is 
# y ≈ 1 + β * ln(x)
# but this will not likely provide a good approximation for β0 different from 0
# the McLaurin expansion about β=3 is
# y ≈ x^3 + (β-3) * ln (x) * x^3

function est_auxβ(Y, X, β)
    a = ones(size(X)[1])
    if β == 0
        z = transpose([transpose(a); transpose(log.(X) .* (X.^β))])
    else
        z = transpose([transpose(a); transpose(X.^β); transpose(log.(X) .* (X.^β))])
    end
    return OLS(Y, z)
end

function est_aux0(Y,X)
    return est_auxβ(Y, X, 0)
end

function est_aux3(Y,X)
    return est_auxβ(Y, X, 3)
end

N=250
β0=3
K = length(β0)
X0 = x((N, K))
Y0 = y_true(β0, X0) # true model is y = x^2 β + ...
β_grid = [β0 .+ i for i in (-.5:.01:.5)]

ii3 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux0, search="grid", β_grid=β_grid)
ii3 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux3, search="grid", β_grid=β_grid)

ii3b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux0, β_init=[β0])
ii3b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux3, β_init=[β0])
NLoptOptions = NLopt_options(lb=[β0 - 0.5], ub=[β0 + 0.5])
ii3b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux0, β_init=[β0], NLoptOptions=NLoptOptions)
ii3b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux3, β_init=[β0], NLoptOptions=NLoptOptions)

# We never seem to get a good result.  Maybe we need a longer Taylor expansion...



#######################
#######################
#######################
#######################

# indirect inference example 4
# y = (x1 + β1)^β2 + β3 (x1 * x2 + β4)^2 + ϵ

ϵ = (σ -> Normal(0, σ))

function y_true(β, X) 
    σ0=1
    f1 = (X[:,1] .+ β[1]) .^ β[2]
    f2 = β[3] .* (X[:,1] .* X[:,2] .+ β[4]) .^ 2
    e = rand(ϵ(σ0),size(X)[1])
    out = f1 + f2 + e
    return(out)
end

x = (Size -> rand(Uniform(0,1), Size))

function setup_X_matrix(X)
    a = ones(size(X)[1])
    Xcross = X[:,1] .* X[:,2]
    return transpose([transpose(a); transpose(X); transpose(X.^2); transpose(X.^3); transpose(X.^4); transpose(Xcross); transpose(Xcross.^2)])
end

function est_aux_ols(Y, X)
    z = setup_X_matrix(X)
    return OLS(Y, z)
end

function est_aux(Y, X)
    # let's combine moments of Y with auxiliary model estimates
    N = size(Y0)[1]
    m1 = sum(Y) / N
    m2 = sum((Y .- m1).^2) / (N-1)
    m3 = sum((Y .- m1).^3) / (N-1)
    m4 = sum((Y .- m1).^4) / (N-1)
    b = est_aux_ols(Y, X)
    out = [m1, m2, m3, m4, b...]
    return out
end

N=400
β0=[3, 2, 1.5, 1]
K = length(β0)
X0 = x((N, 2))
Y0 = y_true(β0, X0) # true model is y = x^2 β + ...
β_grid = [β0 .+ i for i in (-.5:.01:.5)]

W = I(11)
W[1:4,1:4] = zeros(4,4)

ii4 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
ii4 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, W=W)
ii4bs = iibootstrap(β=ii4[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, W=W, J_bs=9)

NLoptOptions = NLopt_options(lb=β0 .- 0.5, ub=β0 .+ 0.5)
ii4b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, W=W, NLoptOptions=NLoptOptions)
ii4bbs = iibootstrap(β=ii4b[2], X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β0, J_bs=9, W=W, NLoptOptions=NLoptOptions)

# lets examine the deviations from β0
devs = ii4bbs .- repeat(ii4b[2]', 9)
round.(devs, digits=2)
# you can see that some might not be centered on zero.  This could be due to poor selection of
# the auxiliary model/moments








#######################
#######################
#######################
#######################

# indirect inference example 5
# classic example: estimate a true MA model with an AR auxiliary model


include(joinpath("estimation", "indirect_inference.jl"))

using Statistics
using Distributions
using LinearAlgebra
using NLopt
using Main.IndirectInference

ϵ = (σ -> Normal(0, σ))

function y_true5(β, N)
    max_l = 1
    Nb = 100 + max_l
    N2 = N + Nb # add in a burn in period
    e = rand(ϵ(1), N2)
    X = [lag(e, 0, 1)'; lag(e, 1, 1)']'
    Y = X * [1; β]
    Y = Y[(N2-N+1-max_l):(N2-max_l)]
    return(Y)
end

function lag(Y, l, max_l)
    Ny = length(Y)
    Nl = Ny - max_l
    Yl = Y[(1+max_l-l):(Ny-l)]
end

function setup_X_matrix(Y, max_l)
    X = ones(length(Y)-max_l)
    for l in 1:max_l
        X = [X'; lag(Y, l, max_l)']'
    end
    y = lag(Y, 0, max_l)
    return y, X
end

function est_aux(Y, P)
    y, z = setup_X_matrix(Y, P)
    return OLS(y, z)
end


N=1000
β0=0.5
Y0 = y_true5(β0, N) 
β_grid = [i for i in (-0.95:0.01:0.95)]
p = 10  # An MA(1) has an AR(∞) representation, but we will truncate at p=10

# We need to setup the problem so that it will fit into the ii format.  
# We can use the following wrapper functions:
est_aux_wrapper = ((Y, X) -> est_aux(Y, p))
y_true_wrapper = ((β, X) -> y_true5(β, length(X)))


ii5 = indirect_inference(Y0=Y0, X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="grid", β_grid=β_grid)
ii5bs = iibootstrap(β=ii5[2], X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="grid", β_grid=β_grid, J_bs=9)

NLoptOptions = NLopt_options(lb= -0.95, ub= 0.95)
ii5b = indirect_inference(Y0=Y0, X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="NL", β_init=[β0], NLoptOptions=NLoptOptions)
ii5bbs = iibootstrap(β=ii5b[2], X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="NL", β_init=[β0], J_bs=9, NLoptOptions=NLoptOptions)

# If I am remembering correctly, Eric Ghysels has a paper providing the 
# binding function for this case, so we could supply it and use a gradient based search
# with an optimal weighting matrix, but the gradient free approach seems to work pretty well.






#######################
#######################
#######################
#######################

# Example 6 - Vector y
# let's piggy back off of example 5 and see if we can do a VMA/VAR example

include(joinpath("estimation", "indirect_inference.jl"))

using Statistics
using Distributions
using LinearAlgebra
using NLopt
using Main.IndirectInference

ϵ = (σ -> Normal(0, σ))

function y_true6(vecπ, N)
    # Note that when Y is a vector process, we need to carefully keep track of matrix dimensions
    # π = [0.5 0.25; 0.2 0.4]'
    # vecπ = vec(π')
    max_l = 1
    ρ = 0.3 # make the errors correlated across series
    n = 2
    Nb = 100 + max_l
    N2 = N + Nb # add in a burn in period
    Ω = [1 ρ; ρ 1]
    s = cholesky(Ω).U
    e = rand(ϵ(1), (N2, n))
    e = e * s # T x n
    X = lag(e, 1, 1)'  # (nq) x T
    ep = lag(e, 0, 1)' # n x T
    # Y = π' * X .+ ep  # [n x (nq)] * [(nq) x T] + [n x T] = n x T
    # Y = Y[:, (N2-N+1-max_l):(N2-max_l)] # remove burn in
    # y = vec(Y)  # nT x 1
    vecY = (kron(X', I(n))) * vecπ + vec(ep)  # nT x 1 = [nT x n(nq)] * [n(nq) x 1] + nT x 1
    vecY = vecY[(n*(N2-N-max_l)+1):(n*(N2-max_l))]
    # y - vecY  # sanity check
    return(vecY)
end

function unvecY(y, N)
    n = Int(length(y) / N)
    out = y[(1:n:(N*n))]
    for i in 2:n
        out = [out'; y[(i:n:(N*n))]']'
    end
    return(out)  # T x n
end

function lag(Y, l, max_l)
    Ny = size(Y)[1]
    Nl = Ny - max_l
    Yl = Y[(1+max_l-l):(Ny-l), :]
end

function setup_X_matrix(Y, max_l)
    X = ones(size(Y)[1]-max_l)
    for l in 1:max_l
        X = [X'; lag(Y, l, max_l)']'
    end
    y = lag(Y, 0, max_l)
    return y, X
end

function est_aux(vecY, P, N)
    Y = unvecY(vecY, N)
    n = size(Y)[2]
    y, z = setup_X_matrix(Y, P) # T x n and T x (nP + 1)
    vecy = vec(y')
    Z = kron(z, I(n))
    return OLS(vecy, Z)
end


# This is a complex problem, so let's use a large N and p
N=5000
π = [0.5 0.25; 0.2 0.4]'
vecπ = vec(π') # vecπ is our β here
Y0 = y_true6(vecπ, N) 
β_grid = [vecπ .+ i for i in (-0.5:0.01:0.45)]
p = 20

# Similar to example 5, we create wrapper functions that ignore X but pass p and N
est_aux_wrapper = ((Y, X) -> est_aux(Y, p, N))
y_true_wrapper = ((β, X) -> y_true6(β, N))

# Note that an extensive grid search would be really computationally expensive
ii6 = indirect_inference(Y0=Y0, X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="grid", β_grid=β_grid)
# ii6bs = iibootstrap(β=ii5[2], X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="grid", β_grid=β_grid, J_bs=9)

NLoptOptions = NLopt_options(lb=fill(-0.95,4), ub=fill(0.95,4))
ii6b = indirect_inference(Y0=Y0, X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="NL", β_init=vecπ, NLoptOptions=NLoptOptions)
# ii6bbs = iibootstrap(β=ii5b[2], X0=Y0, true_model=y_true_wrapper, aux_estimation=est_aux_wrapper, search="NL", β_init=vecπ, J_bs=9, NLoptOptions=NLoptOptions)

ii6[2]
ii6b[2]
vecπ
# looks good
















#####
#####
#####
#####
#####
#####
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







