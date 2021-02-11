
# Newton's method
# looking for where the gradient = 0

# f'(θ_tilde) = (f(θ_1) - f(θ_2)) / (θ_1 - θ_2)

# f''(θ_n) = - f'(θ_n) / (θ_{n+1} - θ_n)

# => θ_{n+1} = θ_n - [f''(θ_n)]^{-1} f'(θ_n)

# When the gradient is not available, numerically approximate it
# f'(θ_n) ≈ (f(θ_n + δ) - f(θ_n)) / δ
# for some small δ
# similarly, may need to numerically approximate the hessian.

# see http://mth229.github.io/newton.html for julia code

# Note that there is also a more general optimization package, Optim.  
# I demonstrate the use of this below


###############################
###############################

# MCMC with Metropolis-Hastings:
# see https://github.com/TuringLang?language=julia for julia code





###############################
##############################
# Examples
#######

# Newton's method:
using ForwardDiff

function gp(g, δ=0.01)
    # numerically approximates g'
    # g'(x) ≈ (f(x + δ) - f(x)) / δ
    return( ( x -> (g(x .+ δ) - g(x)) ./ δ ) )
    # return(( x -> ForwardDiff.gradient(g, x)))
end

function gpp(g, δ=0.01)
    # numerically approximates g'
    return(gp(gp(g, δ), δ))
    # return((x -> ForwardDiff.hessian(g, x)))
end

function nm(f, fp, x)
    # f is the gradien (gp), fp is the derivative of the gradient (gpp) in our usage
    # this code from http://mth229.github.io/newton.html
    # Note this also only works for univariate functions as written
    # we could extend it, but why not just use the Optim package
    xnew, xold = x, Inf
    fn, fo = f(xnew), Inf

    tol = 1e-14
    ctr = 1

    # while (ctr < 100) && (abs(xnew - xold) > tol) && ( abs(fn - fo) > tol )
    while (ctr < 100) && (sum(abs.(xnew - xold)) > tol) && ( sum(abs.(fn - fo)) > tol )
        x = xnew - f(xnew)/fp(xnew) # update step
        # fp_inv = inv(fp(xnew))
        # temp = fp_inverse * fn
        # x = xnew - temp # update step
        xnew, xold = x, xnew
        fn, fo = f(xnew), fn
        ctr = ctr + 1
    end

    if ctr == 100
        error("Did not converge in 100 steps")
    else
    xnew, ctr
    end
end


# Example with nm():
# univariate function
fn1 = (x -> (x-2).^2 + 1 )

fn1p = gp(fn1)
fn1pp = gpp(fn1)

fn1(2)
fn1p(2)
fn1pp(2)

nm(fn1p, fn1pp, 1000)
# converges to x = 2 in 4 iterations


# Example with Optim
# multivariate function
fn2 = (x -> (x[1]-2).^2 * (x[2]-3).^2 + (x[2]-3) + 1 )

using Optim, NLSolversBase
results = optimize(fn2, [5.0,1.0], LBFGS())

x_min = Optim.minimizer(results) # argmin
fn2_min = Optim.minimum(results) # min

# see https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/ for details







