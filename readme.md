This repository provides a julia module for estimation via Indirect Inference and Simulated Method of Moments.  The module can be found in src/SMMIndirectInference.jl, a short tutorial can be found in indirect_inference_tutorial.jl, and some additional examples can be found in indirect_inference_examples.jl.

NOTE:
If you want to use structs, etc. please use a wrapper function.  See the provided examples for details.

TO DO:
- Allow weighting matrix to depend upon b or β
- Add in MCMC for bayesian estimation
- Standard errors for the case when the binding function is known
- Code cleanup

###########################################
###########################################
Implements Indirect Inference and SMM

Indirect Inference Estimation
Usage:
- indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, kwargs)

Required key word arguments:
-   Y0 is the outcome variable from the data
-   X0 is the list of covariates from the data
-   true_model is a generic function Y=f(β, X) describing the structural model
-   aux_estimation is a generic function b_hat = g(Y, X) describing the auxiliary model and estimation routine

Semi-Optional key word arguments:
-   search ∈ {"NL", "grid"}, Default is "NL"
-   β_init: initial value for the "NL" estimation routine (required for search="NL")
-   β_grid: grid of βs for the "grid" search estimation routine (required for search="grid")
-   J: number of times to simulate from the true model
-   NLoptOptions: Options passed to the nonlinear optimizer NLopt (see NLopt_Options() below for details)
-   W: the weighting matrix (I may change this later to get it from aux_estimation)  Note that the
    -       optimal weighting matrix depends upon the binding function in general.  When the binding
    -       function is not known, results are not expected to be efficient.
-   gradient: Must be =true if using search="NL" and a gradient based search algorithm.

Output:
-   array of the structural parameter estimates

Examples:
    ii2 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
    ii2b = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="NL", β_init=β_init, NLoptOptions=NLoptOptions)
    See indirect_inference_examples.jl for complete examples

###########################################
###########################################

Indirect Inference with Inference via the parameteric bootstrap
Usage:
    iibootstrap(;β, X0, true_model, aux_estimation, J_bs=9, kwargs...)

Note:
    iibootstrap() should be called with the same arguments as indirect_inference()

Arguments:
    J_bs: the number of bootstrap samples
    β: the parameter used to simulate the model,
        usually the null hypothesis value β_0 if testing against a null
        or the estimates β_hat
    all other arguments are the same as indirect_inference()

Output:
    Array of size J_bs x K for the J_bs estimates

Examples:
    ii2 = indirect_inference(Y0=Y0, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid)
    ii2bs = iibootstrap(β=ii2, X0=X0, true_model=y_true, aux_estimation=est_aux, search="grid", β_grid=β_grid, J_bs=99)

Note:
-   Not knowing the binding function that links the structural and auxiliary model parameters makes inference complicated.
-   Hence, this implements a parameteric bootstrap.
-   The trade-off is that this is computationally intensive.

###########################################
###########################################

Setting options for the NLopt Optimizer
Usage:
-   NLoptions = NLopt_options(lb=Nothing, ub=Nothing, cons_ineq=Nothing, alg=:LN_NELDERMEAD, xtol=1e-4)

Arguments:
-   lb: array of lower bounds
-   ub: array of upper bounds
-   cons_ineq: array of generic functions representing inequality constraints
-   alg: optimization algorithm (NelderMead is the default)
-   xtol: x tolerance

Output:
-   a struct of options to pass to the NLopt wrapper function

More details regarding NLopt can be found here:
-   https://github.com/JuliaOpt/NLopt.jl
-   https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/



