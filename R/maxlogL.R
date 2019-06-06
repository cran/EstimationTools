#' @title Maximum Likelihood Estimation
#'
#' @description
#' Function to compute maximum likelihood estimators (MLE)
#' of any distribution implemented in \code{R}.
#'
#' @author Jaime Mosquera Gutiérrez, \email{jmosquerag@unal.edu.co}
#'
#' @param x a vector with data to be fitted. This argument must be a matrix with hierarchical distributions.
#' @param dist a length-one character vector with the name of density/mass function of interest. The default value
#'             is "dnorm", to compute maximum likelihood estimators of normal distribution.
#' @param fixed a list with fixed/known parameters of distribution of interest. Fixed parameters
#'              must be passed with its name.
#' @param link a list with names of parameters to be linked, and names of the link object. For names of parameters, please
#'            visit documentation of density function. There are three link functions available: \code{\link{log_link}},
#'            \code{\link{logit_link}} and \code{\link{NegInv_link}}.
#' @param start a numeric vector with initial values for the parameters to be estimated.
#' @param lower a numeric vector with lower bounds, with the same lenght of argument `start` (for box-constrained optimization).
#' @param upper a numeric vector with upper bounds, with the same lenght of argument `start` (for box-constrained optimization).
#' @param optimizer a lenght-one character vector with the name of optimization routine. \code{\link{nlminb}}, \code{\link{optim}}
#'                  and \code{\link{DEoptim}} are available; \code{\link{nlminb}} is the default.
#'                  routine.
#' @param control control parameters of the optimization routine. Please, visit documentation of selected
#'                optimizer for further information.
#' @param ... Further arguments to be supplied to the optimizer.
#'
#' @return A list with class \code{"maxlogL"} containing the following
#'  lists:
#' \item{fit}{A list with output information about estimation and method used.}
#' \item{inputs}{A list with all input arguments.}
#' \item{outputs}{A list with number of parameters, sample size and standard error method.}
#'
#' @details \code{maxlogL} calculates computationally the likelihood function corresponding to
#' the distribution specified in argument \code{dist} and maximizes it through
#' \code{\link{optim}}, \code{\link{nlminb}} or \code{\link{DEoptim}}. \code{maxlogL}
#' generates an S3 object of class \code{maxlogL}.
#'
#' @importFrom stats nlminb optim pnorm
#' @importFrom DEoptim DEoptim
#' @importFrom BBmisc is.error
#' @importFrom numDeriv hessian
#' @export
#'
#' @examples
#' # Estimation with one fixed parameter
#' x <- rnorm(n = 10000, mean = 160, sd = 6)
#' theta_1 <- maxlogL(x = x, dist = 'dnorm', control = list(trace = 1),
#'                  link = list(over = "sd", fun = "log_link"),
#'                  fixed = list(mean = 160))
#' summary(theta_1)
#'
#'# Both parameters of normal distribution mapped with logarithmic function
#' theta_2 <- maxlogL( x = x, dist = "dnorm",
#'                     link = list(over = c("mean","sd"),
#'                                 fun = c("log_link","log_link")) )
#' summary(theta_2)
#'
#' # Parameter estimation in ZIP distribution
#' library(gamlss.dist)
#' z <- rZIP(n=10000, mu=6, sigma=0.08)
#' theta_3  <- maxlogL( x = z, dist='dZIP', start = c(0, 0), lower = c(-Inf, -Inf),
#'                      upper = c(Inf, Inf), optimizer = 'optim',
#'                      link = list(over=c("mu", "sigma"),
#'                      fun = c("log_link", "logit_link")) )
#' summary(theta_3)
#'
#' @references
#' \insertRef{Nelder1965}{EstimationTools}
#'
#' \insertRef{Fox1978}{EstimationTools}
#'
#' \insertRef{Nash1979}{EstimationTools}
#'
#' \insertRef{Dennis1981}{EstimationTools}
#'
#' @importFrom Rdpack reprompt
#'
#' @seealso \code{\link{summary.maxlogL}}, \code{\link{optim}}, \code{\link{nlminb}},
#'          \code{\link{DEoptim}}, \code{\link{DEoptim.control}}
#==============================================================================
# Maximization routine --------------------------------------------------------
#==============================================================================
maxlogL <- function(x, dist = 'dnorm', fixed = NULL, link = NULL,
                    start = NULL, lower = NULL, upper = NULL,
                    optimizer = 'nlminb', control = NULL, ...){

  # List of arguments of density function
  arguments <- as.list(args(dist))

  # Common errors
  if (class(control) != 'list'){
    if (!is.null(control)){
      stop("control argument must be a list \n \n")
    }
  }

  if ( is.null(dist) ) stop("Distribution not specified \n \n")

  solvers <- c('nlminb', 'optim', 'DEoptim')
  if ( !optimizer %in% solvers ){
    stop(c("Select optimizers from the following list: \n \n",
           "  --> ",paste0(solvers, collapse=", ")))
  }
  if ( !is.null(link) ){
    if (length(match(link$over, names(arguments)) ) == 0)
      stop(paste0("Name(s) of linked parameter(s) do not agree with ",
                  "arguments of ", dist, ". \n Please, change name(s) ",
                  "specified in the entry 'over' of 'link' argument in \n",
                  " function maxlogL.\n"))
    if ( is.null(link$over) & !is.null(link$fun) ){
      warn <- paste0("You do not specify parameters to map, ",
                     "however, you specify a link function \n ",
                     "(the entry 'over' in 'link' argument is NULL ",
                     "but the entry 'fun' is not NULL).\n")
      warning(warn)
    }
    if ( !is.null(link$over) & is.null(link$fun) )
      stop(paste0("You specify parameters to map, ",
                  "however, you do not specify a link function \n",
                  "(the entry 'fun' in 'link' argument is NULL ",
                  "but the entry 'over' is not NULL).\n "))
  }
  if ( !is.null(fixed) ){
    if( length(match(names(fixed),names(arguments))) == 0 )
      stop(paste0("Name(s) of fixed (known) parameter(s) do not agree with ",
                  "arguments of ", dist, ". \n Please, change names ",
                  "specified in argument 'fixed' in function ",
                  "maxlogL", "\n"))
  }
  if ( length(x) == 0 | is.null(x) ){
    stop(paste0("Vector of data is needed to perform maximum likelihood ",
                "estimation. \n Please, specify the vector x in maxlogL ",
                "function. \n"))
  }

  # Exclusion of fixed parameters from objective variables
  pos.deletion <- match(names(fixed), names(arguments))
  if ( length(pos.deletion) > 0 ) arguments <- arguments[-pos.deletion]

  # Parameters counting
  nnum <- sapply(1:length(arguments),
                 FUN = function(x) is.numeric(arguments[[x]]))
  nsym <- sapply(1:length(arguments),
                 FUN = function(x) is.symbol(arguments[[x]]))

  # x is a symbol, must be substracted
  npar <- length(nnum[nnum == TRUE]) + length(nsym[nsym == TRUE]) - 1

  #  Negative of log-Likelihood function
  ll <- minus_ll(x = x, dist, dist_args = arguments, over = link$over,
                 link = link$fun, npar = npar, fixed = fixed)

  #  Default feasible region
  if ( is.null(lower) ) lower <- rep(x = -Inf, times = npar)
  if ( is.null(upper) ) upper <- rep(x = Inf, times = npar)
  if ( is.null(start) ) start <- rep(x = 0, times = npar)

  # Link application over initial values
  if ( !is.null(lower) & !is.null(upper) & !is.null(start)){
    if ( !is.null(link$over) & !is.null(link$fun) ){
      linked_params <- link_apply(over = link$over, dist_args = arguments,
                                  npar = npar)
      link_start <- vector(mode = "list", length = length(linked_params))
      link_init <- paste0(link$fun, "()")
      link_start <- lapply( 1:length(linked_params), FUN =
                               function(x) eval(parse(text = link_init[x])) )
      for (i in 1:length(linked_params)){
        g_start <- paste0("link_start[[", i, "]]$g_inv")
        g_start <- eval(parse(text = g_start))
        start[linked_params[i]] <- do.call( what = "g_start",
                                              args = list(x = start[linked_params[i]]) )
      }
    }
  }

  # Optimizers
  if ( optimizer == 'nlminb' ) {
    nlminbcontrol <- control
    fit <- nlminb(start = start, objective = ll,
                  lower = lower, upper = upper, control = nlminbcontrol, ...)
    fit$objective <- -fit$objective
  }

  if ( optimizer == 'optim' ) {
    optimcontrol <- control
    if (npar<2) fit <- optim(par = start, fn = ll, lower = lower, upper=upper)
    fit <- optim(par = start,fn = ll, control = optimcontrol, ...)
    fit$objective <- -fit$value
  }

  if ( optimizer == 'DEoptim' ) {
    if (is.null(lower) | is.null(upper)) stop("'lower' and 'upper'
                                               limits must be defined
                                               for 'DEoptim' optimizer", "\n\n")
    DEoptimcontrol <- c(trace = FALSE, control)
    trace_arg <- which(names(DEoptimcontrol) == "trace")
    if (length(trace_arg) > 1){
      if (length(trace_arg) == 2){
        DEoptimcontrol$trace <- NULL
      } else {
        warn <-"Argument 'trace' in 'DEoptim.control' has multiple definitions \n"
        warning(warn)
      }
    }
    fit <- DEoptim(fn = ll, lower = lower, upper = upper,
                   control = DEoptimcontrol, ...)
    fit$par <- fit$optim$bestmem
    fit$objective <- -fit$optim$bestval
  }

  # Revert link mapping
  if ( !is.null(link$over) & !is.null(link$fun) ){
    linked_params <- link_apply(over = link$over, dist_args = arguments,
                                npar = npar)
    link_revert <- vector(mode = "list", length = length(linked_params))
    link_rev <- paste0(link$fun, "()")
    link_revert <- lapply( 1:length(linked_params), FUN =
                             function(x) eval(parse(text = link_rev[x])) )
    for (i in 1:length(linked_params)){
      g_revert <- paste0("link_revert[[", i, "]]$g_inv")
      g_revert <- eval(parse(text = g_revert))
      fit$par[linked_params[i]] <- do.call( what = "g_revert",
                                   args = list(x = fit$par[linked_params[i]]) )
    }
  }

  ll.noLink <- minus_ll(x = x, dist, dist_args = arguments, over = NULL,
                        link = NULL, npar = npar, fixed = fixed)
  fit$hessian <- try(optim(par = fit$par, fn = ll.noLink, method = 'L-BFGS-B',
                           lower = lower, upper = upper,
                           hessian = TRUE)$hessian, silent = TRUE)
  # fit$hessian <- try(optimHess(par = fit$par, fn = ll.noLink, method = 'L-BFGS-B',
  #                              lower = lower, upper = upper), silent = TRUE)

  StdE_Method <- "Hessian from optim"
  if ( (any(is.na(fit$hessian)) | is.error(fit$hessian)) |
       any(is.character(fit$hessian)) ){
    fit$hessian <- try(numDeriv::hessian(ll.noLink, fit$par), silent = TRUE)
    StdE_Method <- "numDeriv::hessian"
  }
  if ( (any(is.na(fit$hessian)) | is.error(fit$hessian)) |
       any(is.character(fit$hessian)) ) fit$hessian <- NA

  inputs <- list(dist = dist, fixed = fixed,
                 link = link, optimizer = optimizer,
                 start = start, lower = lower, upper = upper,
                 x = x)
  outputs <- list(npar = npar - length(fixed), n = length(x),
                 StdE_Method = StdE_Method)
  result <- list(fit = fit, inputs = inputs, outputs = outputs)
  class(result) <- "maxlogL"
  return(result)
}
#==============================================================================
# Link application ------------------------------------------------------------
#==============================================================================
link_apply <- function(over, dist_args, npar){
  if (is.null(over)){
    return(linked_params=NULL)
  } else {
    if (length(over) > npar) stop("Number of mapped parameters is
                                  greater than the number of
                                  parameters of the distribution")
    numeric_list <- vector(mode = "list", length = npar + 1)
    names_numeric <- rep("", times = npar + 1)
    j <- 1
    for (i in 1:length(dist_args)){
      if (is.numeric(dist_args[[i]]) | is.symbol(dist_args[[i]])){
        numeric_list[[j]]<- dist_args[[i]]
        names_numeric[j] <- names(dist_args[i])
        j <- j + 1
      }
    }
    numeric_list[which(names_numeric=="x")] <- NULL
    names_numeric <- names_numeric[-which(names_numeric=="x")]
    names(numeric_list) <- names_numeric

    args_names <- names(dist_args)
    mapped_param <- match(over, args_names)

    linked_args <- vector(mode="list", length=length(over))
    names_linked <- rep("", times=length(over))
    for (i in 1:length(mapped_param)){
      linked_args[i] <- dist_args[mapped_param[i]]
      names_linked[i] <- names(dist_args[mapped_param[i]])
    }
    names(linked_args) <- names_linked

    linked_params <- match(names_linked, names_numeric)
    return(linked_params)
  }
}
#==============================================================================
# log-likelihood function computation -----------------------------------------
#==============================================================================
minus_ll <- function(x, dist, dist_args, over, link, npar, fixed){
  f <- function(param){
    if( !is.null(link) & !is.null(over) ){
      linked_params <- link_apply(over = over, dist_args = dist_args,
                                  npar = npar)
      if ( !is.null(linked_params) ){
        link_eval <- vector( mode = "list", length = length(linked_params) )
        link <- paste0(link, "()")
        link_eval <- lapply( 1:length(linked_params),
                             FUN=function(x) eval(parse(text = link[x])) )
        for (i in 1:length(linked_params)){
          g_inv <- paste0("link_eval[[", i, "]]$g_inv")
          g_inv <- eval(parse(text=g_inv))
          param[[linked_params[i]]] <- do.call( what = "g_inv", args = list(
            x=param[[linked_params[i]]]) )
        }
      }
    }
    logf <- do.call( what = dist, args = c(list(x = x), param,
                                           log=TRUE, fixed) )
    return(-sum(logf))
  }
  return(f)
}