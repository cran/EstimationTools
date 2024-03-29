% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/coef.maxlogL.R
\encoding{UTF-8}
\name{coef.maxlogL}
\alias{coef.maxlogL}
\alias{oefficients}
\alias{coefMany}
\title{Extract Model Coefficients in a \code{maxlogL} Fits}
\usage{
\method{coef}{maxlogL}(object, parameter = object$outputs$par_names, ...)

coefMany(object, parameter = NULL, ...)
}
\arguments{
\item{object}{an object of \code{maxlogL} class generated by \code{\link{maxlogLreg}}
function.}

\item{parameter}{a character which specifies the parameter is required. In
\code{coefMany} this argument can be an atomic vector
with two or more names of parameters.}

\item{...}{other arguments.}
}
\value{
A named vector with coefficients of the specified distribution parameter.
}
\description{
\ifelse{html}{\href{https://lifecycle.r-lib.org/articles/stages.html#maturing}{\figure{lifecycle-maturing.svg}{options: alt='[Maturing]'}}}{\strong{[Maturing]}}

\code{coef.maxlogL} is the specific method for the generic function \code{coef}
which extracts model coefficients from objects returned by \code{\link{maxlogLreg}}.
\code{coefficients} is an alias for \code{coef}.
}
\examples{
library(EstimationTools)

#--------------------------------------------------------------------------------
# Example 1: coefficients from a model using a simulated normal distribution
n <- 1000
x <- runif(n = n, -5, 6)
y <- rnorm(n = n, mean = -2 + 3 * x, sd = exp(1 + 0.3* x))
norm_data <- data.frame(y = y, x = x)

# It does not matter the order of distribution parameters
formulas <- list(sd.fo = ~ x, mean.fo = ~ x)

norm_mod <- maxlogLreg(formulas, y_dist = y ~ dnorm, data = norm_data,
                       link = list(over = "sd", fun = "log_link"))
coef(norm_mod)
coef(norm_mod, parameter = 'sd')
a <- coefMany(norm_mod, parameter = c('mean', 'sd'))
b <- coefMany(norm_mod)
identical(a, b)


#--------------------------------------------------------------------------------
# Example 2: Parameters in estimation with one fixed parameter
x <- rnorm(n = 10000, mean = 160, sd = 6)
theta_1 <- maxlogL(x = x, dist = 'dnorm', control = list(trace = 1),
                 link = list(over = "sd", fun = "log_link"),
                 fixed = list(mean = 160))
coef(theta_1)


#--------------------------------------------------------------------------------
}
\author{
Jaime Mosquera Gutiérrez, \email{jmosquerag@unal.edu.co}
}
