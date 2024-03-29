% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary.maxlogL.R
\encoding{UTF-8}
\name{summary.maxlogL}
\alias{summary.maxlogL}
\title{Summarize Maximum Likelihood Estimation}
\usage{
\method{summary}{maxlogL}(object, ...)
}
\arguments{
\item{object}{an object of \code{\link{maxlogL}} class which summary is desired.}

\item{...}{additional arguments affecting the summary produced.}
}
\value{
A list with information that summarize results of a \code{maxlogL} class object.
}
\description{
\ifelse{html}{\href{https://lifecycle.r-lib.org/articles/stages.html#maturing}{\figure{lifecycle-maturing.svg}{options: alt='[Maturing]'}}}{\strong{[Maturing]}}

Displays maximum likelihood estimates computed with \code{\link{maxlogL}} with
its standard errors, AIC and BIC.
This is a \code{summary} method for \code{\link{maxlogL}} object.
}
\details{
This \code{summary} method computes and displays AIC, BIC,
estimates and standard errors from a estimated model stored i a \code{maxlogL}
class object. It  also displays and computes Z-score and p values of significance
test of parameters.
}
\examples{
library(EstimationTools)

#--------------------------------------------------------------------------------
### First example: One known parameter

x <- rnorm(n = 10000, mean = 160, sd = 6)
theta_1 <- maxlogL(x = x, dist = 'dnorm', control = list(trace = 1),
                 link = list(over = "sd", fun = "log_link"),
                 fixed = list(mean = 160))
summary(theta_1)


#--------------------------------------------------------------------------------
# Second example: Binomial probability parameter estimation with variable
# creation

N <- rbinom(n = 100, size = 10, prob = 0.3)
phat <- maxlogL(x = N, dist = 'dbinom', fixed = list(size = 10),
                link = list(over = "prob", fun = "logit_link"))

## Standard error calculation method
print(phat$outputs$StdE_Method)

## 'summary' method
summary(phat)

#--------------------------------------------------------------------------------
# Third example: Binomial probability parameter estimation with no variable
# creation

N <- rbinom(n = 100, size = 10, prob = 0.3)
summary(maxlogL(x = N, dist = 'dbinom', fixed = list(size = 10),
                link = list(over = "prob", fun = "logit_link")))

#--------------------------------------------------------------------------------
# Fourth example: Estimation in a regression model with simulated normal data
n <- 1000
x <- runif(n = n, -5, 6)
y <- rnorm(n = n, mean = -2 + 3 * x, sd = exp(1 + 0.3* x))
norm_data <- data.frame(y = y, x = x)
formulas <- list(sd.fo = ~ x, mean.fo = ~ x)

norm_mod <- maxlogLreg(formulas, y_dist = y ~ dnorm, data = norm_data,
                       link = list(over = "sd", fun = "log_link"))

## 'summary' method
summary(norm_mod)


#--------------------------------------------------------------------------------

}
\seealso{
\code{\link{maxlogL}}, \code{\link{maxlogLreg}},
\code{\link{bootstrap_maxlogL}}
}
\author{
Jaime Mosquera Gutiérrez, \email{jmosquerag@unal.edu.co}
}
