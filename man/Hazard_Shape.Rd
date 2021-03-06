% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Hazard_Shape.R
\encoding{UTF-8}
\name{Hazard_Shape}
\alias{Hazard_Shape}
\title{Summary of \code{HazardShape} objects}
\usage{
Hazard_Shape(object)
}
\arguments{
\item{object}{an object of class \code{HazardShape}, generated with
\code{\link{TTT_hazard_shape}}.}
}
\description{
This function displays the estimated hazard shape given a
data set.
}
\examples{
#--------------------------------------------------------------------------------
# Example 1: Increasing hazard and its corresponding TTT plot with simulated data
hweibull <- function(x, shape, scale){
  dweibull(x, shape, scale)/pweibull(x, shape, scale, lower.tail = FALSE)
  }
curve(hweibull(x, shape = 2.5, scale = pi), from = 0, to = 42,
               col = "red", ylab = "Hazard function", las = 1, lwd = 2)

y <- rweibull(n = 50, shape = 2.5, scale = pi)
my_initial_guess <- TTT_hazard_shape(formula = y ~ 1)
Hazard_Shape(my_initial_guess)


#--------------------------------------------------------------------------------
}
\author{
Jaime Mosquera Gutiérrez \email{jmosquerag@unal.edu.co}
}
