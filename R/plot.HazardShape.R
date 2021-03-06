#' Plot of \code{HazardShape} objects
#'
#' @encoding UTF-8
#' @author Jaime Mosquera Gutiérrez \email{jmosquerag@unal.edu.co}
#'
#' @description
#' Draws the empirical total time on test (TTT) plot and its non-parametric
#' (LOESS) estimated curve useful for identifying hazard shape.
#'
#' @aliases plot.HazardShape
#'
#' @param x an object of class \code{initValOW}, generated with
#'          \code{\link{TTT_hazard_shape}}.
#' @param xlab,ylab titles for x and y axes, as in \code{\link{plot}}.
#' @param curve_options a list with further arguments useful for customization
#'                      of non-parametric estimate plot.
#' @param xlim the x limits (x1, x2) of the plot.
#' @param ylim the y limits (x1, x2) of the plot.
#' @param col The colors for lines and points. Multiple colors can be specified. This is
#'            the usual color argument of \code{\link[graphics]{plot.default}}.
#' @param lty a vector of line types, see \code{\link{par}} for further information.
#' @param lwd a vector of line widths, see \code{\link{par}} for further information.
#' @param main a main title for the plot.
#' @param par_plot some graphical parameters which can be passed to the plot. See
#' \strong{Details} section for further information.
#' @param legend_options a list with fur further arguments useful for customization. See
#' \strong{Details} section for further information.
#'                      of the legend of the plot.
#' @param ... further arguments passed to empirical TTT plot.
#'
#' @details
#' This plot complements the use of \code{\link{TTT_hazard_shape}}. It is always
#' advisable to use this function in order to check the result of non-parametric estimate
#' of TTT plot. See the first example in \strong{Examples} section for an illustration.
#'
#' \code{par_plot} admits some parameters of \code{\link[graphics]{par}} function.
#' The following \emph{has preestablished values}:
#' \itemize{
#' \item{ \code{mai}:}{the margins can be manipulated with \code{mar}. The right margin
#' has a value equals to 7.2 using \code{mar} and all new values take it as reference value.}
#' \item{\code{xpd}:}{Is set as \code{TRUE}, and cannot be modified.}
#' }
#'
#' On the other hand, \code{legend_options} allows many of the parameters of
#' \code{\link[graphics]{legend}} function. The following \emph{has preestablished values}:
#' \itemize{
#' \item{\code{x, y}:}{legend is always located on the right side, outside the plot.
#' \item{\code{x} and \code{y}} coordinates cannot be manipulated, instead of this, it exists
#' the argument \code{pos}, which can take character or numeric values. In the first case,
#' it can be \code{"top", "center"} and \code{"bottom"}, in the later, it can be any value
#' of the y-coordinate, between 0 and 1.}
#' \item{\code{legend}:}{text of the legend cannot be edited.}
#' \item{\code{pch}:}{cannot be manipulated, it depends on \code{pch} parameter of the plot.}
#' \item{\code{col}:}{cannot be manipulated, it depends on \code{col} parameters of the plot
#' and the curve_options.}
#' #' \item{\code{lty}:}{cannot be manipulated, it depends on \code{lty} parameters of the plot
#' and the curve_options.}
#' #' \item{\code{lwd}:}{cannot be manipulated, it depends on \code{lwd} parameters of the plot
#' and the curve_options.}
#' #' \item{\code{pt.cex}:}{cannot be manipulated, it depends on \code{cex} parameter of
#' the plot.}
#' \item{\code{xpd}:}{It is set as \code{TRUE}, and cannot be modified.}
#' }
#'
#' If \code{legend_optinos = "NoLegend"}, no legend is generated.
#'
#' The possible arguments for \code{...} can be consulted in
#' \code{\link[graphics]{plot.default}}  and \code{\link{par}}.
#'
#' @examples
#' #--------------------------------------------------------------------------------
#' # Example 1: Increasing hazard and its corresponding TTT plot with simulated data
#' hweibull <- function(x, shape, scale){
#'   dweibull(x, shape, scale)/pweibull(x, shape, scale, lower.tail = FALSE)
#'   }
#' curve(hweibull(x, shape = 2.5, scale = pi), from = 0, to = 42,
#'                col = "red", ylab = "Hazard function", las = 1, lwd = 2)
#'
#' y <- rweibull(n = 50, shape = 2.5, scale = pi)
#' my_initial_guess <- TTT_hazard_shape(formula = y ~ 1)
#' plot(my_initial_guess, par_plot=list(mar=c(3.7,3.7,1,1.5),
#'                                      mgp=c(2.5,1,0)))
#'
#'
#' #--------------------------------------------------------------------------------
#'
#' @importFrom graphics par points
#' @importFrom autoimage reset.par
#' @export
plot.HazardShape <- function(x, xlab="i/n", ylab=expression(phi(i/n)),
                             xlim=c(0,1), ylim=c(0,1), col=1, lty=NULL, lwd=NA,
                             main="", curve_options=list(col=2, lwd=2, lty=1),
                             par_plot=list(mar=c(5.1,4.1,4.1,2.1)),
                             legend_options=NULL, ...){
  object <- x
  rm(x)

  autoimage::reset.par()
  if (is.null(par_plot$mar)){
    par_plot$mar=c(5.1,4.1,4.1,2.1)
  }

  if (is.character(legend_options)){
    if (legend_options == "NoLegend"){
      xpd <- FALSE
      mar <- c(par_plot$mar[1:3], par_plot$mar[4])
    }
  } else {
    xpd <- TRUE
    legend_options <- list(pos=1.04)
    mar <- c(par_plot$mar[1:3], par_plot$mar[4]+7.2)
  }
  par_plot$mar <- NULL
  do.call("par", c(list(mar=mar, xpd=xpd), par_plot))

  plot(object$TTTplot[,1], object$TTTplot[,2], xlab=xlab, ylab=ylab, xlim=xlim,
       ylim=ylim, main=main, col=col, lty=lty, lwd=lwd, ...)
  lines(c(0,1), c(0,1), lwd=2, lty=2)

  do.call("curve", c(list(expr=substitute(object$interpolation(x)), add=TRUE),
                     curve_options))

  plot_options <- substitute(...())
  if (is.null(plot_options$pch)) plot_options$pch <- 1
  if (is.null(plot_options$cex)) plot_options$cex <- 1

  if (!is.character(legend_options)){
    legend_text <- c("Empirical TTT", "Spline curve")
    if (is.null(legend_options$pos)){
      x <- "topright"; y <- NULL
    } else {
      if (is.numeric(legend_options$pos)){
        y <- legend_options$pos
        x <- 1.07
        legend_options <- within(legend_options, rm(pos))
      }
      if (is.character(legend_options$pos)){
        possible_pos <- c("top", "center", "bottom")
        if (!(legend_options$pos %in% possible_pos))
          stop((c("Select positions from the following list: \n \n",
                  "  --> ", paste0(possible_pos, collapse=", "))))
        if (legend_options$pos == "center") legend_options$pos <- ""

        x <- paste0(legend_options$pos, "right")
        legend_options <- within(legend_options, rm(pos))
        legend_arguments <- c("y", "inset", "legend", "xpd", "col", "lty", "lwd")
        match_legend <- match(legend_arguments, legend_options, nomatch=0)
        match_legend <- which(match_legend != 0)

        if (length(match_legend) > 0)
          stop(paste0("Argument(s)", legend_arguments[match_legend], "cannot be",
                      "manipulated. They have default unchangeable values."))
      }
    }

    do.call("legend", c(list(x=x, y=y, legend=legend_text,
                             pch=c(plot_options$pch,NA), inset=c(-0.41,0),
                             col=c(col, curve_options$col),
                             lty=c(lty,curve_options$lty),
                             pt.cex=plot_options$cex,
                             lwd=c(lwd,curve_options$lwd), xpd=TRUE),
                        legend_options))
  } else {
    if (legend_options != "NoLegend")
      stop("'NoLegend' option is the only character string valid")
  }
  autoimage::reset.par()
}
