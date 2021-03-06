#' @title Internal functions for formula and data handle
#' @description Utility functions useful for passing data from functions
#' inside others.
#'
#' @encoding UTF-8
#' @author Jaime Mosquera Gutiérrez, \email{jmosquerag@unal.edu.co}
#' @param model_frame a model frame build internally in some functions on
#'        \strong{EstimationTools} package
#' @export
#' @keywords internal
#' @rdname internalfunc
#'
#==============================================================================
# Formula conversion for 'survfit' using --------------------------------------
#==============================================================================
formula2Surv <- function(model_frame){
  vars <- names(model_frame)
  y_var <- paste0('Surv(', vars[1L], ', rep(1, nrow(model_frame)))')
  right_side <- if (length(vars) > 1){
    paste(vars[2L:end(vars)[1]], collapse = "+")
  } else {
    "1"
  }
  formula <- as.formula(paste0(y_var, "~", right_side))
  return(formula)
}
#==============================================================================
# Data preparation for TTT computation ----------------------------------------
#==============================================================================
#' @export
#' @keywords internal
#' @rdname internalfunc
#'
fo_and_data <- function(y, fo, model_frame, data, fo2Surv = TRUE){
  if ( !is.Surv(y) ){
    if ( fo2Surv ) fo <- formula2Surv(model_frame)
    if ( missing(data) | is.null(data) ) data <- model_frame
  } else {
    if ( missing(data) | is.null(data) ){
      vars <- names(model_frame)
      ySurv <- vars[1L]
      yname <- gsub("Surv\\((.*?),.*", "\\1", ySurv)
      statusname <- gsub(paste0("Surv\\(", yname, ",(.*?)\\)"), "\\1", ySurv)
      right_hand <- attr(stats::terms(fo), 'term.labels')

      if (length(right_hand) == 0){
        factorname <- NULL
        data <- data.frame(y[,1], y[,2])
      } else {
        factorname <- as.character(right_hand[1])
        other_column <- model_frame[,2]
        data <- data.frame(y[,1], y[,2], other_column)
      }
      colnames(data) <- c(yname, statusname, factorname)
    }
  }
  return(list(data = data, fo = fo))
}
