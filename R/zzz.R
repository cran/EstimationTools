.onAttach <- function(libname, pkgname){
  initmessage1 <- "\n\n><<<<<<<<<<<<<<<<<<<<<<<<   EstimationTools Version "
  initmessage2 <- utils::packageDescription("EstimationTools")$Version
  initmessage3 <- "   >>>>>>>>>>>>>>>>>>>>>>>><
  Feel free to report bugs in https://github.com/Jaimemosg/EstimationTools/issues"
  packageStartupMessage(paste0(initmessage1, initmessage2, initmessage3))
}
