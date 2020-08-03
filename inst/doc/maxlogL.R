### R code from vignette source 'maxlogL.Rnw'

###################################################
### code chunk number 1: preliminaries
###################################################
options(prompt = "R> ", continue = "+  ", width = 70, useFancyQuotes = FALSE)
library(EstimationTools)


###################################################
### code chunk number 2: download (eval = FALSE)
###################################################
## install.packages("EstimationTools")
## library(EstimationTools)


###################################################
### code chunk number 3: func (eval = FALSE)
###################################################
## maxlogL(x, dist, optimizer, lower = NULL, upper = NULL)


###################################################
### code chunk number 4: first_example
###################################################
set.seed(1000)
z <- rnorm(n = 1000, mean = 10, sd = 1)
fit1 <- maxlogL(x = z, dist = 'dnorm', start=c(2, 3),
                lower=c(-15, 0), upper=c(15, 10))
summary(fit1)


###################################################
### code chunk number 5: link_example
###################################################
fit2 <- maxlogL(x = z, dist = 'dnorm',
                link = list(over = "sd", fun = "log_link"))
summary(fit2)


###################################################
### code chunk number 6: link2_example
###################################################
fit3 <- maxlogL(x = z, dist = 'dnorm',
                link = list(over = c("mean", "sd"),
                            fun = c("log_link", "log_link")))
summary(fit3)


###################################################
### code chunk number 7: binomial
###################################################
set.seed(100)
N <- rbinom(n = 100, size = 10, prob = 0.3)
phat <- maxlogL(x = N, dist = 'dbinom', fixed = list(size = 10),
                link = list(over = "prob", fun = "logit_link"))
summary(phat)


###################################################
### code chunk number 8: EEBdensity (eval = FALSE)
###################################################
## dEEB <- function(x, mu = 0.5, sigma = 1, nu = 1.5, size = 10,
##                  log = FALSE){
##   if (any(x<0))
##     stop(paste("x must greater than zero", "\n", ""))
##   if (any(mu < 0) | any(mu > 1))
##     stop(paste("mu must be between 0 and 1", "\n", ""))
##   if (any(sigma<=0 ))
##     stop(paste("sigma must be greater than zero", "\n", ""))
##   if (any(nu<=0 ))
##     stop(paste("nu must be greater than zero", "\n", ""))
## 
##   loglik <- log(sigma * nu * size *  mu) - sigma * x +
##             (nu - 1) * log(1 - exp(-sigma * x)) -
##             log(1 - (1 - mu)^size) +
##             (size - 1) * log(1 - mu * (1 - exp(-sigma * x))^nu)
##   if (log == FALSE)
##     density <- exp(loglik) else density <- loglik
##   return(density)
## }


###################################################
### code chunk number 9: norm_a (eval = FALSE)
###################################################
## param.val=list(mean=5, sd=2)
## distrib <- 'dnorm'
## 
## Parameters1 <- read.table(file=paste0("params_", distrib, "_", "nlminb",
##                                       ".txt"),
##                           header=TRUE)
## Parameters2 <- read.table(file=paste0("params_", distrib, "_", 'optim',
##                                       ".txt"),
##                           header=TRUE)
## Parameters <- rbind(Parameters1, Parameters2)
## 
## par(family="serif", cex.lab=2.5, cex.axis=2.5,
##     mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5), las=0)
## mu <- aggregate(mean ~ n + method, data=Parameters, FUN=mean)
## plot(mu[mu$method=='nlminb',]$n, mu[mu$method=='nlminb',]$mean,
##      xlab="n", ylab=expression(hat(mu)), type="l", lwd=2)
## lines(mu[mu$method=='optim',]$n, mu[mu$method=='optim',]$mean,
##       lwd=1, col="dodgerblue3")
## legend("bottomright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$mean, col=2, lwd=4)


###################################################
### code chunk number 10: norm_b (eval = FALSE)
###################################################
## par(family="serif", cex.lab=2.5, cex.axis=2.5,
##     mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5), las=0)
## sigma <- aggregate(sd ~ n + method, data=Parameters, FUN=mean)
## plot(sigma[sigma$method=='nlminb',]$n, sigma[sigma$method=='nlminb',]$sd,
##      xlab="n", ylab=expression(hat(sigma)), type="l", lwd=2)
## lines(sigma[sigma$method=='optim',]$n, sigma[sigma$method=='optim',]$sd,
##       type="l", lwd=1, col="dodgerblue3")
## legend("bottomright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$sd, col=2, lwd=4)


###################################################
### code chunk number 11: EEBa (eval = FALSE)
###################################################
## param.val=list(mu = 0.5, sigma = 1, nu = 1.5, size = 10)
## distrib <- 'dEEB'
## 
## Parameters1 <- read.table(file=paste0("params_", distrib, "_", "nlminb",
##                                       ".txt"),
##                           header=TRUE)
## Parameters2 <- read.table(file=paste0("params_", distrib, "_", 'optim',
##                                       ".txt"),
##                           header=TRUE)
## Parameters <- rbind(Parameters1, Parameters2)
## 
## par(family="serif", cex.lab=2.5, cex.axis=2.5,
##     mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5))
## mu <- aggregate(mu ~ n + method, data=Parameters, FUN=mean)
## plot(mu[mu$method=='nlminb',]$n, mu[mu$method=='nlminb',]$mu,
##      xlab="n", ylab=expression(hat(mu)), type="l", lwd=2)
## lines(mu[mu$method=='optim',]$n, mu[mu$method=='optim',]$mu,
##       lwd=1, col="dodgerblue3")
## legend("topright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$mu, col=2, lwd=3)


###################################################
### code chunk number 12: EEBb (eval = FALSE)
###################################################
## par(family="serif", cex.lab=2.5, cex.axis=2.5,
##     mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5))
## sigma <- aggregate(sigma ~ n + method, data=Parameters, FUN=mean)
## plot(sigma[sigma$method=='nlminb',]$n[3:nrow(sigma)],
##      sigma[sigma$method=='nlminb',]$sigma[3:nrow(sigma)],
##      xlab="n", ylab=expression(hat(sigma)), type="l", lwd=2)
## lines(sigma[sigma$method=='optim',]$n, sigma[sigma$method=='optim',]$sigma,
##       lwd=1, col="dodgerblue3")
## legend("topright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$sigma, col=2, lwd=3)


###################################################
### code chunk number 13: EEBc (eval = FALSE)
###################################################
## par(family="serif", cex.lab=2.5, cex.axis=2.5,
##     mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5))
## nu <- aggregate(nu ~ n + method, data=Parameters, FUN=mean)
## plot(subset(nu, method=="nlminb")$n[4:nrow(nu)], ylim=c(1.48,2),
##      nu[nu$method=='nlminb',]$nu[4:nrow(nu)], xlim=c(1,1000),
##      xlab="n", ylab=expression(hat(nu)), type="l", lwd=2)
## lines(nu[nu$method=='optim',]$n[4:nrow(nu)],
##       nu[nu$method=='optim',]$nu[4:nrow(nu)],
##       type="l", lwd=1, col="dodgerblue3")
## legend("topright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$nu, col=2, lwd=3)


###################################################
### code chunk number 14: ZIPa (eval = FALSE)
###################################################
## param.val=list(mu=6, sigma=0.08)
## distrib <- 'dZIP'
## 
## Parameters1 <- read.table(file=paste0("params_", distrib, "_", "nlminb",
##                                       ".txt"),
##                           header=TRUE)
## Parameters2 <- read.table(file=paste0("params_", distrib, "_", 'optim',
##                                       ".txt"),
##                           header=TRUE)
## Parameters <- rbind(Parameters1, Parameters2)
## 
## par(family="serif", cex.lab=2.5, cex.axis=2.5,
##     mar=c(7,6.5,4,2)+0.1, mai=c(1.5,1.5,0.5,0.5), las=0)
## mu <- aggregate(mu ~ n + method, data=Parameters, FUN=mean)
## plot(mu[mu$method=='nlminb',]$n, mu[mu$method=='nlminb',]$mu,
##      xlab="", ylab="", type="l", lwd=2)
## lines(mu[mu$method=='optim',]$n, mu[mu$method=='optim',]$mu,
##       lwd=1, col="dodgerblue3")
## legend("bottomright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$mu, col=2, lwd=4)
## title(ylab=expression(hat(lambda)), mgp=c(5,1,0), cex.lab=2.5)
## title(xlab=expression(n), mgp=c(3.5,1,0), cex.lab=2.5)
## # mtext("(a)", side=1, line=6.5, cex=3)


###################################################
### code chunk number 15: ZIPb (eval = FALSE)
###################################################
## par(family="serif", cex.lab=2, cex.axis=2.5,
##     mar=c(7,6.5,4,2)+0.1, mai=c(1.5,1.5,0.5,0.5), las=0)
## sigma <- aggregate(sigma ~ n + method, data=Parameters, FUN=mean)
## plot(sigma[sigma$method=='nlminb',]$n, sigma[sigma$method=='nlminb',]$sigma, xlab="", ylab="", type="l", lwd=2)
## lines(sigma[sigma$method=='optim',]$n, sigma[sigma$method=='optim',]$sigma,
##       type="l", lwd=1, col="dodgerblue3")
## legend("bottomright", legend=c('nlminb', 'optim (BFGS)'),
##        col=c(1,"dodgerblue3"), lwd=c(2,1), cex=2.5)
## abline(h=param.val$sigma, col=2, lwd=4)
## title(ylab=expression(hat(pi)), mgp=c(5,1,0), cex.lab=2.5)
## title(xlab=expression(n), mgp=c(3.5,1,0), cex.lab=2.5)
## # mtext("(b)", side=1, line=6.5, cex=3)


###################################################
### code chunk number 16: dPL
###################################################
dPL <- function(x, mu, sigma, log=FALSE){
  if (any(x < 0))
    stop(paste("x must be positive", "\n", ""))
  if (any(mu <= 0))
    stop(paste("mu must be positive", "\n", ""))
  if (any(sigma <= 0))
    stop(paste("sigma must be positive", "\n", ""))

  loglik <- log(mu) + 2*log(sigma) - log(sigma+1) +
    log(1+(x^mu)) + (mu-1)*log(x) - sigma*(x^mu)

  if (log == FALSE)
    density <- exp(loglik)
  else density <- loglik
  return(density)
}


###################################################
### code chunk number 17: PLfit
###################################################
# Fitting of tensile strenght data
st <- Fibers$Strenght
theta <- maxlogL(x = st, dist = "dPL",
                 link = list(over = c("mu", "sigma"),
                             fun = c("log_link", "log_link")))
summary(theta)


###################################################
### code chunk number 18: PLexample_a
###################################################
par(family="serif", cex.lab=2.5, cex.axis=2.5, mgp=c(3.5,1.1,0),
    mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5))
hist(st, freq=FALSE, main=NULL, ylim=c(0.,0.8), xlab="u")
curve(dPL(x, mu=theta$fit$par[1], sigma=theta$fit$par[2]), from=0,
      to=4, col=2, lwd=3, add=TRUE)


###################################################
### code chunk number 19: PLexample_b
###################################################
par(family="serif", cex.lab=2.5, cex.axis=2.5, mgp=c(3.5,1.1,0),
    mar=c(5,4,4,2)+0.1, mai=c(0.9,1.5,0.5,0.5))
pPL <- function(q, mu, sigma,
                lower.tail=TRUE, log.p=FALSE){
  if (any(q < 0))
    stop(paste("q must be positive", "\n", ""))
  if (any(mu <= 0))
    stop(paste("mu must be positive", "\n", ""))
  if (any(sigma <= 0))
    stop(paste("sigma must be positive", "\n", ""))

  cdf <- 1 - (1+((sigma/(sigma+1))*q^mu))*exp(-sigma*(q^mu))

  if (lower.tail == TRUE)
    cdf <- cdf
  else cdf <- 1 - cdf
  if (log.p == FALSE)
    cdf <- cdf
  else cdf <- log(cdf)
  cdf
}
library(survival)
KM <- survfit(Surv(st, rep(1,length(st)))~1)
plot(KM, conf.int = FALSE, lwd=2, ylab="Estimated survival function",
     xlab = "u")
curve(1-pPL(x, mu=theta$fit$par[1], sigma=theta$fit$par[2]), from=0,
      to=4, col=2, lwd=3, add=TRUE, xlab = "u")
legend("bottomleft", legend=c("Kaplan-Meier estimator","PL survival function"),
       col=c(1,2), lwd=c(2,3), cex=2.5)


###################################################
### code chunk number 20: maxlogL.Rnw:604-612
###################################################
# Power model implementation
power_logL <- function(x, a, b, log = FALSE){
  p <- a * x[,1]^(-b)
  f <- dbinom(x = x[,2], size = m, prob = p)
  if (log == TRUE)
    density <- log(f) else density <- f
  return(density)
}


###################################################
### code chunk number 21: maxlogL.Rnw:617-628
###################################################
# Power model estimation
m <- 100 # Independent trials
t <- c(1,3,6,9,12,18) # time intervals
p.ob <- c(0.94,0.77,0.40,0.26,0.24,0.16) # Observed proportion
x <- p.ob*m # Correct responses
x <- as.integer(x)
Xi <- matrix(c(t,x), ncol=2, nrow=6)

retention.pwr <- maxlogL(x = Xi, dist = "power_logL", lower = c(0.01,0.01),
                         upper = c(1,1), start = c(0.1,0.1))
summary(retention.pwr)


###################################################
### code chunk number 22: maxlogL.Rnw:637-657
###################################################
# Exponential model implementation
exp_logL <- function(x, a, b, log = FALSE){
  p <- a * exp(-x[,1]*b)
  f <- dbinom(x = x[,2], size = m, prob = p)
  if (log == TRUE)
    density <- log(f) else density <- f
  return(density)
}

# Exponential model estimation
m <- 100 # Independent trials
t <- c(1,3,6,9,12,18) # time intervals
p.ob <- c(0.94,0.77,0.40,0.26,0.24,0.16) # Observed proportion
x <- p.ob*m # Correct responses
x <- as.integer(x)
Xi <- matrix(c(t,x), ncol=2, nrow=6)

retention.exp <- maxlogL(x = Xi, dist = 'exp_logL', lower = c(0.1,0.1),
                         upper = c(2,2), start = c(0.1,0.2))
summary(retention.exp)


###################################################
### code chunk number 23: RetentionPlot (eval = FALSE)
###################################################
## power <- function(x, a, b, log = FALSE){
##   p <- a * x^(-b)
##   return(p)
## }
## expo <- function(x, a, b, log = FALSE){
##   p <- a * exp(-x*b)
##   return(p)
## }
## t <- c(1,3,6,9,12,18) # time intervals
## p.ob <- c(0.94,0.77,0.40,0.26,0.24,0.16) # Observed proportion
## par(family="serif", cex.lab=2.5, cex.axis=2.5, mgp=c(4.5,1.5,0),
##     mar=c(6.5,4,4,2)+0.1, mai=c(1.5,1.5,0.5,0.5))
## plot(t, p.ob, pch=19, xlab="Retention interval (sec.)",
##      ylab="Proportion", cex=2.5)
## a.pwr <- retention.pwr[1]; b.pwr <- retention.pwr[2]
## a.exp <- retention.exp[1]; b.exp <- retention.exp[2]
## curve(power(x, a=a.pwr, b=b.pwr), from=1, to=18, lwd=2, lty=1, add=TRUE)
## curve(expo(x, a=a.exp, b=b.exp), from=1, to=18, lwd=2, lty=2, add=TRUE)
## legend("topright", legend=c("Observed", "Power model", "Exponential model"),
##        lwd=c(0,2,2), pch=c(19,NA,NA), lty=c(0,1,2), cex=2.5)


