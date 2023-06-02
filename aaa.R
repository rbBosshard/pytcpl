fitpoly1 = function(conc, resp, bidirectional = TRUE, verbose = FALSE, nofit = FALSE){
    #median at each conc, for multi-valued responses
    rmds <- tapply(resp, conc, median)
    #get max response and corresponding conc
    if(!bidirectional) mmed = rmds[which.max(rmds)] else mmed = rmds[which.max(abs(rmds))] #shortened this code

    conc_max <- max(conc)

    er_est <- if ((rmad <- mad(resp)) > 0) log(rmad) else log(1e-16)

    ###--------------------- Fit the Model ----------------------###
    ## Starting parameters for the Model
    a0 <- mmed/conc_max #use largest response with desired directionality
    if(a0 == 0) a0 <- .01  #if 0, use a smallish number
    g <- c(a0, # linear coeff (a); set to run through the max resp at the max conc
            er_est) # logSigma (er)

    ## Generate the bound matrices to constrain the model.
    #                a   er
    Ui <- matrix(c( 1,   0,
                    -1,   0),
                byrow = TRUE, nrow = 2, ncol = 2)

    if(!bidirectional){
    bnds <- c(0, -1e8*abs(a0)) # a bounds (always positive)

    } else {
    bnds <- c(-1e8*abs(a0), -1e8*abs(a0)) # a bounds (positive or negative)

    }

    Ci <- matrix(bnds, nrow = 2, ncol = 1)

    ## Optimize the model
    fit <- try(constrOptim(g,
                            tcplObj,
                            ui = Ui,
                            ci = Ci,
                            mu = 1e-6,
                            method = "Nelder-Mead",
                            control = list(fnscale = -1,
                                            reltol = 1e-10,
                                            maxit = 6000),
                            conc = conc,
                            resp = resp,
                            fname = "poly1"),
                silent = !verbose)

    print(fit)
}
