rm(list = ls())

##
call_package = function(package){
  if(!suppressMessages(suppressWarnings(require(package, character.only = T)))){
    install.packages(package);suppressMessages(suppressWarnings(require(package, character.only = T)))}  
}


plot_output <- function(
    list_output = list(output),
    vars        = outputNames[-(1:3)],
    leg         = paste( "Run", 1:length(list_output) ),
    leg_title   = "LEGEND",
    nrow_plot   = ceiling( sqrt((length(vars)+1) * 8/11) ),
    ncol_plot   = ceiling( (length(vars)+1)/nrow_plot ),
    lty         = rep(1,length(list_output)),
    lwd         = rep(3,length(list_output))
) {
  par( mfrow=c(nrow_plot,ncol_plot), mar=c(2, 2, 2, 1) )
  if (!is.list(list_output)) list_output <- list(list_output) ; nlist <- length(list_output)
  col_vars <- match(vars,outputNames)                         ; nvars <- length(vars)
  for (iv in 1:nvars) {
    c       <- col_vars[iv]
    g_range <- range( sapply( 1:nlist, function(il){range(list_output[[il]][,c])} ) )
    plot( list_output[[1]][,1], list_output[[1]][,c],
          xlab="", ylab="", cex.main=1,
          main=paste(outputNames[c]," ",outputUnits[c],sep=""),
          type='l', col=1, lty=lty[1], lwd=lwd[1], ylim=g_range )
    if (nlist >= 2) {
      for (il in 2:nlist) {
        points( list_output[[il]][,1], list_output[[il]][,c],
                col=il, type='l', lty=lty[il], lwd=lwd[il] )
      }
    }
    if ( (iv%%(nrow_plot*ncol_plot-1)==0) || (iv==nvars) ) {
      plot(1,type='n', axes=FALSE, xlab="", ylab="")
      legend("bottomright", leg, lty=lty, lwd=lwd, col=1:nlist, title = leg_title)
    }
  }
}

run_model <- function(p     = params,
                      w     = matrix_weather,
                      calf  = calendar_fert,
                      calpC = calendar_prunC,
                      calpT = calendar_prunT,
                      caltT = calendar_thinT,
                      n     = NDAYS) {
  .Fortran('CAF2021', p,w,calf,calpC,calpT,caltT,n,NOUT,
           matrix(0,n,NOUT))[[9]]
}

##
nc <- 6 ; nt <- 3 ; nz <- nt*2

yNames <- yUnits <- list()
yNames[[1]]  <- c( "Time", "year", "doy" )
yNames[[2]]  <- c( paste0("Ac(",1:nc,")")      , paste0("At(",1:nt,")"),
                   paste0("fNgrowth(",1:nc,")"), paste0("fTran(",1:nc,")") )

yNames[[3]]  <- c( "Cabg_f"         , paste0("harvCP(",1:nc,")"),
                   "harvDM_f_hay"   , paste0("LAI(",1:nc,")") )

yNames[[4]]  <- c( "CabgT_f"              , paste0("CAtree_t(",1:nt,")"),
                   paste0("h_t(",1:nt,")"), paste0("LAIT_c(",1:nc,")"),
                   paste0("treedens_t(",1:nt,")") )

yNames[[5]]  <- c( "Csoil_f"  , "Nsoil_f"   )


yNames[[6]]  <- c( paste0("CST_t(",1:nt,")"), paste0("SAT_t(",1:nt,")") )


yNames[[7]]  <- c( "Nmineralisation_f", "Nfert_f"    , "NfixT_f",
                   "NsenprunT_f"      , "Nsenprun_f" ,
                   "Nleaching_f"      , "Nemission_f",
                   "Nrunoff_f"        , "Nupt_f"     , "NuptT_f" )

                  
yNames[[8]]  <- c( paste0("CLITT(",1:nc,")")    , paste0("NLITT(",1:nc,")"),
                   paste0("harvCST_t(",1:nt,")"), paste0("harvNST_t(",1:nt,")") )
                  
yNames[[9]]  <- c( "CsenprunT_f", "Csenprun_f" , "Rsoil_f", "Crunoff_f" )


yNames[[10]] <- c( "WA_f"  ,
                   "Rain_f", "Drain_f", "Runoff_f" , "Evap_f"    ,
                   "Tran_f", "TranT_f", "Rainint_f", "RainintT_f" )

                  
yNames[[11]] <- c( "C_f"      ,  "gC_f", "dC_f", "prunC_f", "harvCP_f" ) 


yNames[[12]] <- c( "CT_f"     , "gCT_f", "harvCBT_f", "harvCPT_f", "harvCST_f" )


yNames[[13]] <- c( paste0("CPT_t(",1:nt,")"), paste0("harvCPT_t(",1:nt,")"),
                   paste0("harvNPT_t(",1:nt,")") )

yNames[[14]] <- c( "DayFl",
                   "DVS(1)", "SINKP(1)", "SINKPMAXnew(1)", "PARMA(1)",
                   "DVS(2)", "SINKP(2)", "SINKPMAXnew(2)", "PARMA(2)" )

yNames[[15]] <- c( "CR_f", "CW_f", "CL_f", "CP_f",
                   paste0("CRT_t(",1:nt,")") , paste0("CBT_t(",1:nt,")"),
                   paste0("CLT_t(",1:nt,")") )

yNames[[16]] <- c( paste0("LAIT_t(",1:nt,")"), paste0("fTranT_t(",1:nt,")") )


yNames[[17]] <- c( "D_Csoil_f_hay", "D_Csys_f_hay", "D_Nsoil_f_hay" )


yNames[[18]] <- c( "NfixT_f_hay", "Nleaching_f_hay" )


yNames[[19]] <- "Shade_f"


yNames[[20]] <- paste0("z(",1:nz,")")


yNames[[21]] <- "f3up"


yNames[[22]] <- paste0("DayHarv(",1:nc,")")


yNames[[23]] <- paste0("fNgrowth_t(",1:nt,")")


outputNames  <- unlist(yNames) ; outputUnits <- unlist(yUnits)

NOUT         <- as.integer( length(outputNames) )

call_package('yaml')

## reading parameters

args <- commandArgs(trailingOnly = TRUE)

config = suppressWarnings(yaml::read_yaml(args[1]))

# working path
wp = config$GENERAL$working_path

# management
calendar_prunC_ = t(unname(as.matrix(data.frame(config$MANAGEMENT$coffe_prun[[1]]))))
calendar_fert_ = t(unname(as.matrix(data.frame(config$MANAGEMENT$fertilization[[1]]))))
calendar_prunT_ <- array( -1, c(3,100,3) )
calendar_prunT_[,,1] = unname(as.matrix(data.frame(config$MANAGEMENT$tree_prun[[1]][[1]])))
calendar_prunT_[,,2] = unname(as.matrix(data.frame(config$MANAGEMENT$tree_prun[[1]][[2]])))
calendar_prunT_[,,3] = unname(as.matrix(data.frame(config$MANAGEMENT$tree_prun[[1]][[3]])))
calendar_thinT_ <- array( -1, c(3,100,3) )
calendar_thinT_[,,1] = unname(as.matrix(data.frame(config$MANAGEMENT$tree_thinning[[1]][[1]])))
calendar_thinT_[,,2] = unname(as.matrix(data.frame(config$MANAGEMENT$tree_thinning[[1]][[2]])))
calendar_thinT_[,,3] = unname(as.matrix(data.frame(config$MANAGEMENT$tree_thinning[[1]][[3]])))
parameters = config$PARAMETERS[[1]]

# weather
weather = t(unname(as.matrix(data.frame(config$WEATHER[[1]]))))
totaldays = config$NDAYS

NMAXDAYS              <- as.integer(10000)
NWEATHER              <- as.integer(8)
matrix_weather        <- matrix( 0., nrow=NMAXDAYS, ncol=NWEATHER )

matrix_weather[1:dim(weather)[1],] = weather

# bin config
MODEL_dll      <- config$GENERAL$caf_dll_path
dyn.load( MODEL_dll )
#
output <- run_model( p     = parameters,
                     w     = matrix_weather,
                     calf  = calendar_fert_,
                     calpC = calendar_prunC_,
                     calpT = calendar_prunT_,
                     caltT = calendar_thinT_,
                     n     = totaldays )

dfoutput = data.frame(output)
names(dfoutput) = unlist(yNames)
write.csv(dfoutput, file.path(wp,"output.csv"))

png(file.path(wp,"harvesting.png"))
plot_output( output, vars=c("Ac(1)","Ac(2)","harvDM_f_hay") )
dev.off()
