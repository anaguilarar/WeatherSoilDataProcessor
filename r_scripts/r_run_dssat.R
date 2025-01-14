rm(list = ls())
call_package = function(package){
  if(!suppressMessages(suppressWarnings(require(package, character.only = T)))){
    install.packages(package);suppressMessages(suppressWarnings(require(package, character.only = T)))}  
}

call_package('tidyverse')
call_package('DSSAT')
call_package('yaml')

rundssat <-function(i,path.to.extdata,TRT,crop_code, bin_dssatpath = NULL, exp_filename = 'EXTE'){
  setwd(path.to.extdata)

  if(is.null(bin_dssatpath)){
    bin_dssatpath = "/opt/DSSAT/v4.8.1.40/dscsm048"
  }
  # Generate a DSSAT batch file using a tibble
  options(DSSAT.CSM= bin_dssatpath)
  v48 = tibble(FILEX=paste0(exp_filename, formatC(width = 4, as.integer((i)), flag = "0"),'.',crop_code,'X'), TRTNO=TRT, RP=1, SQ=0, OP=0, CO=0)
  
  write_dssbatch(v48, file_name="DSSBatch.v48")
  # Run DSSAT-CSM
  run_dssat(file_name="DSSBatch.v48",suppress_output = TRUE)
  # Change output file name
  new_file <-  paste0(exp_filename, formatC(width = 4, as.integer((i)), flag = "0"),'.OUT')
  # Check if the output file already exists and remove it if it does
  if (file.exists(new_file)) {
    file.remove(new_file)
  }
  file.rename("Summary.OUT",new_file)
  gc()
}


args <- commandArgs(trailingOnly = TRUE)

config = suppressWarnings(yaml::read_yaml(args[1]))
path.to.extdata = config$GENERAL$working_path
bin_dssatpath = 'C:/DSSAT48/DSCSM048.exe'
crop_code = config$MANAGEMENT$crop_code
TRT = 1:30

i = config$GENERAL$roi_id
exp_filename = config$GENERAL$output_name

rundssat(i,path.to.extdata,TRT,crop_code, bin_dssatpath = bin_dssatpath, exp_filename = exp_filename)
