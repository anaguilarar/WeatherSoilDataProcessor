
rm(list = ls())
call_package = function(package){
  if(!suppressMessages(suppressWarnings(require(package, character.only = T)))){
    install.packages(package);suppressMessages(suppressWarnings(require(package, character.only = T)))}  
}
#conda install conda-forge::r-pak
#conda install conda-forge::r-dssat
call_package('tidyverse')
call_package('DSSAT')
call_package('yaml')

create_experimental_files <-function(path_to_config){
  
  ##read paths
  #Read config
  config = suppressWarnings(yaml::read_yaml(path_to_config))
  
  SOIL = config$SOIL
  MANAGEMENT = config$MANAGEMENT
  GENERAL = config$GENERAL
  #source(GENERAL[['dssat_functions_path']])
  i = GENERAL[['roi_id']]
  #Read soil
  SDUL = SOIL[['SDUL']]
  SLLL = SOIL[['SLLL']]
  SLB = SOIL[['SLB']]
  ID_SOIL=SOIL[['ID_SOIL']]

  #Read in original FileX
  file_x <- read_filex(MANAGEMENT[['template_path']])
  #Set the experimental directory
  setwd(GENERAL[['working_path']])

  #Make proposed changes to FileX
  file_x$FIELDS$WSTA<-config$WEATHER$file_name   
  file_x$FIELDS$ID_SOIL<-as.numeric(ID_SOIL)
  file_x$CULTIVARS$CR <- MANAGEMENT[['crop_code']]
  file_x$CULTIVARS$INGENO <- MANAGEMENT[['varietyid']]
  
  #Assume a proportion between wilting point and field capacity as initial condition
  file_x$`INITIAL CONDITIONS`$SH2O<- mapply(function(sdul, slll, index) {
    slll + ((sdul-slll) * index)}, list(SDUL), list(SLLL), MoreArgs = list(index = SOIL[['index_soilwat']]), SIMPLIFY = FALSE)
  #soil initial conditions
  file_x$`INITIAL CONDITIONS`$ICBL <- list(SLB)
  # make sure that they have the same length
  file_x$`INITIAL CONDITIONS`$SH2O <- list(file_x$`INITIAL CONDITIONS`$SH2O[[1]][1:length(SLB)])
  file_x$`INITIAL CONDITIONS`$SNH4 <- list(file_x$`INITIAL CONDITIONS`$SNH4[[1]][1:length(SLB)])
  file_x$`INITIAL CONDITIONS`$SNO3 <- list(file_x$`INITIAL CONDITIONS`$SNO3[[1]][1:length(SLB)])
  
  file_x$`INITIAL CONDITIONS`$ICDAT <- as.POSIXct(as.Date(MANAGEMENT[['startingDate']]))
  file_x$`PLANTING DETAILS`$PDATE <- as.POSIXct(as.Date(MANAGEMENT[['plantingDate']]))
  
  if(MANAGEMENT[['fertilizer']] == T){
    file_x$`FERTILIZERS (INORGANIC)`$FDATE <- as.POSIXct(MANAGEMENT[['plantingDate']]) #fertilizer at planting
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`$TNAME <- paste0(file_x$`FERTILIZERS (INORGANIC)`$FERNAME[1], " Planting 0")
  }else{
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`$TNAME <- paste0("Initial planting")
  }
  
  file_x$`HARVEST DETAILS`$HDATE <- as.POSIXct(as.Date(MANAGEMENT[['harvestDate']]))
  file_x$`SIMULATION CONTROLS`$SDATE <- as.POSIXct(as.Date(MANAGEMENT[['startingDate']]))
  file_x$`SIMULATION CONTROLS`$NYERS <- GENERAL[['number_years']]
  
  for (j in 1:MANAGEMENT[['plantingWindow']]){
    file_x$`INITIAL CONDITIONS`<- file_x$`INITIAL CONDITIONS` %>% add_row(!!!file_x$`INITIAL CONDITIONS`[file_x$`INITIAL CONDITIONS`$C==1,])
    file_x$`INITIAL CONDITIONS`[1+j,]$C <- 1+j
    file_x$`INITIAL CONDITIONS`[1+j,]$ICDAT <- as.POSIXct(as.Date(MANAGEMENT[['startingDate']])) %m+% weeks(j)
    
    file_x$`PLANTING DETAILS` <- file_x$`PLANTING DETAILS` %>% add_row(!!!file_x$`PLANTING DETAILS`[file_x$`PLANTING DETAILS`$P==1,])
    file_x$`PLANTING DETAILS`[1+j,]$P <- 1+j
    file_x$`PLANTING DETAILS`[1+j,]$PDATE <- as.POSIXct(as.Date(MANAGEMENT[['plantingDate']])) %m+% weeks(j)
    
    if(MANAGEMENT[['fertilizer']] == T){
      file_x$`FERTILIZERS (INORGANIC)` <- file_x$`FERTILIZERS (INORGANIC)` %>% add_row(!!!file_x$`FERTILIZERS (INORGANIC)`[file_x$`FERTILIZERS (INORGANIC)`$F==1,])
      file_x$`FERTILIZERS (INORGANIC)`[1+j,]$F <- 1+j
      file_x$`FERTILIZERS (INORGANIC)`[1+j,]$FDATE <- as.POSIXct(as.Date(MANAGEMENT[['plantingDate']])) %m+% weeks(j)
    }
    
    file_x$`HARVEST DETAILS` <- file_x$`HARVEST DETAILS` %>% add_row(!!!file_x$`HARVEST DETAILS`[file_x$`HARVEST DETAILS`$H==1,])
    file_x$`HARVEST DETAILS`[1+j,]$HDATE <- as.POSIXct(as.Date(MANAGEMENT[['harvestDate']])) %m+% weeks(j)
    file_x$`HARVEST DETAILS`[1+j,]$H <- 1+j
    
    file_x$`SIMULATION CONTROLS`<- file_x$`SIMULATION CONTROLS` %>% add_row(!!!file_x$`SIMULATION CONTROLS`[file_x$`SIMULATION CONTROLS`$N==1,])
    file_x$`SIMULATION CONTROLS`[1+j,]$N <- 1+j
    file_x$`SIMULATION CONTROLS`[1+j,]$SDATE <- as.POSIXct(as.Date(MANAGEMENT[['startingDate']])) %m+% weeks(j)
    
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------` <- file_x$`TREATMENTS                        -------------FACTOR LEVELS------------` %>% 
      add_row(!!!file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`$N==1,])
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$N <- 1+j
    if(MANAGEMENT[['fertilizer']] == T){
      file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$TNAME <- paste0(file_x$`FERTILIZERS (INORGANIC)`$FERNAME[1], " + ", j ,"weeks")
    }else {
      file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$TNAME <- paste0("Planting + ", j ,"weeks")
    }
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$IC <- 1+j
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$MP <- 1+j
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$MH <- 1+j
    file_x$`TREATMENTS                        -------------FACTOR LEVELS------------`[1+j,]$SM <- 1+j
  }

  print(paste0(GENERAL[['working_path']], '/', GENERAL[['output_name']], formatC(width = 4, as.integer((i)), flag = "0")))
  #Overwrite original FileX with new values
  write_filex(file_x,paste0(GENERAL[['output_name']], formatC(width = 4, as.integer((i)), flag = "0"),'.',MANAGEMENT[['crop_code']],'X'))
  rm(file_x)
  gc()
}
#path_to_config = 'D:/OneDrive - CGIAR/scripts/agwise/WeatherSoilDataProcessor/tmp_experimental_file_config.yaml'
#path_config = 'D:/OneDrive - CGIAR/scripts/agwise/WeatherSoilDataProcessor/tmp_experimental_file_config.yaml'

args <- commandArgs(trailingOnly = TRUE)
path_to_config = 'D:/OneDrive - CGIAR/projects/suelos_honduras/dssat_runs_hnd/puertocastilla_020108/clay/experimental_file_config.yaml'

create_experimental_files(args[1])



