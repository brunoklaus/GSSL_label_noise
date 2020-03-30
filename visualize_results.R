
library(tidyverse)

setwd("/home/klaus/eclipse-workspace/NoisyGSSL")
DF_FOLDER = file.path(getwd(),"results","csvs")


OUTPUT_PREFIX = "out_"
ALG_PREFIX = "alg_"



#' Combine dataframes. The columns in each csv may differ.
#'
#' @param df_list a list with each dataframe
#' @return A tibble, combining each of the given dataframes.
dfCombine <- function(df_list){
  #Fix missing columns by introducing NA
  all_colnames = unique(unlist(sapply(df_list,
                                      colnames)))
  for (i in 1:length(df_list)) {
    for (j in all_colnames) {
      if (!j %in% colnames((df_list[[i]]))){
        df_list[[i]] <-  cbind(df_list[[i]],rep(NA,nrow(df_list[[i]])))
        colnames(df_list[[i]])[ncol(df_list[[i]])] <- j
      }
    }
  }
  df <- as.tibble(do.call(rbind,df_list))
  return(df)
}


#' Reads a dataframe
#'
#' @param fPath path to csv file
#' @return the dataframe
dfRead <- function(fPath) {
  df <- read.csv(fPath,sep=",")
  df <- df[,colnames(df) != "X"]
  if ("dataset_sd" %in% colnames(df)){
    df$dataset <- paste(df$dataset,"_sd=",df$dataset_sd,sep="")
    df <- df[,colnames(df) != "dataset_sd"]
  }
  return(df)
}

df_debug  <- dfRead(file.path(DF_FOLDER,"exp_debug_joined.csv"))
df_ldst_filterstats <- dfRead(file.path(DF_FOLDER,"filter_LDST_stats_joined.csv"))

df_digit1 <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v1_digit1_joined.csv"))
df_g241c <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v1_g241c_joined.csv"))
df_chap_v1_lgc <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v1_lgc_alpha=0.99_joined.csv"))

df_chap_g241n_coil <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v1_g241n_joined.csv"))


df_chap_v3 <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v3_joined.csv"))
df_chap_v3_p2 <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v3_p2_joined.csv"))

df_chap_v2 <- dfRead(file.path(DF_FOLDER,"exp_chapelle_v2_joined.csv"))

df_ldst_filterstats <- dfRead(file.path(DF_FOLDER,"filter_LDST_stats_joined.csv"))

df_ldst_chapelle <- dfRead(file.path(DF_FOLDER,"experiment3","chapelle","filter_LDST_chapelle_updated_joined.csv"))
df_ldst_spiral  <- dfRead(file.path(DF_FOLDER,"experiment3","spiral","filter_LDST_spiral_updated_joined_backup.csv"))


df_gtam_constantprop <- dfCombine(list(
  dfRead(file.path(DF_FOLDER,"exp_chapelle_v4_gtamConstantProp_joined.csv")),
  dfRead(file.path(DF_FOLDER,"exp_chapelle_v4_gtamConstantProp_g241c_joined.csv"))
))
df_gtam_constantprop$alg_algorithm <- "GTAMc"
df_gtam_constantprop <- df_gtam_constantprop[,!grepl("constantP",colnames(df_gtam_constantprop))]
df_gtam_constantprop$spec_name <- "Experiment 1"


df_exp1_chapelle <- dfCombine(
  lapply(list("exp_chapelle_v1_digit1_joined.csv",
              "exp_chapelle_v1_g241c_joined.csv",
              "exp_chapelle_v1_g241n_joined.csv",
              "exp_chapelle_v1_lgc_alpha=0.99_joined.csv",
              "exp1_RF_joined.csv",
              "exp1_USPS_part2_joined.csv",
              "21Ag_exp1_p1_joined.csv",
              "27Ag_exp1_MR_p1_joined.csv",
              "27Ag_exp1_MR_p2_joined.csv",
              "27Ag_exp1_MR_p3_joined.csv"
  ), function(x)dfRead(file.path(DF_FOLDER,"experiment1",x)))
)

df_exp1_chapelle <- dplyr::filter(df_exp1_chapelle,is.na(alg_p) | alg_p >= 1) 
df_exp1_chapelle$alg_constantProp = F
df_exp1_chapelle$alg_n_estimators = 100
df_exp1_chapelle$spec_name = "Experiment 1"

df_exp2_chapelle <- dfCombine(
  lapply(list("exp_chapelle_v3_joined.csv",
              "exp_chapelle_v3_p2_joined.csv",
              "exp2_USPS_part1_joined.csv",
              "exp2_USPS_part2_joined.csv",
              "exp2_RF_joined.csv",
              "27Ag_exp2_MR_p1_joined.csv",
              "27Ag_exp2_MR_p2_joined.csv"
  ), function(x)dfRead(file.path(DF_FOLDER,"experiment2",x)))
)
df_exp2_chapelle$spec_name = "Experiment 2"
df_exp2_chapelle <- df_exp2_chapelle[,!colnames(df_exp2_chapelle)%in%c("alg_constantProp","alg_n_estimators")]
df_exp2_chapelle <- dplyr::filter(df_exp2_chapelle,is.na(alg_p) | alg_p >= 1) 



df_chap_filtered <-  dfRead(file.path(DF_FOLDER,"exp_chapelle_with_filter_USPS_joined.csv"))
df_chap_filtered$alg_algorithm <- as.character(df_chap_filtered$alg_algorithm)
df_chap_filtered$flt_filter <- as.character(df_chap_filtered$flt_filter)

df_chap_filtered[df_chap_filtered$flt_filter=="LDST","alg_algorithm"] <- 
  sapply(df_chap_filtered[df_chap_filtered$flt_filter=="LDST","alg_algorithm"],
         function(x){paste0("LDST+",x)})
#df_chap_filtered <- df_chap_filtered[,!grepl("flt",colnames(df_chap_filtered))]


df_USPS_filterstats <- dfRead(file.path(DF_FOLDER,"filter_LDST_USPS_joined.csv"))

df_USPS_filtered <- dfRead(file.path(DF_FOLDER,"exp_chapelle_with_filter_USPS_joined.csv"))

df_Aug21_filtered <- dfCombine(
  lapply(list("exp_chapelle_with_filter_digit1_joined.csv",
              "exp_chapelle_with_filter_USPS_joined.csv"
  ), function(x)dfRead(file.path(DF_FOLDER,"Aug21_exp4",x))))



df_USPS_exp1 <- dfRead(file.path(DF_FOLDER,"experiment1",
                                 "exp1_USPS_part1_joined.csv"))

df_digit1_exp4 <- dfRead(file.path(DF_FOLDER,"Aug21_exp4",
                                   "exp_chapelle_with_filter_digit1.csv"))


df_Ag24_exp4 <- dfRead(file.path(DF_FOLDER,"Ag24_exp4",
                                 "24Ag_exp4_joined.csv"))

df_Aug29_exp4 <- dfCombine(
  lapply(list("29Ag_exp4_MRF_g241c_lgc_joined.csv",
              "29Ag_exp4_MRF_COIL2_gfhf_joined.csv",
              "29Ag_exp4_MRF_USPS_gfhf_joined.csv",
              "29Ag_exp4_MRF_Digit1_gfhf_joined.csv",
              "29Ag_exp4_MRF_g241n_gfhf_joined.csv"
  ), function(x)dfRead(file.path(DF_FOLDER,"Aug29_exp4",x))))
df_Aug29_exp4$spec_name = "Experiment 4"



df_Aug29_exp4_MRF_MR <- dfCombine(
  lapply(list("29Ag_exp4_MRF_COIL2_MR_joined.csv",
              "29Ag_exp4_MRF_Digit1_MR_joined.csv",
              "29Ag_exp4_MRF_g241c_MR_joined.csv",
              "29Ag_exp4_MRF_g241n_MR_joined.csv",
              "29Ag_exp4_MRF_USPS_MR_joined.csv"
  ), function(x)dfRead(file.path(DF_FOLDER,"Aug29_exp4_MRF_MR",x))))
df_Aug29_exp4$spec_name = "Experiment 4"



df_Nov13_LGCLVO <- dfCombine(
  lapply(list("13Dez_filter_SIIS_LGCLVO_ISOLET_p1_joined.csv",
              "13Dez_filter_SIIS_LGCLVO_ISOLET_p2_joined.csv",
              "13Dez_filter_SIIS_LGCLVO_ISOLET_p3_joined.csv",
              "13Dez_filter_SIIS_LGCLVO_ISOLET_p4_joined.csv"
              
              
              
  ), function(x)dfRead(file.path(DF_FOLDER,"12Dez",x))))


df_Nov13_LGCLVO$spec_name = "Experiment 3"
df_Nov13_LGCLVO <- dplyr::filter(df_Nov13_LGCLVO,input_dataset=="Digit1")



ds <- "Digit1"
aff <- "constant"


if (aff == "constant"){
  df_Nov13_LGCLVO <- dfCombine(
    lapply(list(paste0("SIIS/",aff,"/12Dez_SIIS_constant_chap_",ds,"_joined.csv"),
                paste0("GTAM/",aff,"/deg/12Dez_GTAM_deg_constant_chap_",ds,"_joined.csv"),
                paste0("GTAM/",aff,"/nodeg/12Dez_GTAM_constant_chap_",ds,"_joined.csv"),
                paste0("LGC/",aff,"/12Dez_LGC_constant_chap_",ds,"_joined.csv")
                
                
    ), function(x)dfRead(file.path(DF_FOLDER,"12Dez",x))))
  
  df_Nov13_LGCLVO$spec_name = "Experiment 1"
} else{
  df_Nov13_LGCLVO <- dfCombine(
    lapply(list(paste0("SIIS/",aff,"/12Dez_SIIS_chap_",ds,"_joined.csv"),
                paste0("GTAM/",aff,"/deg/12Dez_GTAM_deg_chap_",ds,"_joined.csv"),
                paste0("GTAM/",aff,"/nodeg/12Dez_GTAM_chap_",ds,"_joined.csv"),
                paste0("LGC/",aff,"/12Dez_LGC_chap_",ds,"_joined.csv"),
                paste0("GFHF/",aff,"/12Dez_GFHF_chap_",ds,"_joined.csv")
                
    ), function(x)dfRead(file.path(DF_FOLDER,"12Dez",x))))
  df_Nov13_LGCLVO$spec_name = "Experiment 2"
}
if ("SIIS" %in% unique(df_Nov13_LGCLVO$alg_algorithm)){
df_Nov13_LGCLVO[df_Nov13_LGCLVO$alg_algorithm == "SIIS","alg_mu"] <- df_Nov13_LGCLVO[df_Nov13_LGCLVO$alg_algorithm == "SIIS","alg_alpha"] 
df_Nov13_LGCLVO[df_Nov13_LGCLVO$alg_algorithm == "SIIS","alg_alpha"] <- NA

df_Nov13_LGCLVO[df_Nov13_LGCLVO$alg_algorithm == "SIIS","alg_mu_2"] <- df_Nov13_LGCLVO[df_Nov13_LGCLVO$alg_algorithm == "SIIS","alg_beta"] 
}
if ("GTAM" %in% unique(df_Nov13_LGCLVO$alg_algorithm)){
  
  df_Nov13_LGCLVO$alg_weigh_by_degree <- as.character(df_Nov13_LGCLVO$alg_weigh_by_degree)
  df_Nov13_LGCLVO[df_Nov13_LGCLVO$alg_algorithm == "GTAM" &
                    is.na(df_Nov13_LGCLVO$alg_weigh_by_degree),"alg_weigh_by_degree"] <- "False"
  df_Nov13_LGCLVO$alg_weigh_by_degree <- as.factor(df_Nov13_LGCLVO$alg_weigh_by_degree)
  colnames(df_Nov13_LGCLVO)[colnames(df_Nov13_LGCLVO) == "alg_weigh_by_degree"] <- "use_deg"
}


ds <- "g241n"
aff <- "constant"
df_Dez21 <-  dfCombine(
  lapply(list(paste0("21Dez_eigfuncs_",ds,"_joined.csv")
              
  ), function(x)dfRead(file.path(DF_FOLDER,"21Dez",x))))

df_ldst_chapelle <-dfCombine(
  lapply(list("31Dez_LDST3_mod_chap_joined.csv"
              
  ), function(x)dfRead(file.path(DF_FOLDER,x))))
df_ldst_chapelle$flt_weigh_by_degree = levels(df_ldst_chapelle$flt_weigh_by_degree)[1]
df_ldst_chapelle$flt_useZ = levels(df_ldst_chapelle$flt_useZ)[1]
df_ldst_chapelle$flt_gradient_fix=levels(df_ldst_chapelle$flt_gradient_fix)[1]

df_list = list(
  df_ldst_chapelle)
df = dfCombine(df_list)
if (nrow(df) == 0){stop("DF has 0 rows!")}
#############################################################
# DF fixes

if ("out_filter_recall_mean" %in% colnames(df)) {
  #df$out_filter_FNR_mean <- 1 - df$out_filter_recall_mean
  #df$out_filter_FNR_sd <- df$out_filter_recall_sd
  #df$out_filter_fallout_mean <- 1 - df$out_filter_specificity_mean
  #df$out_filter_fallout1_sd <- df$out_filter_specificity_sd
  
  
  df$out_filter_precision_mean[is.na(df$out_filter_precision_mean)] <- 0
  df$out_filter_recall_mean[is.na(df$out_filter_recall_mean)] <- 0
  df$out_filter_f1_score_mean[is.na(df$out_filter_f1_score_mean)] <- 0
}
if ("alg_alpha" %in% colnames(df)){
  df$alg_mu[!is.na(df$alg_alpha)] = sapply(df$alg_alpha[!is.na(df$alg_alpha)],
                                           function(x){return((1-x)/x)}) 
  df = subset(df, select=-c(alg_alpha))
}
df$alg_algorithm = as.character(df$alg_algorithm)
df[df$alg_algorithm == "MREG","alg_algorithm"] = "LE"
df$alg_algorithm = as.factor(df$alg_algorithm)

REMOVE_KEYWORDS = c("min","max","values")
for(x in REMOVE_KEYWORDS){
  df <- df[,!grepl(pattern = x, x = colnames(df))]
}
#Round off values
df[,sapply(df,class)=="numeric"] <-  round(df[,sapply(df,class)=="numeric"],digits=5)
output_variables <- c("experiments","mean_acc","sd_acc","elapsed_time",
                      "min_acc","max_acc","median_acc","each_acc")

output_variables <- colnames(df)[sapply(colnames(df),
                                        function(x){grepl("out_",x)})]

affmat_variables <- colnames(df)[sapply(colnames(df),
                                        function(x){grepl("aff_",x)})]
hyperparams <- !(colnames(df) %in% c(affmat_variables,output_variables))
hyperparams <- colnames(df)[hyperparams]


df[,grepl("acc_",colnames(df))] <- round(100* df[,grepl("acc_",colnames(df))],digits=2)


############################################################################
####################################################
#' Creates a dataframe corresponding to an experiment
#' 
#' 
texDf <- function(df,x_var,use_sd=T, out_var="acc",
                  out_var_statistic="mean",type_var = "") {
  out_var = paste0(OUTPUT_PREFIX,out_var)
  f_cond <- Vectorize(function(x){
    return (startsWith(x,OUTPUT_PREFIX))
  })
  ignore_var = colnames(df)[f_cond(colnames(df))]
  df <- as.tibble(df)
  
  #ignore_var <- ignore_var[!ignore_var %in% c(x_var,out_var)] 
  
  #' Converts a dataframe row to a string listing the names and values
  #' of that dataframe's columns, except if the value is NA.  
  #' 
  #' @param a a row from some dataframe.
  #' @param sep the string used to separate each (name,value) pair
  toStrExceptNA <- function(a,sep=";"){
    temp <- as.logical(!is.na(a[1,]))
    a <- a[1,temp]
    if (length(a)==0) return("")
    return(paste0(sapply(1:length(a),
                         function(i){
                           paste0(colnames(a)[i],"=",
                                  as.character(a[[1,i]]),
                                  sep)
                         }
    ),collapse=""))
    
  }
  
  #b1: filters out columns that have only one value (that value could be NA)
  b1 <- unname(apply(df,2,function(x){
    return(length(unique(x[!is.na(x) & !x==""])))
  }
    ) > 1)
    #b2: filters out columns in 'ignore_var' 
    b2 <- !colnames(df) %in% c(x_var,out_var,type_var,ignore_var)
    
    

    #g: string describing the setting
    g <- ""
    if(any(b1 & b2)){
      g <- sapply(1:nrow(df[,b1 & b2]),
                  function(i){
                    return(toStrExceptNA(df[i,b1 & b2]))
                  }
      )
    }
    b3 <- (startsWith(colnames(df),OUTPUT_PREFIX))
    
    #g_compl: string describing the non-output variables that appear
    #with a single, unique value
    g_compl <- ""
    if (any(!b1 & !b3)){
      g_compl <- sapply(1:nrow(df),
                        function(i){
                          return(toStrExceptNA(df[i,!b1 & !b3],sep=";\n"))
                        }
      )
    }
    
    
    df_fixed = df[,!b1 & !b3]
    df_vars = df[,b1 & b2]
    
    n <- nrow(df)
    new_df <- as.tibble(data.frame(sd=numeric(n)))
    new_df[x_var] <- sapply(df[,x_var],as.numeric)
    new_df[out_var] <- sapply(df[,paste0(out_var,"_",out_var_statistic)],as.numeric)
    new_df$sd <- unlist(df[,paste0(out_var,"_sd")])
    
    new_df <- cbind(df_vars,new_df[x_var],new_df[out_var],new_df["sd"])
    return(list(new_df, paste0(g_compl[1]) ))
    
    
    
    
}


####################################################
#' Creates TEX table
#' 
#' 
createTexTable <- function(fPath,tex_df,setting,x_var){
  n_settings = which(colnames(tex_df)==x_var)-1
  print(colnames(tex_df)[(n_settings+1):ncol(tex_df)])
  
  for (i in 1:ncol(tex_df)){
    if(is.numeric(tex_df[,i])){
      tex_df[,i] <- round(tex_df[,i], digits=4)
    }
    tex_df[,i] <- as.character(tex_df[,i])
  }
  
  
  fileConn <-file(fPath,open = "w")
  writeLines(text = c("\\begin{table}[h]","\\tiny",
                      paste0("\\begin{tabular}{@{}",paste0(rep("l",
                                                               n_settings+1+(ncol(tex_df)-(n_settings+1))/2),collapse = ""),
                             "@{}}","\\toprule")),con = fileConn)
  close(fileConn)
  fileConn <-file(fPath,open = "a")
  
  translate <-t(as.data.frame(list(c("noise_corruption_level", "Noise"),
                                   c("alg_alpha","$\\alpha$"),
                                   c("alg_mu","$\\mu$"),
                                   c("alg_p","p"),
                                   c("alg_algorithm","Algorithm"),
                                   c("out_acc_labeled","Accuracy (labeled)"),
                                   c("flt_gradient_fix","Stability fix"),
                                   
                                   c("out_acc_unlabeled","Accuracy (unlabeled)"),
                                   
                                   c("CMN_acc","Accuracy (w/ class mass normalization)"),
                                   c("acc","Accuracy"))))
  translate <- as.data.frame(translate)
  
  colnames(translate) <- c("old","new")
  rownames(translate) <- 1:nrow(translate)
  translate$old <- as.character(translate$old)
  translate$new <- as.character(translate$new)
  
  
  for (i in 1:nrow(translate)) {
    if (translate$old[[i]] %in% colnames(tex_df)){
      #print(translate$old[[i]])
      colnames(tex_df)[which(colnames(tex_df) == translate$old[[i]])] <- translate$new[[i]]
    }
    if (translate$old[[i]] == x_var){
      x_var <- translate$new[[i]]
    }
    
  }
  
  
  NL <- "\\\\"
  colnames(tex_df) <- str_replace_all(colnames(tex_df),
                                      "_","\\\\textunderscore ")
  setting <- str_replace_all(setting,
                             "_","\\\\textunderscore ")
  
  setting_sp <- unlist(str_split(string = setting,pattern = "\n"))
  
  setting <- ""
  for (i in 1:length(setting_sp)){
    setting <- paste0(setting,setting_sp[[i]],collapse="")
    if (i %% 3 == 0){
      print(i)
      setting <- paste0(setting,"\\\\ ",collapse="")
    }
    
  }
  
  
  
  if ("Noise" %in% colnames(tex_df)){
    tex_df[,"Noise"] <- sapply(tex_df[,"Noise"],
                               function(x){
                                 return(paste0(100*as.numeric(x),"\\%"))
                               })
  }
  
  
  
  
  hdr <- paste(colnames(tex_df)[1:(n_settings+1)],collapse = " & ")
  hdr <- paste(hdr, paste(colnames(tex_df)[seq(n_settings+2,ncol(tex_df),2)],collapse = " & "),sep = " & ",collapse = "")
  
  
  writeLines(c(hdr,NL),fileConn)
  
  for (x_un in unique(tex_df[,x_var])){
    writeLines(c(paste0("\\midrule ",NL)),fileConn)
    df = tex_df[tex_df[,x_var] == x_un,]
    for (i in 1:nrow(df)){
      ln <- paste0(df[i,1:n_settings],collapse = " & ")
      ln <- paste0(ln," & ",df[i,x_var])
      for (j in seq(n_settings+2,ncol(df),2)){
        if (!is.na(df[i,j]) & as.numeric(df[i,j]) == max(as.numeric(df[,j]),na.rm = T)) {
          obs <- paste0(" & ","\\textbf{",df[i,j],"$\\pm$",df[i,j+1],"}")
        } else{
          
          obs <- paste0(" & ",df[i,j],"$\\pm$",df[i,j+1])
        }
        ln <- paste0(ln,obs)
      }
      ln <- paste0(ln,NL)
      ln <- str_replace_all(ln," NA ","---")
      writeLines(c(ln),fileConn)
    }
  }
  
  writeLines(c("\\end{tabular}",
               paste0("\\caption{",setting,"}"),
               "\\end{table}"),fileConn)
  
  
  close(fileConn)
  
}

#############################################################
#Translate variable names
translate <-t(as.data.frame(list(c("out_filter_FNR", "False Negative Rate"),
                                 c("out_filter_recall", "Recall"),
                                 c("out_filter_specificity", "Specificity"),
                                 c("out_filter_acc", "Filter accuracy"),
                                 c("out_filter_f1_score","F1 Score"),
                                 c("input_dataset","Dataset"),
                                 c("flt_filter","Filter"),
                                 c("alg_algorithm","Classifier"),
                                 c("out_acc_labeled","Accuracy (labeled)"),
                                 c("out_acc_unlabeled","Accuracy (unlabeled)"),
                                 
                                 c("flt_normalize_rows","row normalization"),
                                 
                                 c("out_filter_precision","Noise elimination precision"),
                                 c("flt_tuning_iter","Number of labels removed")
)
))
translate_var <-  function(y){
  if (endsWith(y,'_mean')){
    underscore_ids <- which(grepl(pattern = "_",unlist(strsplit("out_filter_er2_mean",""))))
    y <- substr(y,start=0,stop= underscore_ids[length(underscore_ids)] - 1)
  }
  
  
  for (i in seq(nrow(translate))){
    if (y == translate[i,1]){
      return(unname(translate[i,2]))
    }
  }
  return(y)
  stop(paste0("DID NOT FIND VAR NAME OF ",y))
}


####################################################
#' Creates a line plot
#'
#' @param 
#' @return the plot
linePlot <- function(df,x_var,use_sd=T, out_var="acc",
                     out_var_statistic="mean",type_var = "") {
  out_var = paste0(OUTPUT_PREFIX,out_var)
  f_cond <- Vectorize(function(x){
    return (startsWith(x,OUTPUT_PREFIX))
  })
  ignore_var = colnames(df)[f_cond(colnames(df))]
  #ignore_var <- ignore_var[!ignore_var %in% c(x_var,out_var)] 
  
  # ' 
  toStrExceptNA <- function(a,sep=";"){
    temp <- as.logical(!is.na(a[1,]))
    a <- a[1,temp]
    if (length(a)==0) return("")
    return(paste0(sapply(1:length(a),
                         function(i){
                           paste0(translate_var(colnames(a)[i]),"=",
                                  as.character(a[[1,i]]),
                                  sep)
                         }
    ),collapse=""))
    
  }
  compareNA <- function(v1,v2) {
    same <- (v1 == v2) | (is.na(v1) & is.na(v2))
    same[is.na(same)] <- FALSE
    return(same)
  }
  #b1: filters out columns that have only one value (that value could be NA)
  b1 <- !apply(df,2,function(x){
    all(sapply(x,function(y)compareNA(x[1],y)))}
  )
  #b2: filters out columns in 'ignore_var' 
  b2 <- !colnames(df) %in% c(x_var,out_var,type_var,ignore_var)
  
  
  #g: string describing the setting
  g <- ""
  if(any(b1 & b2)){
  g <- sapply(1:nrow(df[,b1 & b2]),
              function(i){
                return(toStrExceptNA(df[i,b1 & b2]))
              }
  )
 }
  b3 <- (startsWith(colnames(df),OUTPUT_PREFIX))
  
  #g_compl: string describing the non-output variables that appear
  #with a single, unique value
  g_compl <- ""
  if (any(!b1 & !b3)){
  g_compl <- sapply(1:nrow(df),
                    function(i){
                      return(toStrExceptNA(df[i,!b1 & !b3],sep=";\n"))
                    }
  )
  }
  print(g)
  
  n <- nrow(df)
  new_df <- as.tibble(data.frame(setting=factor(n),x=numeric(n),y=numeric(n)))
  new_df$Color <- as.factor(g)
  new_df$x <- sapply(df[,x_var],as.numeric)
  new_df$y <- sapply(df[,paste0(out_var,"_",out_var_statistic)],as.numeric)
  new_df$sd <- df[,paste0(out_var,"_sd")]
  new_df$y_min <- sapply(df[,paste0(out_var,"_",out_var_statistic) ] - df[,paste0(out_var,"_sd")],as.numeric)
  new_df$y_max <- sapply(df[,paste0(out_var,"_",out_var_statistic)] + df[,paste0(out_var,"_sd")],as.numeric)
  
  if(type_var == ""){
    g <- ggplot(data = new_df,aes(x,y,
                                  ymin =y_min, ymax = y_max, 
                                  colour=Color,fill=Color)) 
  } else {
    new_df$Type =  as.factor(sapply(df[type_var],function(x){paste0(translate_var(type_var),"=",x)}))
    new_df$Type = factor(new_df$Type,levels = levels(new_df$Type)[c(2,1)] )
    
    g <- ggplot(data = new_df,aes(x,y,
                                  ymin =y_min, ymax = y_max,linetype=Type,
                                  colour=Color,fill=Color)) 
  }
  g <- g + geom_line(size=1.5)
  
  
  if(use_sd){
    g <- g + geom_ribbon(alpha=0.1) 
  }
  g <- g +
    theme_light()  + 
    theme(axis.text.y = element_text(colour = 'black', size = 12), 
          axis.title.y = element_text(size = 20, 
                                      hjust = 0.5, vjust = 0.2),
          axis.title.x = element_text(size = 20, 
                                      hjust = 0.5, vjust = 0.2),
          
          legend.position=c(.9,.75),
          legend.title = element_text(color = "black", size = 16),
          legend.text = element_text(color = "black", size = 12)) + 
    theme(strip.text.y = element_text(size = 11, hjust = 0.5,
                                      vjust =    0.5, face = 'bold')) +
    scale_color_brewer(palette="Dark2") + 
    scale_fill_brewer(palette="Dark2") +
    ggtitle(paste0(g_compl[1])) + xlab(translate_var(x_var)) +
    ylab(translate_var(out_var))
  return(g)
}


#######################################################################
#Experiment Config
EXP_NAME = file.path("31_Dezembro_LDSTvsLGCLVO")
cond <- T
EXP_TYPE = 4

if (EXP_TYPE %in% c(1,2)){
  x_var = "noise_corruption_level"
  type_var = "alg_algorithm"
}else if (EXP_TYPE == 4){
  x_var = "flt_tuning_iter"
  type_var = "flt_filter"
} else{
  stop("BAD EXP_TYPE")
}
#######################################################################
#  Create table for whole dataset
for (ds in unique(df$input_dataset)) {
  for (out_var in c("CMN_rownorm_acc_unl","acc_labeled","CMN_acc","acc_unlabeled")){
    out_var_statistic = "mean"
    
    #Filter df and create joined table
    filtered_df <- dplyr::filter(df,cond & input_dataset == ds)
    if (nrow(filtered_df)==0){stop("ERROR: zero rows satisfy criteria")}
    
    #Set export folder
    export_folder = file.path(getwd(),"results","plots_R",EXP_NAME,
                              paste0('dataset=',as.character(ds)))
    if (!dir.exists(export_folder)){
      dir.create(export_folder,recursive = T)
    }
    fPath_table = file.path(export_folder,paste0(out_var,"~",x_var,"_",out_var_statistic,
                                                 "_JOINEDTABLE",".txt"))
    
    
    l <- texDf(filtered_df,x_var = x_var,type_var="", 
               out_var = out_var, out_var_statistic = out_var_statistic,
               use_sd = use_sd)
    joined_df = l[[1]]
    setting = l[[2]]
    
    #joined_df$input_labeled_percent <- sapply(joined_df$input_labeled_percent,
    #                                          function(x){paste0("Acc.(",100*x,"% labeled)")})
    if ("input_labeled_percent" %in% colnames(joined_df)){
      joined_df$input_labeled_percent <- as.factor(joined_df$input_labeled_percent )
      
      joined_df <-joined_df %>% 
        gather(variable,value,c(paste0(OUTPUT_PREFIX,out_var),sd)) %>%
        unite(temp,input_labeled_percent,variable) %>%
        spread(temp,value)
    }
    
    
    
    #Correct column names
    outvar_cols = seq(which(colnames(joined_df)==x_var)+1,ncol(joined_df),2)
    colnames(joined_df)[outvar_cols] = sapply(colnames(joined_df)[outvar_cols],
                                              function(x){
                                                val <- as.numeric(as.character(unlist(strsplit(x,"_"))[1]))
                                                return(paste0("Acc.(",val*100,"\\% labeled)"))
                                              })
    sd_cols = outvar_cols + 1
    colnames(joined_df)[sd_cols] = sapply(colnames(joined_df)[sd_cols],
                                          function(x){
                                            val <- as.numeric(as.character(unlist(strsplit(x,"_"))[1]))
                                            return(paste0("Acc. stddev(",val*100,"\\% labeled)"))
                                          })
    
    
    #Invert order
    library(stringr)
    numb_extr <- function(x){
      return(as.numeric(str_match(
        colnames(joined_df)[x],"\\((([0-9]|\\.)+)")[,2]))
    }
    outvar_cols = seq(which(colnames(joined_df)==x_var)+1,ncol(joined_df),2)
    sd_cols = outvar_cols + 1
    outvar_cols.sorted =  outvar_cols[order(numb_extr(outvar_cols),decreasing = T)]
    sd_cols.sorted =  sd_cols[order(numb_extr(outvar_cols),decreasing = T)]
    
    joined_df[outvar_cols] =  joined_df[outvar_cols.sorted]
    colnames(joined_df)[outvar_cols] = colnames(joined_df)[outvar_cols.sorted]
    
    joined_df[sd_cols] =  joined_df[sd_cols.sorted]
    colnames(joined_df)[sd_cols] = colnames(joined_df)[sd_cols.sorted]
    
    print(fPath_table)
    
    createTexTable(fPath_table,joined_df,setting,x_var=x_var)
    
  }
}

########################################################################33
#Create images
#
#
out_var_list = c("filter_f1_score","filter_precision","filter_specificity","filter_recall")
#out_var_list = c("acc_labeled","acc_unlabeled")

for (ds in unique(df$input_dataset)) {
  for (lp in unique(df$input_labeled_percent)){
    
    #Filter df and create joined table
    filtered_df <- dplyr::filter(df,cond)
    if (nrow(filtered_df)==0){stop("ERROR: zero rows satisfy criteria")}
    
    for (out_var in out_var_list){
      
      filtered_df <- dplyr::filter(df,cond & flt_mu==0.1111 & input_labeled_percent==lp)
      
      if(lp == 0.01){
        filtered_df <- dplyr::filter(filtered_df,flt_tuning_iter <= 15)
        
      }
      if (nrow(filtered_df)==0){next}
      
      ### Determine export Folder
      export_folder = file.path(getwd(),"results","plots_R",EXP_NAME,
                                paste0('dataset=',as.character(ds)),
                                paste0('labeledPercent=',as.character(lp))
      ) 
      if (!dir.exists(export_folder)){
        dir.create(export_folder,recursive = T)
      }
      ### Save lineplot to folder
      out_var = out_var
      out_var_statistic = "mean"
      use_sd = F
      fPath = file.path(export_folder,paste0(out_var,"~",x_var,"_",out_var_statistic,
                                             ",sd=",use_sd,".png"))
      fPath_table = file.path(export_folder,paste0(out_var,"~",x_var,"_",out_var_statistic,
                                                   ".txt"))
      
      
      l <- texDf(filtered_df,x_var = x_var,type_var=type_var, 
                 out_var = out_var, out_var_statistic = out_var_statistic,
                 use_sd = use_sd)
      new_df = l[[1]]
      setting = l[[2]]
      #createTexTable(fPath_table,new_df,setting,x_var=x_var)
      #print(fPath_table)
      
      
      g <- linePlot(df = filtered_df,x_var = x_var,type_var=type_var, 
                    out_var = out_var, out_var_statistic = out_var_statistic,
                    use_sd = use_sd)
      ggsave(fPath, width = 12, height = 8,dpi=150)
      plot(g)
    }
  }
}
####################################################################

latexToCSV <- function(fPath){
  
  library("data.table") # loads the smart "fast" fread function
  
  #import data into R
  data <- fread(fPath, skip=8, sep="&", data.table=F) # skips in compatible lines
  names <- read.table(fPath, skip=6, nrow=1, sep="&", stringsAsFactors=F)
  
  #remove latex symbols from header
  names<-gsub("\\\\", "", as.character(names))
  names<-gsub(" ", "", as.character(names))
  colnames(data) <- names
  
  #data[,ncol(data)]<-gsub("\\\\", "", as.character(data[,ncol(data)]))
  #data[,ncol(data)]<-gsub(" ", "", as.character(data[,ncol(data)]))
  data[,ncol(data)] <- as.character(data[,ncol(data)]) #change type as desired
  return(data)
}
export_latex <- function(x, file, outPath){
  print(xtable(data.frame(x), type="latex", file=outPath))
} 

lp <- 0.1
fPath = paste0("/home/klaus/eclipse-workspace/Quali/Overleaf/images_results/CHAPELLE_V3/",
               "dataset=Digit1/labeledPercent=",lp,"/CMN_acc~noise_corruption_level_mean.txt")

df <- latexToCSV(fPath = fPath)



#################################################################
for (sd in c(0.4,1,2,3))
{
  temp = mlbench::mlbench.2dnormals(n=1000,sd = sd)
  X = as.tibble(temp$x)
  colnames(X) = c("V1","V2")
  write.csv(X,
            paste0("gaussians_sd=",sd,"_X.csv"))
  X[,"y"] = temp$classes
  ggplot(X,aes(V1,V2,color=y)) + geom_point()
  
  write.csv(as.numeric(temp$classes),
            paste0("gaussians_sd=",sd,"_Y.csv"))
}
########################
setwd("/home/klaus/eclipse-workspace/NoisyGSSL/src/input/dataset/toy_data")
cx <- read.csv("COIL_X.csv")
cx <- cx[,-1]
cy <- read.csv("COIL_Y.csv")
cy <- cy[,-1]
coil <- cbind(cx,cy)


cx2 <- read.csv("COIL2_X.csv")
cx2 <- cx2[,-1]
cy2 <- read.csv("COIL2_Y.csv")
cy2 <- cy2[,-1]
coil2 <- cbind(cx2,cy2)



cy <- read.csv("COIL_Y.csv")