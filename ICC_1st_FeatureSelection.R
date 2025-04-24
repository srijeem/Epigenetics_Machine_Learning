### ICC (Intra Class Correlation)
### Feature Selection Step to remove unstable probes 
### from both timepoints (Timepoint 1 and Timepoint 2)


# Set working directory
getwd()

# Set the directory (Anonymized path)
workdirectory = "path_to_project_folder/Scripts"   

setwd(workdirectory)

# Load necessary libraries
library(dplyr)
library(BiocManager)
library(affy)
library(IlluminaHumanMethylationEPICmanifest)
library(limma)
library(minfi)
library(minfiData)
library(preprocessCore)
library(tidyverse)
library(irr)
library(tidyr)

# Load the phenotype file (Anonymized path)  ## This to bring in metadata about the samples in the study
phenofile <- read.csv("path_to_project_folder/Pheno/Phenofile.csv")
head(phenofile)

# Load cross-hybridized probes (Anonymized path) ## This is to filter out cross-reactive probes as a quality control step
cross = read.csv("path_to_project_folder/Normalisation/CrossHybridizedProbes.csv", stringsAsFactors = FALSE)

# Set options
options(stringsAsFactors = F)

# Load paths for idats, phenofile, and cross-hybridized probes
idatpath = "path_to_project_folder/data/idats"
crosspath = "path_to_project_folder/Normalisation/CrossHybridizedProbes.csv"
phenopath = "path_to_project_folder/Pheno/Phenofile.csv"

# Read the phenofile
targets <- read.csv(phenopath, sep = ",", header = T)
head(targets)
BasePath = idatpath

# Convert relevant columns to factors
targets$DNA_BaselineFollowUp = as.factor(targets$Longitudinal_DNA_Data)
targets$R1_Hypertension = as.factor(targets$T1_HTN)
targets$R2_Hypertension = as.factor(targets$T2_HTN)
targets$Site = as.factor(targets$Site)

# Read the training target file (Anonymized path)
traintargets = read.table("path_to_project_folder/Scripts/training90")
head(traintargets)

# Convert relevant columns to factors
traintargets$DNA_BaselineFollowUp = as.factor(traintargets$Longitudinal_DNA_Data)
traintargets$R1_Hypertension = as.factor(traintargets$T1_HTN)
traintargets$R2_Hypertension = as.factor(traintargets$T2_HTN)
traintargets$Site = as.factor(traintargets$Site)

# Check the number of rows and columns in the training data
num_entries <- nrow(traintargets)
print(num_entries)

num_cols <- ncol(traintargets)
print(num_cols)

# Read the beta values for training data (Anonymized path)
beta_train = read.table("path_to_project_folder/Scripts/RenormalizedBetaTrain.txt")

######################################################################################################################################################################
######################################################################################################################################################################

### ICC Analysis

# Create a dataframe with Timepoint, SampleID, and beta values
Timepoint = traintargets$Longitudinal_DNA_Data
UniqueID = traintargets$U_ID
R2 = traintargets$T2_HTN

ltsdata <- data.frame(
  UniqueID,
  Timepoint,
  t(beta_train)
)
head(ltsdata)

# Compute ICC results for each column in ltsdata from the 3rd column onwards
icc_results <- foreach(i = 3:ncol(ltsdata), .combine = "rbind") %do% 
  {
    # Select current column along with first two columns
    iccscore <- ltsdata[,c(1,2,i)] %>% 
      tidyr::pivot_wider(names_from = Timepoint, values_from = 3) %>% 
      dplyr::select(2,3) %>% 
      irr::icc(., model = "twoway", type = "consistency")

    # Return a data frame with ICC results
    data.frame(icc = iccscore$value,
               icc_lbound = iccscore$lbound,
               icc_ubound = iccscore$ubound,
               fvalue = iccscore$Fvalue,
               pvalue = iccscore$p.value,
               row.names = colnames(ltsdata)[i])
  }

# Adjust p-values using Benjamini-Hochberg method
icc_results$padj <- p.adjust(icc_results$pvalue, method = "BH")
head(icc_results)

# Write the ICC scores to a CSV file (Anonymized path)
write.csv(icc_results, "path_to_project_folder/Scripts/icc_score_90_training_split.csv")

#### END OF ICC ANALYSIS ###
