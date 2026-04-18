OV_patientDNA_sampleList.txt : list of samples analysed for CNAs

## CNA_tables:
Copy number tables for each patient separately, extracted from QDNAseq and saved as RData files. Each RData contains three R dataframes:
- bins.df : genomic bins used for counting read depth, characteristics supplied by QDNAseq
- cn.df : bin-wise copy number values, normalised to 1
- seg.df : segmented copy number values, normalised to 1

## liquidCNA results:
Estimates_xxx.RData: Output of the liquidCNA algorithm for each patient. Each RData contains the following R objects:
- pHat.df : estimated purity values
- seg.df.corr : dataframe with purity-corrected segmented copy number values
- seg.av.corr : dataframe with purity-corrected copy number values per segment, each row is a separate segment after filtering out short segments
- seg.plot : dataframe with purity-corrected delta copy number values (compared to baseline sample) per segment, chromosomal location of each segment and annotation of whether segments are clonal (filtered=TRUE) or subclonal/uncertain (order=TRUE/FALSE)
- fitInfo : list output by the subclonal segment ordering/fitting step of liquidCNA, used for diagnostic purposes 
- final.medians: dataframe containing final estimates of subclonal ratio

Subclonal_ratio_estimates.extended.txt: table containing all liquidCNA results and additional CA125 values

Drivers_subclonalCNA.txt: table listing driver genes located in identified genomic segments with subclonal CNA