library(Seurat)
library(SPARK)

# Read sparse matrix from h5 file into Seurat object
adata <- Load10X_Spatial("C:/Users/demeter_turos/PycharmProjects/chrysalis/data/V1_Human_Lymph_Node/")

adata <- PercentageFeatureSet(adata, "^mt-", col.name = "percent_mito")

adata <- subset(adata, subset = nCount_Spatial > 6000)
gene_counts <- rowSums(GetAssayData(adata, slot = "counts") > 0)
keep_genes <- names(gene_counts[gene_counts >= 10])
adata <- subset(adata, features = keep_genes)

expression_data <- as.matrix(adata@assays$Spatial@data)
dim(expression_data)
locs <- GetTissueCoordinates(adata)

sparkX <- sparkx(expression_data,locs,numCores=1,option="mixture")

head(sparkX$res_mtest)
write.csv(sparkX$res_mtest,  "C:/Users/demeter_turos/PycharmProjects/chrysalis/dev/benchmarks/
                              fig_1_lymph_node_cell2loc/spark.csv")
