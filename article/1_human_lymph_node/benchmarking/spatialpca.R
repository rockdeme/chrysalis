library(SpatialPCA)
library(ggplot2)
library(Matrix)
library(Seurat)


# HUMAN LYMPH NODE
# Read sparse matrix from h5 file into Seurat object

start_time <- Sys.time()

adata <- Load10X_Spatial("C:/Users/demeter_turos/PycharmProjects/chrysalis/data/V1_Human_Lymph_Node/",)

adata <- PercentageFeatureSet(adata, "^mt-", col.name = "percent_mito")

adata <- subset(adata, subset = nCount_Spatial > 6000)
gene_counts <- rowSums(GetAssayData(adata, slot = "counts") > 0)
keep_genes <- names(gene_counts[gene_counts >= 10])
adata <- subset(adata, features = keep_genes)

xy_coords <- adata@images$slice1@coordinates
xy_coords <- xy_coords[c('imagerow', 'imagecol')]
colnames(xy_coords) <- c('x_coord', 'y_coord')

count_sub <- adata@assays$Spatial@data
print(dim(count_sub))  # The count matrix
xy_coords <- as.matrix(xy_coords)
rownames(xy_coords) <- colnames(count_sub) # the rownames of location should match with the colnames of count matrix
LIBD <- CreateSpatialPCAObject(counts=count_sub, location=xy_coords, project="SpatialPCA", gene.type="spatial",
                               sparkversion="spark", numCores_spark=5, gene.number=3000, customGenelist=NULL,
                               min.loctions=20, min.features=20)

LIBD <- SpatialPCA_buildKernel(LIBD, kerneltype="gaussian", bandwidthtype="SJ", bandwidth.set.by.user=NULL)
LIBD <- SpatialPCA_EstimateLoading(LIBD,fast=FALSE,SpatialPCnum=20)
LIBD <- SpatialPCA_SpatialPCs(LIBD, fast=FALSE)


saveRDS(LIBD, file = "C:/Users/demeter_turos/PycharmProjects/chrysalis/data/cell2loc_human_lymph_node/spatialpca/libd.rds")

LIBD <- readRDS(file = "C:/Users/demeter_turos/PycharmProjects/chrysalis/data/cell2loc_human_lymph_node/spatialpca/libd.rds")

write.csv(LIBD@SpatialPCs,  "C:/Users/demeter_turos/PycharmProjects/chrysalis/data/cell2loc_human_lymph_node/spatialpca/spatial_pcs.csv")

clusterlabel <- walktrap_clustering(clusternum=8, latent_dat=LIBD@SpatialPCs, knearest=70)
# here for all 12 samples in LIBD, we set the same k nearest number in walktrap_clustering to be 70.
# for other Visium or ST data, the user can also set k nearest number as
# round(sqrt(dim(SpatialPCAobject@SpatialPCs)[2])) by default.
clusterlabel_refine <- refine_cluster_10x(clusterlabels=clusterlabel, location=LIBD@location, shape="hexagon")

end_time <- Sys.time()

elapsed_time <- end_time - start_time
print(elapsed_time)

cbp<-c('#db5f57', '#dbc257', '#91db57', '#57db80', '#57d3db', '#5770db', '#a157db', '#db57b2')
plot_cluster(location=xy_coords,clusterlabel=clusterlabel_refine, pointsize=1.5,
             title_in=paste0("SpatialPCA"), color_in=cbp)
