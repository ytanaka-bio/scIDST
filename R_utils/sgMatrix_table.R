sgMatrix_table <- function(seurat,dir,...){
	       library(Seurat)
	       library(Matrix)
	       library(R.utils)

	       matfile = paste(dir, "/matrix.mtx",sep="")
	       f_file  = paste(dir, "/features.tsv",sep="")
	       b_file  = paste(dir, "/barcodes.tsv",sep="")

	       dir.create(dir)

	       writeMM(seurat@assays$RNA@counts,file = matfile)
	       f_data = data.frame(rownames(seurat@assays$RNA@counts))
	       rownames(f_data) <- f_data[,1]
	       write.table(f_data,f_file,sep="\t",quote=F,row.names=T,col.names=F)
	       b_data = data.frame(colnames(seurat@assays$RNA@counts))
	       write.table(b_data,b_file,sep="\t",quote=F,row.names=F,col.names=F)

	       gzip(matfile)
	       gzip(f_file)
	       gzip(b_file)
}
