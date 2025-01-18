#copy files to cluster uy

rsync -avz  -e "ssh -p 10022" /data/maestria/resultados/deep_cstrd/ henry.marichal@cluster.uy:/clusteruy/home/henry.marichal/datasets/deep_cstrd
