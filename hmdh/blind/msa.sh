date > time.txt
rm *.log
for i in `cat List_hmdh`
do
	name=`echo $i|cut -d"." -f1`
	mpirun -np 1 vina --config conf.txt --ligand $i  --log "$name".log --out "$name"_out.pdbqt 
done
	
date >> time.txt
