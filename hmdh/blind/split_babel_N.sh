date > StartTime
echo "Last Centroid Distance Moved to Blind Docking"
rm Centroid_Distance
com=`./Pocket_Centroid.exe pocket.pdb`
rm *_ligand_*.pdb
 rm *_ligand_*.pdbqt
for i in `ls *log |sed 's:.log:_out.pdbqt:g'`
do
        name=`echo $i|cut -d"." -f1`
		echo "*****$name******" 
        vina_split --input $i

        for p in `ls "$name"*"lig"*".pdbqt"`
        do
                #echo $p
                name2=`echo $p | cut -d"." -f1`
                echo $name2
                obabel -ipdbqt "$p" -opdb -O$name2".pdb"
                #val=`./Pair-Wise-Distance.exe pocket.pdb "$name2".pdb`
                val=`./LigandCentroid-Distance.exe  "$name2".pdb $com |cut -d"=" -f2`
                echo "$name2\t$val" >> Centroid_Distance
        done

done
date > EndTime
