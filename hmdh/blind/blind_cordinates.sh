#!/bin/bash
#ls *.mol2 > Listmol2
#sed 's/mol2/pdbqt/' Listmol2 > List

d=$(ls *_pregrid.pdbqt)

./Pocket_Centroid.EXE $d  > aaa

a=$(awk '{print $1}' aaa)
b=$(awk '{print $2}' aaa)	
c=$(awk '{print $3}' aaa)	
echo $a $b $c


echo 'receptor = '$d'


center_x = '$a'
center_y = '$b'
center_z = '$c'

size_x = 42
size_y = 42
size_z = 42

num_modes = 10
energy_range = 4
exhaustiveness = 100' > conf.txt

