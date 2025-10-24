c=0
l=1
rm Ligand_*
rm -r Initial_LC 
for i in `cat Centroid_Distance |sort -n | uniq |  sed 's: ::g' | sed 's:,:@:g' | sed 's:\t:,:g' `
do
	name=`echo $i|cut -d"," -f1| sed 's:@:,:g'`
	value=`echo $i|cut -d"," -f2`
	
	n=`echo $i|cut -d"_" -f1|sed 's:@:,:g'`
	
	N=`echo $name|cut -d"_" -f1`
	line=`egrep  "$N"_ Centroid_Distance | wc -l`
	#echo -e "$N\t$line"
	energy=`grep "REMARK VINA RESULT:" $name.pdb | awk '{print $4}'`

	#echo -e "$name\t$value\t$energy"| awk '{print $1"\t"$2"\t"1/$2"\t"$3"\t"2.713^(-$3/0.59)}'	
	c=`expr $c + 1`
	echo "$name\t$value\t$energy" >> Ligand_"$n"_initial
	#echo "$name\t$value\t$energy"
	#echo "-----"
#	echo $c
	if [ $c -eq $line ]
	then
		#echo $i
		sum=`awk '{sum+=2.713^(-$3/0.59)}END{print sum}' Ligand_"$n"_initial`
#		echo $sum
		awk -v s=$sum '{print (2.713^(-$3/0.59))/s"\t"$3}' Ligand_"$n"_initial > tmp
#		paste Ligand_"$n" tmp | awk '{print $1"\t"100*(1/$2)*($4)"\t"$NF}' > tmp1
		#paste Ligand_"$n" tmp
		paste Ligand_"$n"_initial tmp | awk '{print $1"\t"(100-$2)"\t"$4"\t"(100-$2)*($4)"\t"$NF}' > tmp1
		mv tmp1 Ligand_"$n"
		#l1=`wc -l Ligand_"$n"_initial`
		#l2=`wc -l tmp`
		#echo -e "$name\t$l1\t$l2\t$n"
		c=0
		l=`expr $l + 1`
	fi
done
mkdir Initial_LC 
mv  Ligand_*_initial Initial_LC
echo -e "Sno,Name,Score,Distance,Boltzman,Energy,Categ" > LigandScore_LC
sn=1

for i in `ls Ligand_*`
do  
	n=`echo $i|sed 's:,:-:g'`
	line=`awk '{if($2>=95) print $0}' $i | wc -l|cut -d" " -f1` 
	score=`awk '{sum+=$4;sum2+=$2;sum3+=$3;sum4+=$5}END{print sum/10","sum2/10","sum3/10","sum4/10}' $i`
	echo "$sn,$n,$score,$line"| sed '/initial/d'|sed 's:Ligand_::g' >> LigandScore_LC
	sn=`expr $sn + 1`
done  

