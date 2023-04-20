Color="\033[0;33m" # Yellow is an ok color  
Color_Off="\033[0m"
cd ..

files=`ls ./src/tests/*.py`

for file in $files
do
	file=${file#"./"}
	file=${file%".py"}
	file=$(echo "$file" | sed 's/\//\./g')

	echo "${Color}${file}${Color_Off}"
	python3 -m unittest ${file}
	echo "\n"	
done