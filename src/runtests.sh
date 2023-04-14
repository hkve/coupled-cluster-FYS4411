cd ..

files=`ls ./src/tests/*.py`

for file in $files
do
	file=${file#"./"}
	file=${file%".py"}
	file=$(echo "$file" | sed 's/\//\./g')

	python3 -m unittest ${file}	
done

# py -m unittest src.tests.
