cd ..

files=`ls ./src/tests/*.py`

for file in $files
do
	echo $file
done
# py -m unittest src.tests.
