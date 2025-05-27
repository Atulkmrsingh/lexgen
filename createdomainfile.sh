
domains=( "<broadcast>" "<civil>" "<comp>" "<geo>" "<psycho>" "<maths>")
for j in "${domains[@]}" 
do 
    langs=( "gu" "hi"  "kn" "mr")
    for i in "${langs[@]}"
        touch "../exp1/devtestexp2/$j/en-$i/test.en"
        touch "../exp1/devtestexp2/$j/en-$i/outtest.$i"
        touch "../exp1/devtestexp2/$j/en-$i/test.$i"
        touch "../exp1/devtestexp2/$j/en-$i/test.dom"
done
