cd data

echo "- Downloading Glove"
mkdir glove
cd glove
wget --continue http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -o glove.840B.300d.zip
rm *.zip
cd ..

echo "- Downloading stanford Sentiment Treebank"
mkdir sts
cd sts
wget --continue http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip -o stanfordSentimentTreebank.zip
mv stanfordSentimentTreebank/* .
rm -rf stanfordSentimentTreebank
rm *.zip
cd ..
