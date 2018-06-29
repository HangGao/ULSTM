mkdir -p data
mkdir -p save
mkdir -p corpus
cd data

echo "- Downloading Glove"
mkdir -p glove
cd glove
wget --continue http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -o glove.840B.300d.zip
rm *.zip
cd ..

echo "- Downloading stanford Sentiment Treebank"
mkdir -p sts
cd sts
wget --continue http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip -o stanfordSentimentTreebank.zip
mv stanfordSentimentTreebank/* .
rm -rf stanfordSentimentTreebank
rm *.zip
cd ..
