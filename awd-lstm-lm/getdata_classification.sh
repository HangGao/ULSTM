echo "- Downloading Glove"
mkdir glove
cd glove
wget --continue http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -o glove.840B.300d.zip
rm *.zip
cd ..

echo "- Downloading Yelp review data at https://www.yelp.com/dataset"
cd ..
