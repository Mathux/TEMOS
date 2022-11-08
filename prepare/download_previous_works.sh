#!/bin/bash
echo -e "The generation of previous work (on the test set of KIT-ML) will stored in the 'previous_work' folder\n"
gdown "https://drive.google.com/uc?id=1ZstP5VOyyKfqgec-G69SR4HEIvwWM5uc"

echo -e "Please check that the md5sum is: 6b753eec02b432448b70c72c131f0ae8"
echo -e "+ md5sum previous_work.tgz"
md5sum previous_work.tgz

echo -e "If it is not, please rerun this script"

sleep 5
tar xfzv previous_work.tgz

echo -e "Cleaning\n"
rm previous_work.tgz

echo -e "Downloading done!"
