#!/bin/bash

for file in hindi/comparable_hindi_articles/hi_*;
do 
	# echo "$file";
	IFS=/ read -a foo <<< "$file"
	# echo "${foo[2]}"
	stemmedFile="hindi/comparable_hindi_stemmed/""${foo[2]}"
	echo "$stemmedFile"
	shallow_parser_hin $file $stemmedFile 
	
done








