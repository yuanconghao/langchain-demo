#!/bin/sh

for i in `ls`; do mv -f $i `echo "news_"$i`; done
