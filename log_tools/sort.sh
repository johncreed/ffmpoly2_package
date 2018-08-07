#! /bin/bash

sort -k $1 -g *.merge | xargs -I arg0 grep "arg0" *.merge | less
