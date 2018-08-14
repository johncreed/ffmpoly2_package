#! /bin/bash

sort -k 3 -g *.merge | xargs -d '\n' -I arg0 grep "arg0" *.merge | less
