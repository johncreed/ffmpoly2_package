all: output/ffmpoly2.pdf

output/%.pdf: %.tex
	mkdir -p output
	latexmk -recorder -pdf -output-directory=output -f -pdflatex='pdflatex  -shell-escape %O %S' $<
clean:
	rm -rf output
	find figures -name \*.pdf | xargs rm

bib:
	scp r03944049@linux1.csie.ntu.edu.tw:~cjlin/latex/sdp.bib .
