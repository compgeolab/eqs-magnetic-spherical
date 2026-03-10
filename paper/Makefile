# Rules for compiling the PDF from the LaTeX sources and displaying the output

### Documents to build
PDF = preprint.pdf dissertation.pdf manuscript.pdf
#PDF = preprint.pdf manuscript.pdf
### File Types (for dependencies)
TEX = $(filter-out $(PDF:.pdf=.tex), $(wildcard *.tex))
BIB = $(wildcard *.bib)
FIG = $(wildcard figures/*)

preprint: preprint.pdf

manuscript: manuscript.pdf

dissertation: dissertation.pdf

all: $(PDF)

%.pdf: %.tex $(TEX) $(BIB) $(FIG)
	tectonic -X compile $<

show: $(PDF)
	for fname in $^ ; do \
		xdg-open $$fname ; \
	done

wordcount:
	@echo -n "Main text + abstract: "
	@echo $$((`detex content.tex | wc --words` + `detex abstract.tex | wc --words`))

clean:
	rm -f $(PDF)
