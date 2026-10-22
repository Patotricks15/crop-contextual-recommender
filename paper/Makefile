.PHONY: all clean docker

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

docker:
	docker run --rm -v "$(CURDIR)":/work -w /work texlive/texlive:latest \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
