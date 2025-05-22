.PHONY: all clean distclean

TEX := pdflatex
BIB := bibtex
BST := makeindex
MAIN := report
PUT := texput
TEX_SRC := $(MAIN).tex
BIB_SRC := $(MAIN).bib
BST_SRC := $(MAIN).idx
TEX_FLAGS := -interaction=nonstopmode

SRC_DIR := .
OBJ_DIR := .
BIN_DIR := bin
DOC_DIR := doc
FIG_DIR := assets

TEX_FILES := $(wildcard $(SRC_DIR)/*.tex)
BIB_FILES := $(wildcard $(SRC_DIR)/*.bib)
BST_FILES := $(wildcard $(SRC_DIR)/*.idx)
FIG_FILES := $(wildcard $(FIG_DIR)/*.ttf)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.tex,$(OBJ_DIR)/%.pdf,$(TEX_FILES))

# Main build target
.PHONY all: $(OBJ_DIR)/$(MAIN).pdf
	@$(TEX) < report.tex | grep -v pdfT | sed "s/$(PUT)/$(MAIN)/g"
	@echo "Build complete!" && rm texp*

# rd to compile TeX files into PDF
$(OBJ_DIR)/$(MAIN).pdf: $(SRC_DIR)/report.tex $(BIB_FILES) $(BST_FILES) $(FIG_FILES) | $(OBJ_DIR)
	@echo "Compiling $<..."
	@cp $(FIG_FILES) $(MAIN).pdf
	@$(TEX) < report.tex | grep -v pdfT | sed "s/$(PUT)/$(MAIN)/g"
#	@cp $(OBJ_DIR)/$(MAIN).pdf $(SRC_DIR) &> $(NL)

#$(OBJ_DIR) $(BIN_DIR) $(DOC_DIR):
#	@mkdir -p $@

clean:
	@echo "Cleaning up..."
	@$(RM) -r $(OBJ_DIR) $(BIN_DIR) $(DOC_DIR)

distclean: clean
	@echo "Distcleaning..."
	@$(RM) -r .dep

run: all
	@echo "Document $(MAIN).pdf is ready for viewing."

.DEFAULT: all

