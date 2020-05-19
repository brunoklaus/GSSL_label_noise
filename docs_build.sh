cd ./sphinx_docs
sphinx-apidoc -f -o ./_modules ../src
sphinx-build   . ../docs
sphinx-build -b coverage . ../docs
cat _build/python.txt
#xdg-open _build/index.html


