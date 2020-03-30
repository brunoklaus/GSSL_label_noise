cd ./docs
sphinx-apidoc -f -o ./_modules ../src
sphinx-build   . _build
sphinx-build -b coverage . _build
cat _build/python.txt
#xdg-open _build/index.html


