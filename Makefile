.PHONY: info
info:
	@echo "Build with uv + taskipy; see docs/CONTRIBUTING.md."
	@echo
	@echo "- type 'make docs' to build the HTML documentation"
	@echo "- type 'make clean' to remove build artifacts"
	@echo "- type 'make distclean' to also remove packaged archives"
	@echo
	@echo "Packaging is done by uv, not by this Makefile:"
	@echo "  uv build                      # sdist + wheel"
	@echo "  uv run task build-pkg-macos   # or build-pkg-linux / build-pkg-win"


# build the HTML documentation (does not need network access)
.PHONY: docs
docs:
	$(MAKE) -C docs html
	@echo "docs/_build/html/index.html"


# remove all package build artifacts
.PHONY: clean
clean:
	uv run task clean
	-rm -rf build dist *.tar.gz *.egg*


# remove all package build artifacts and packaged archives
.PHONY: distclean
distclean: clean
	-rm -f *.deb *.rpm
	$(MAKE) -C docs clean
