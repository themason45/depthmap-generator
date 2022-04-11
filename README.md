

### Embree installation

Brew install embree:

```shell
brew install embree
```
Find the embree install location using:

```shell
brew info embree
```

To install embree, clone the python wrapper library found here: [https://github.com/sampotter/python-embree.git](https://github.com/sampotter/python-embree.git)

Then run:

```shell
python setup.py build_ext -I{INSTALL_LOCATION}/include -L{INSTALL_LOCATION}/lib/
python setup.py install
```

Substituting {INSTALL_LOCATION} for the location found with `brew info embree`