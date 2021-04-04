# Welcome to PolyMath!

PolyMath is a framework comprised of both a high-level language and an embedded Python language for compilation on heterogenous hardware.

This document will help you get up and running.  

### Step 0: Check prerequisites
The following dependencies must be met by your system:
  * python >= 3.7 (For [PEP 560](https://www.python.org/dev/peps/pep-0560/) support)


### Step 1: Clone the PolyMath source code
  ```console
  $ git clone https://github.com/he-actlab/polymath
  $ cd polymath
  ```


### Step 2: Create a [Python virtualenv](https://docs.python.org/3/tutorial/venv.html)
Note: You may choose to skip this step if you are doing a system-wide install for multiple users.
      Please DO NOT skip this step if you are installing for personal use and/or you are a developer.
```console
$ python -m venv general
$ source general/bin/activate
$ python -m pip install pip --upgrade
```

### Step 3: Install PolyMath
If you already have a working installation of Python 3.7 or Python 3.8, the easiest way to install GeneSys is:
```console
$ pip install -e .
```

### Step 4: Run an example
You can look at the examples in the `examples/` directory to see how the PolyMath language works. 



