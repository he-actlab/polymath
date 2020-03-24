..

PolyMath: A Programming Stack for Cross-Domain Acceleration.
============================================================

PolyMath is a cross-domain stack comprised of a high-level cross-domain language, a multi-granular intermediate representation, and a compilation infrastructure that provides the expressiveness to bring together Robotics, Digital Signal Processing, Deep Learning, and Data Analytics, while targeting a variety of accelerators.

This repository implements the following:

* A Python-Embedded language
* A compiler for the embedded language which converts high-level programs to an mg-DFG
* Serialization and de-serialization to protobuf for the mg-DFG to allow portability of programs
* A compiler pass infrastructure for defining optimization transformations or analyses on mg-DFGs
* Lowering compiler passes which transform the mg-DFG to different types of accelerator. Currently, `TVM <https://github.com/apache/incubator-tvm/>`_ and `TABLA <http://act-lab.org/artifacts/tabla/>`_ lowering passes are provided, but similar passes can be extended to other types of accelerator.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   polymath


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`