.. _developers_guide:

=================
Developer's Guide
=================

------------------
How to contribute?
------------------

All contributions, even the smallest, are welcome!
There are many ways to contribute to FESTIM.

Be active in the community by:

* Reporting a bug
* Proposing a feature

And/or contribute to the source code by:

* Improving the documentation
* Fixing bugs
* Implementing new features

------------------------
Contributing to the code
------------------------

For complete information on contributions with GitHub see this guide on `GitHub <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_.

.. tip::

   If you're a beginner, look for `issues tagged with "Good first issue" <https://github.com/festim-dev/FESTIM/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_.

   These issues are usually relatively easy to tackle and perfect for newcomers.

#. `Fork the repository <https://github.com/festim-dev/FESTIM/fork>`_

   By forking the repository, you create a copy where you can safely make changes.

#. Clone your fork

   .. code-block:: bash

      git clone https://github.com/[your_username]/FESTIM

   Remember to replace ``[your_username]`` with your GitHub username.

#. Make your changes

   Commit your changes locally and push them to your fork.

   .. code-block:: bash

      git add [modified files]
      git commit -m "Your commit message"
      git push

#. Test your code

   If you are adding new features or fixing bugs, it is important to test your code.
   See :ref:`Test suite` for more information.

#. Format your code using `Black <https://github.com/psf/black>`_.

   The source code of FESTIM is formated with the Black code formatter. Using of a unified code style simplifies the code review
   and increases its readability. See :ref:`Code formatting` for more information.

#. Optional: Build the documentation

   You may want to build the documentation to see if your changes are correctly reflected or if you are updating the docs.
   See :ref:`Documentation guide` for more information.

#. `Open a PR <https://github.com/festim-dev/FESTIM/compare>`_

#. Wait for a :ref:`maintainer<Maintainers>` to review your PR

   Before merging your changes, they have to be reviewed. We ensure the changes don't break anything during the review and eventually propose/request improvements.
   The time before the review will depend on the maintainers' availability.

#. When everything is in order, the maintainers will merge your PR!

-----------
Maintainers
-----------

The maintainers are the people who have the right to merge PRs to the repository.
They consist of the following individuals:

- Remi Delaporte-Mathurin (`@RemDelaporteMathurin <https://github.com/RemDelaporteMathurin>`_)
- James Dark (`@jhdark <https://github.com/jhdark>`_)
- Vladimir Kulagin (`@KulaginVladimir <https://github.com/KulaginVladimir>`_)

The project lead is Remi Delaporte-Mathurin.

----------
Test suite
----------

FESTIM uses continuous integration (CI) to ensure code quality and eliminate as many bugs as possible.

In a nutshell, every time a change is pushed to the repository (or to a PR), a suite of tests is automatically triggered.
This is to make sure the changes don't break existing functionalities.
It is also very useful to catch bugs that developers could have missed.
Click `here <https://www.atlassian.com/continuous-delivery/continuous-integration>`_ for more information on CI.

All the tests can be found in the `test folder <https://github.com/festim-dev/FESTIM/tree/fenicsx/test>`_ at the root of the FESTIM repository.

You need to have the right dependencies installed to test your code (see :ref:`installation<Installation>`).

.. note::

   Make sure to install ``pytest`` to run the test suite locally:

   .. code-block:: bash

      pip install pytest

   And then run the tests using:

   .. code-block:: bash

      pytest test/
   
Whenever contributors open a PR, **the tests must pass** in order for the PR to be merged in.

In some cases, new tests need to be written to account for more use cases or to catch bugs that weren't previously caught.

---------
Debugging
---------

When you find a bug in the code, there are several steps to follow to make things easier for maintainers.

#. | `Raise an issue <https://github.com/festim-dev/FESTIM/issues/new/choose>`_
   |
   | This is important to keep track of things.
   | The issue is a place to talk about the bug, troubleshoot users and sometimes find workarounds.
   | It also greatly helps maintainers find the origin of the bug to fix it faster.

#. | Write a test
   | To make the test suite more robust, first write a test that catches the bug.
   | This may appear useless, but it will help the future contributors by alerting them if they reproduce this error.
   | It will also be useful to prove your fix is effective.

#. Make your changes and open a PR.

--------------------------
Implementing a new feature
--------------------------

#. | `Raise an issue <https://github.com/festim-dev/FESTIM/issues/new/choose>`_
   |
   | Before spending time implementing a new great feature, it is better to open an issue first to discuss with the maintainers.
   | For all you know, someone is already working at implementing it and all your time would be spent for nothing.
   | 
   | It is also beneficial to discuss with the community on how this new feature would be used.

#. :ref:`Make your changes<contributing to the code>`. Don't forget to :ref:`adapt the documentation <Documentation guide>` if necessary.

#. Write a test to test your feature

#. Open a PR


----------------
Code formatting
----------------

Before merging your PR, the modified scripts should be formatted to maintain the consistency of the coding style. FESTIM is formatted using 
`Black <https://github.com/psf/black>`_. To install Black, run the following command:

.. code-block:: bash

   pip install black

After the installation, you can format a file using:

.. code-block:: bash

   black my_script.py

Alternatively, you can format all files in the current directory with:

.. code-block:: bash

   black .


If you use Visual Studio Code, you can install the `extension <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_ 
with support for the Black formatter. Then, you can set Black as a default formatter for python and enable formatting "on save" for your code. 

-------------------
Documentation guide
-------------------

The documentation is a crucial part of the project. It is the first thing users will see when they want to use FESTIM.
It is important to keep it up to date and clear.

The documentation is written in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ and is located in the `docs folder <https://github.com/festim-dev/FESTIM/tree/fenicsx/docs>`_ at the root of the FESTIM repository.

The documentation is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_.

To build the documentation locally, you can use the following command:

.. code-block:: bash

   cd docs/source
   make html

This will generate the documentation in the `docs/source/_build/html` folder.
You can then open the `index.html` file in your browser to see the documentation.
To remove everything and start from scratch, you can use the following command:

.. code-block:: bash

   make clean

Alternatively, you can use the following command to build the documentation in one line:

.. code-block:: bash

   cd docs
   sphinx-build -b html source build

.. note::

   Make sure to have the right dependencies installed. You can create a new conda environment with the following command:

   .. code-block:: bash
      
      conda env create -f docs/environment.yml
   
   This will create a new environment called `festim2-docs` with all the necessary dependencies.
   Activate it using:

   .. code-block:: bash

      conda activate festim2-docs

The documentation is hosted on `Read the Docs <https://readthedocs.org/>`_ and is automatically updated when a new commit is pushed to the repository or to a Pull Request.

.. note::

   The documentation is built using the `sphinx_book_theme <https://sphinx-book-theme.readthedocs.io/en/latest/>`_ theme.

When contributing to the documentation, make sure to:

#. Write clear and concise documentation
#. Use the right syntax
#. Update the documentation when new features are added
#. Test the documentation using:

.. code-block:: bash

   cd docs/source
   make doctest

or using: 

.. code-block:: bash

   cd docs
   sphinx-build -b doctest source build