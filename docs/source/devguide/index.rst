.. _developers_guide:

=================
Contributing
=================

All contributions are welcome, no matter how small.
Whether you are fixing a typo, reporting a bug, or implementing a new feature, your help is appreciated.

------------------
Ways to contribute
------------------

**In the community:**

* `Report a bug <https://github.com/festim-dev/FESTIM/issues/new/choose>`_
* `Propose a feature <https://github.com/festim-dev/FESTIM/issues/new/choose>`_

**In the codebase:**

* Improve the documentation
* Fix bugs
* Implement new features

------------------------
Contributing to the code
------------------------

For a general overview of contributing to GitHub projects, see the
`GitHub contributing guide <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_.

.. tip::

   If you are new to the project, look for
   `issues tagged "good first issue" <https://github.com/festim-dev/FESTIM/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_.
   These are typically straightforward and a great way to get started.

#. `Fork the repository <https://github.com/festim-dev/FESTIM/fork>`_

   Forking creates a personal copy of the repository where you can freely make changes.

#. Clone your fork and add the upstream remote

   .. code-block:: bash

      git clone https://github.com/[your_username]/FESTIM
      cd FESTIM
      git remote add upstream https://github.com/festim-dev/FESTIM

   Replace ``[your_username]`` with your GitHub username.
   The upstream remote lets you pull in changes from the main repository to keep your fork up to date:

   .. code-block:: bash

      git fetch upstream
      git merge upstream/main

#. Create a branch

   Always work on a dedicated branch rather than directly on ``main``:

   .. code-block:: bash

      git checkout -b my-feature-branch

#. Set up your development environment

   Create a dedicated conda environment and install FESTIM from conda-forge.
   This pulls in all required dependencies, including FEniCSx:

   .. code-block:: bash

      conda create -n festim-dev
      conda activate festim-dev
      conda install -c conda-forge festim

   Then uninstall the conda-managed FESTIM package and replace it with your
   local clone in editable mode, so that any changes you make are picked up
   immediately without reinstalling:

   .. code-block:: bash

      pip uninstall festim
      pip install -e .

   .. note::

      The ``-e`` flag installs the package in *editable* mode.
      Python will import directly from your local source tree,
      meaning you do not need to reinstall after each change.

#. Make your changes

   Commit your changes with a clear, descriptive message:

   .. code-block:: bash

      git add [modified files]
      git commit -m "Short description of the change"
      git push origin my-feature-branch

#. Test your code

   Before opening a PR, run the test suite locally to make sure nothing is broken.
   See :ref:`Test suite` for more information.

#. Format your code

   FESTIM uses `Black <https://github.com/psf/black>`_ for consistent code formatting.
   See :ref:`Code formatting` for more information.

#. Optional: build the documentation

   If you modified or added documentation, build it locally to verify it renders correctly.
   See :ref:`Documentation guide` for more information.

#. `Open a PR <https://github.com/festim-dev/FESTIM/compare>`_

   Include a clear description of what the PR does and reference any related issues
   (e.g. ``Closes #123``).

#. Wait for a :ref:`maintainer <Maintainers>` to review your PR

   Maintainers will review your changes to ensure correctness and code quality, and may
   request further modifications. Review time depends on maintainer availability.

#. Once approved, a maintainer will merge your PR. Congratulations!

-----------
Maintainers
-----------

Maintainers are the people with merge rights on the repository:

- Remi Delaporte-Mathurin (`@RemDelaporteMathurin <https://github.com/RemDelaporteMathurin>`_) -- project lead
- James Dark (`@jhdark <https://github.com/jhdark>`_)
- Vladimir Kulagin (`@KulaginVladimir <https://github.com/KulaginVladimir>`_)

----------
Test suite
----------

FESTIM uses continuous integration (CI) to maintain code quality.
Every push to the repository or a pull request triggers the test suite automatically,
catching regressions and bugs early.
See `Atlassian's CI guide <https://www.atlassian.com/continuous-delivery/continuous-integration>`_
for a general introduction to CI.

All tests live in the `test folder <https://github.com/festim-dev/FESTIM/tree/fenicsx/test>`_
at the root of the repository.

.. note::

   Install ``pytest`` if you haven't already:

   .. code-block:: bash

      pip install pytest

   Then run the full test suite from the project root:

   .. code-block:: bash

      pytest test/

**The tests must pass before a PR can be merged.**

When fixing a bug or adding a feature, please add or update tests to cover the new behaviour.

---------
Debugging
---------

When you find a bug, follow these steps to make things easier for maintainers:

#. `Raise an issue <https://github.com/festim-dev/FESTIM/issues/new/choose>`_

   Opening an issue creates a record of the bug and gives maintainers and contributors
   a place to troubleshoot, discuss potential fixes, and document workarounds.

#. Write a test that reproduces the bug

   A failing test is the clearest way to demonstrate a bug and verify that your fix works.
   It also guards against the same bug reappearing in the future.

#. Fix the bug and open a PR.

--------------------------
Implementing a new feature
--------------------------

#. `Raise an issue <https://github.com/festim-dev/FESTIM/issues/new/choose>`_

   Before writing any code, open an issue to discuss the feature with the maintainers.
   Someone may already be working on it, or the maintainers may have context that
   shapes the design.

#. :ref:`Make your changes <contributing to the code>` and
   :ref:`update the documentation <Documentation guide>` if needed.

#. Write tests to cover the new feature.

#. Open a PR.

----------------
Code formatting
----------------

FESTIM uses `Black <https://github.com/psf/black>`_ to enforce a consistent code style,
which reduces noise in code reviews and keeps the codebase readable.

Install Black:

.. code-block:: bash

   pip install black

Format a single file:

.. code-block:: bash

   black my_script.py

Format all files in the current directory:

.. code-block:: bash

   black .

Check formatting without modifying any files:

.. code-block:: bash

   black --check .

.. tip::

   If you use Visual Studio Code, install the
   `Black Formatter extension <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_
   and enable **Format on Save** to apply Black automatically whenever you save a file.

-------------------
Documentation guide
-------------------

The documentation is often the first thing a user encounters.
Keeping it accurate, clear, and up to date is as important as keeping the code correct.

The docs are written in
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
and live in the `docs folder <https://github.com/festim-dev/FESTIM/tree/fenicsx/docs>`_.
They are built with `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and hosted on
`Read the Docs <https://readthedocs.org/>`_, which rebuilds automatically on every
commit and pull request.

**Setting up the documentation environment**

.. note::

   Create a dedicated conda environment with all documentation dependencies:

   .. code-block:: bash

      conda env create -f docs/environment.yml
      conda activate festim2-docs

**Building the docs**

From the ``docs/source`` directory:

.. code-block:: bash

   cd docs/source
   make html

Or from the ``docs`` directory in a single step:

.. code-block:: bash

   cd docs
   sphinx-build -b html source build

The generated HTML will be in ``docs/source/_build/html``.
Open ``index.html`` in your browser to preview the result.

To remove all build artefacts and start from scratch:

.. code-block:: bash

   cd docs/source
   make clean

**Running doctests**

Verify that all code examples in the documentation execute correctly:

.. code-block:: bash

   cd docs/source
   make doctest

Or equivalently:

.. code-block:: bash

   cd docs
   sphinx-build -b doctest source build

**Documentation checklist**

When contributing to the documentation, make sure to:

* Write clearly and concisely
* Use correct reStructuredText syntax
* Update existing pages when behaviour changes
* Add documentation for any new features
* Run the doctests to confirm all code examples are correct