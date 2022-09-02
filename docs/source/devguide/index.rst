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

.. _contributing to the code:

------------------------
Contributing to the code
------------------------

.. admonition:: Tip
   :class: tip

   If you're a beginner, look for `issues tagged with "Good first issue" <https://github.com/RemDelaporteMathurin/FESTIM/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_.

   These issues are usually fairly easy to tackle and therefore perfect for newcomers.

1) `Fork the repository <https://github.com/RemDelaporteMathurin/FESTIM/fork>`_

By forking the repository, you create a copy of it where you can safely make changes.

2) Make your changes
3) `Open a PR <https://github.com/RemDelaporteMathurin/FESTIM/compare>`_
4) Wait for a maintainer to review your PR

Before merging your changes, they have to be reviewed. During the review, we make sure the changes don't break anything and eventually propose/request improvements.
The time before the review will depend on the maintainers' availability.

5) When everything is in order, the maintainers will merge your PR!

----------
Test suite
----------

FESTIM uses continuous integration (CI) to ensure code quality and eliminate as much bugs as possible.

In a nutshell, every time a change is pushed to the repository (or to a PR), a suite of tests is automatically triggered.
This is to make sure the changes don't break existing functionalities.
It is also very useful to catch bugs that developers could have missed.
Click `here <https://www.atlassian.com/continuous-delivery/continuous-integration>`_ for more information on CI.

All the tests can be found in the `test folder <https://github.com/RemDelaporteMathurin/FESTIM/tree/main/test>`_ at the root of the FESTIM repository.

Whenever contributors open a PR, **the tests must pass** in order for the PR to be merged in.

In some cases, new tests need to be written to account for more use cases or to catch bugs that weren't previously caught.

---------
Debugging
---------

When you find a bug in the code, there are several steps to follow to make things easier for maintainers.

#. | `Raise an issue <https://github.com/RemDelaporteMathurin/FESTIM/issues/new/choose>`_
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

#. | `Raise an issue <https://github.com/RemDelaporteMathurin/FESTIM/issues/new/choose>`_
   |
   | Before spending time implementing a new great feature, it is better to open an issue first to discuss with the maintainers.
   | For all you know, someone is already working at implementing it and all your time would be spent for nothing.
   | 
   | It is also beneficial to discuss with the community on how this new feature would be used.

#. :ref:`Make your changes<contributing to the code>`

#. Write a test to test your feature
