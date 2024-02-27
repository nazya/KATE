# Remove that Square Root: a New Efficient Scale-Invariant Version of AdaGrad

This repository documents the code to reproduce the experiments reported in the paper:
> Remove that Square Root: a New Efficient Scale-Invariant Version of AdaGrad

In this work, we introduce a novel optimization algorithm called KATE, a scale invariant adaptation of AdaGrad. Here we provide a screenshot of KATE's pseudocode from the paper.

![KATE pseudocode](image/KATE_pseudocode.png)

In this repository we compare the performance of KATE with well-known algorithms like AdaGrad anbd ADAM on logistic regression, image classification and text classification problems. If you use this code for your research, please cite the paper as follow

```

```

## Table of Contents
=================

<!--ts-->
   * [Installation](#installation)
   * [Usage](#usage)
      * [STDIN](#stdin)
      * [Local files](#local-files)
      * [Remote files](#remote-files)
      * [Multiple files](#multiple-files)
      * [Combo](#combo)
      * [Auto insert and update TOC](#auto-insert-and-update-toc)
      * [GitHub token](#github-token)
      * [TOC generation with Github Actions](#toc-generation-with-github-actions)
   * [Tests](#tests)
   * [Dependency](#dependency)
   * [Docker](#docker)
     * [Local](#local)
     * [Public](#public)
<!--te-->


## Requirements
```setup
conda env create -f environment.yml
```

## Notebooks
To reproduce the results of the paper, run the train notebook, then run corresponding plot notebook.
