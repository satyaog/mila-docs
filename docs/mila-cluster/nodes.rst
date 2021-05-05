Cluster Nodes
##############

.. raw:: html

    <iframe src="https://dashboard.server.mila.quebec/d/DuQNAfQZz/homepage?viewPanel=4&orgId=1" height="345px" width="100%"></iframe>

Complete List of Nodes
^^^^^^^^^^^^^^^^^^^^^^

.. _node_list:


.. role:: h(raw)
   :format: html

..
   Je trouve cela un peu futile de maintenir cette documentation à jour manuellement.
   Peut-être pourrions nous créer dans ce dossier des sripts qui pourraient créer une entrée RST et qui pourraient être exécutés sur un noeud au Mila pour les mises à jour.


+---------------------------------------+----------------------------+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
|                                       |             GPU            |      |         |              |              |             |              |        |                     |
|               Name                    +--------------+-------------+ CPUs | Sockets | Cores/Socket | Threads/Core | Memory (Gb) | TmpDisk (Tb) |  Arch  |       Features      |
|                                       |    Primary   |  Secondary  |      |         |              |              |             |              |        +---------------------+
|                                       +----------+---+---------+---+      |         |              |              |             |              |        | GPU Arch and Memory |
|                                       |   Model  | # |  Model  | # |      |         |              |              |             |              |        |                     |
+=======================================+==========+===+=========+===+======+=========+==============+==============+=============+==============+========+=====================+
| :h:`<h5 style="margin: 5px 0 0 0;">KEPLER</h5>`                                                                                                                               |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| kepler[2-3]                           |    k80   | 8 |         |   |  16  |    2    |       4      |       2      |     256     |      3.6     | x86_64 |      tesla,12GB     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| kepler4                               |    m40   | 4 |         |   |  16  |    2    |       4      |       2      |     256     |      3.6     | x86_64 |     maxwell,24GB    |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| kepler5                               |   v100   | 2 |   m40   | 1 |  16  |    2    |       4      |       2      |     256     |      3.6     | x86_64 |      volta,12GB     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">MILA</h5>`                                                                                                                                 |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| mila01                                |   v100   | 8 |         |   |  80  |    2    |      20      |       2      |     512     |       7      | x86_64 |      tesla,16GB     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| mila02                                |   v100   | 8 |         |   |  80  |    2    |      20      |       2      |     512     |       7      | x86_64 |      tesla,32GB     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| mila03                                |   v100   | 8 |         |   |  80  |    2    |      20      |       2      |     512     |       7      | x86_64 |      tesla,32GB     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">POWER9</h5>`                                                                                                                               |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| power9[1-2]                           |   v100   | 4 |         |   |  128 |    2    |      16      |       4      |     586     |     0.88     | power9 |  tesla,nvlink,16gb  |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">TITAN RTX</h5>`                                                                                                                            |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| rtx[6,9]                              | titanrtx | 2 |         |   |  20  |    1    |      10      |       2      |     128     |      3.6     | x86_64 |     turing,24gb     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| rtx[1-5,7-8]                          | titanrtx | 2 |         |   |  20  |    1    |      10      |       2      |     128     |     0.93     | x86_64 |     turing,24gb     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<h5 style="margin: 5px 0 0 0;">New Compute Nodes</h5>`                                                                                                                    |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<b>cn-a[01-11]</b>`               |  rtx8000 | 8 |         |   |  80  |    2    |      20      |       2      |     380     |      3.6     | x86_64 |      turing,48g     |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<b>cn-b[01-05]</b>`               |   v100   | 8 |         |   |  80  |    2    |      20      |       2      |     380     |      3.6     | x86_64 |  tesla,nvlink,32gb  |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+
| :h:`<b>cn-c[01-40]</b>`               |  rtx8000 | 8 |         |   |  64  |    2    |      32      |       1      |     386     |      3       | x86_64 |     turing,48g      |
+---------------------------------------+----------+---+---------+---+------+---------+--------------+--------------+-------------+--------------+--------+---------------------+



Special Nodes
^^^^^^^^^^^^^^^

Power9
"""""""

.. _power9_nodes:

Power9_ servers are using a different processor instruction set than Intel and AMD (x86_64).
As such you need to setup your environment again for those nodes specifically.

* Power9 Machines have 128 threads. (2 processors / 16 cores / 4 way SMT)
* 4 x V100 SMX2 (16 GB) with NVLink
* In a Power9 machine GPUs and CPUs communicate with each other using NVLink instead of PCIe.

This allow them to communicate quickly between each other. More on LMS_

Power9 have the same software stack as the regular nodes and each software should be included to deploy your environment
as on a regular node.


.. _LMS: https://developer.ibm.com/linuxonpower/2019/05/17/performance-results-with-tensorflow-large-model-support-v2/
.. _Power9: https://en.wikipedia.org/wiki/POWER9

.. .. prompt:: bash $, auto
..
..     # on mila cluster's login node
..     $ srun -c 1 --reservation=power9 --pty bash
..
..     # setup anaconda
..     $ wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-ppc64le.sh
..     $ chmod +x Anaconda3-2019.07-Linux-ppc64le.sh
..     $ module load anaconda/3
..
..     $ conda config --add channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
..     $ conda create -n p9 python=3.6
..     $ conda activate p9
..     $ conda install powerai=1.6.0
..
..     # setup is done!


AMD
""""

.. warning::

    As of August 20 the GPUs had to return back to AMD.
    Mila will get more samples. You can join the amd_ slack channels to get the latest information

.. _amd: https://mila-umontreal.slack.com/archives/CKV5YKEP6/p1561471261000500

Mila has a few node equipped with MI50_ GPUs.

.. _MI50: https://www.amd.com/en/products/professional-graphics/instinct-mi50

.. prompt:: bash $, auto

    $ srun --gres=gpu -c 8 --reservation=AMD --pty bash

    # first time setup of AMD stack
    $ conda create -n rocm python=3.6
    $ conda activate rocm

    $ pip install tensorflow-rocm
    $ pip install /wheels/pytorch/torch-1.1.0a0+d8b9d32-cp36-cp36m-linux_x86_64.whl
