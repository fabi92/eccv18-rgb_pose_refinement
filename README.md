This code accompanies the paper

**Fabian Manhardt, Wadim Kehl, Nassir Navab and Federico Tombari: Deep Model-Based 6D Pose Refinement in RGB. ECCV 2018.**[(PDF)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Fabian_Manhardt_Deep_Model-Based_6D_ECCV_2018_paper.pdf)


and allows to reproduce parts of our work. Note that due to IP issues we can only provide our trained networks and the inference part. This allows to refine 6D hypotheses from monocular data.

**Unfortunately, the code for training cannot be made available.**

In order to use the code, you need to downloa the used datasets (hinterstoisser, tejani) in SIXD format (e.g. from [here](http://cmp.felk.cvut.cz/sixd/challenge_2017/) ) and use the test_refinement.py script to do the magic. Invoke 'python3 test_refinement.py --help' to see the available commands.

Note that we only provide the network for obj_02 of the LineMOD dataset. The other networks will be also released soon.
