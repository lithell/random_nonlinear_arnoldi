# Random Nonlinear Arnoldi

This project is an implementation of the randomized nonlinear Arnoldi algorithm. It aims to demonstrate the algorithm and makes no claim to being optimized or fast.

## Usage 

This project offers two main functions. The first is the 'NLA' function, which is an implementation of the Nonlinear Arnoldi algorithm of Voss. The second is the function 'sNLA', i.e. sketched nonlinear Arnolid, which is the main result of this work. Much of the functionality of this project is implemented to use the framework in the NEP-PACK library of eigensolvers, and so depends on this package. 

To use the functions, define your NEP as in NEP-PACK, and then you should be good to go. 'sNLA' requires some additonal functionality. For instance, the user needs to define a sketching function. In this work we have employed a SRCT-embedding, but 'sNLA' should in theory work with any other valid sketching function.
