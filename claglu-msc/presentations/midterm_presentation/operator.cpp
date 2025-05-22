enum Dim {X, Y, Z};
DiffType DiffX = Centered;

DiffOpChain<Dim::X,DiffX, 
             DiffOpChain<Dim::X,DiffX,FView_t> > diff_xx(FView_t F, Vector_t hInv);

// Call it on an index
std::cout << diff_xx(42,42,42) << std::endl;
