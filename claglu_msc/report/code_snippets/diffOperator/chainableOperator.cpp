constexpr int Dim = 2;

typedef double T;
// Inverse mesh-spacing
ippl::Vector<Dim, T> hInv = {40.0, 40.0};

// Field of type double and size [100]^2
Field<Dim, T> field(100, 100, 1.0 / hInv);

// Define the stencils applied along the x and y dimension
DiffType DiffX = DiffType::Forward;
DiffType DiffY = DiffType::Backward;

// Operator that is applied first
typedef DiffOpChain<OpDim::Y, Dim, T, DiffY, FView_t> firstOperator;
// Operator that is applied after the first
DiffOpChain<OpDim::X, Dim, T, DiffX, firstOperator> diff_xy(field, hInv);

// Compute curvature at index (42,42)
double result = diff_xy(42, 42);
