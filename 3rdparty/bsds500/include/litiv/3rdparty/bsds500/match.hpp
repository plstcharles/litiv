
#ifndef __match_hh__
#define __match_hh__

namespace BSDS500 {

    class Matrix;

    // returns the cost of the assignment
    double matchEdgeMaps (
        const Matrix& bmap1, const Matrix& bmap2,
        double maxDist, double outlierCost,
        Matrix& match1, Matrix& match2);

} // namespace BSDS500

#endif // __match_hh__
