/*//////////////////////////////////////////////////////////////////////////////////////////////////
///  HOCR.h    Higher-Order Clique Reduction software
///  Version 1.02
////////////////////////////////////////////////////////////////////////////////////////////////////

Copyright 2009-2011 Hiroshi Ishikawa.
This software can be used for research purposes only.
This software or its derivatives must not be publicly distributed
without a prior consent from the author (Hiroshi Ishikawa).

THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For the latest version, check: http://www.f.waseda.jp/hfs/

////////////////////////////////////////////////////////////////////////////////////////////////////

Software for minimizing a higher-order function of binary variables x_1,...,x_n.
What it actually does is to reduce the function into a first-order MRF, or a 
Quadratic Pseudo-Boolean function, i.e., a function of the form
E(x_1, ..., x_n, ..., x_m) = \sum_i Ei(x_i) + \sum_{ij} Eij(x_i,x_j),
on which algorithms such as QPBO and BP can be used.
The additional variables are added to reduce the order of the energy.
The number of variables increases exponentially as the order of the
given energy increases.

The technique in this software is described in the following paper:

Hiroshi Ishikawa. "Higher-Order Clique Reduction in Binary Graph Cut."
In CVPR2009: IEEE Computer Society Conference on Computer Vision and Pattern Recognition,
Miami Beach, Florida. June 20-25, 2009.

An extended version of the paper appeared as:

Hiroshi Ishikawa. "Transformation of General Binary MRF Minimization to the First Order Case."
IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 33, no. 6, pp. 1234-1249,
June 2011.

////////////////////////////////////////////////////////////////////////////////////////////////////

This software is implemented so that it can be used most conveniently 
in combination with the QPBO software by Vladimir Kolmogorov available
at http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/software.html

The code was tested on Microsoft Visual C++ 2008 Express Edition SP1
and gcc version 4.1.2.
Any report on bugs and results of trying on other platforms is appreciated.

////////////////////////////////////////////////////////////////////////////////////////////////////

	
	
Example usage:
Minimize E(x, y, z, w) = x + 4y - z - 2w(y-1) + xy(z+1) - xw(y+1)(z+1), where x,y,z,w are in {0,1}.

There are two ways.

1. Reduce the higher-order terms as you add terms. The reduced quadratic terms
	are stored in the QPBO object.
    If the same higher-order monomial appears later, another new variable is created.

#include "HOCR/HOCR.h"
#include "QPBO/QPBO.h"
void main()
{
	typedef int REAL;
	QPBO<REAL> qpbo(8,20);
	HOCR<REAL,4,QPBO<REAL> > hocr(qpbo);
	hocr.AddNode(4); // Add four variables.  Variable indices are x: 0, y: 1, z: 2, w: 3
	hocr.AddUnaryTerm(0, 0, 1); // Add the term x
	hocr.AddUnaryTerm(1, 0, 4); // Add the term 4y
	hocr.AddUnaryTerm(2, 0, -1); // Add the term -z
	hocr.AddPairwiseTerm(1, 3, 0, 2, 0, 0); // Add the term -2w(y-1)
	int vars3[3] = {0,1, 2};
	REAL vals3[8] = {0, 0, 0, 0, 0, 0, 1, 2};
	hocr.AddHigherTerm(3, vars3, vals3); // // Add the term  xy(z+1)
	int vars4[4] = {0,1, 2, 3};
	REAL vals4[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -2, 0, -2, 0, -4};
	hocr.AddHigherTerm(4, vars4, vals4); // // Add the term  -xw(y+1)(z+1)
	qpbo.MergeParallelEdges();
	qpbo.Solve();
	qpbo.ComputeWeakPersistencies();
	int x = qpbo.GetLabel(0);
	int y = qpbo.GetLabel(1);
	int z = qpbo.GetLabel(2);
	int w = qpbo.GetLabel(3);
	printf("Solution: x=%d, y=%d, z=%d, w=%d\n", x, y, z, w);
}


2. Use the Pseudo-Boolean function object to build the energy function object,
    Convert to quadratic, then convert it to QPBO object.
	Better when the same higher-order term can appear later in the course of
	adding terms.

#include "HOCR/HOCR.h"
#include "QPBO/QPBO.h"
void main()
{
	typedef int REAL;
	PBF<REAL,4> pbf;
	pbf.AddUnaryTerm(0, 0, 1); // Add the term x
	pbf.AddUnaryTerm(1, 0, 4); // Add the term 4y
	pbf.AddUnaryTerm(2, 0, -1); // Add the term -z
	pbf.AddPairwiseTerm(1, 3, 0, 2, 0, 0); // Add the term -2(y-1)w
	int vars3[3] = {0,1, 2};
	REAL vals3[8] = {0, 0, 0, 0, 0, 0, 1, 2};
	pbf.AddHigherTerm(3, vars3, vals3); // Add the term  xy(z+1)
	int vars4[4] = {0,1, 2, 3};
	REAL vals4[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -2, 0, -2, 0, -4};
	pbf.AddHigherTerm(4, vars4, vals4); // Add the term  -xw(y+1)(z+1)
	PBF<REAL,2> qpbf;
	pbf.toQuadratic(qpbf); // Reduce to Quadatic pseudo-Boolean function
	pbf.clear(); // free memory
	int numvars = qpbf.maxID(); // Number of variables
	QPBO<int> qpbo(numvars, numvars * 4);
	convert(qpbo, qpbf); // copy to QPBO object by V. Kolmogorov
	qpbf.clear(); // free memory
	qpbo.MergeParallelEdges();
	qpbo.Solve();
	qpbo.ComputeWeakPersistencies();
	int x = qpbo.GetLabel(0);
	int y = qpbo.GetLabel(1);
	int z = qpbo.GetLabel(2);
	int w = qpbo.GetLabel(3);
	printf("Solution: x=%d, y=%d, z=%d, w=%d\n", x, y, z, w);
}

//////////////////////////////////////////////////////////////////////////////////////////////////*/


#ifndef __HOCR_H__
#define __HOCR_H__

#include "HOCR0.h" // internal classes and functions

////////////////////////////////////////////////////////////////////////////////////////////////////
// class HOCR: Higher-Order Clique Reduction
// An interface class that contains a reference to an optimization object with
// the same interface as the QPBO object by V. Kolmogorov
// REAL: type for coefficients
// MAXD: maximum degree
// OPTIMIZER: An optimizer class. The required interfaces are
// AddNode, AddUnaryTerm, and AddPairwiseTerm. See QPBO.h

template <typename REAL, int MAXD, typename OPTIMIZER>
class HOCR
{
public:
	typedef int VID; // variable id
	static const int MaxDegree = MAXD;

	// Constructor. 
	HOCR(OPTIMIZER& o) : optimizer(o), newvar(-1) {}

	// Adds node(s) to the graph. See QPBO.h.
	VID AddNode(int num = 1) {return optimizer.AddNode(num);}

	// Adds unary term Ei(x_i) with cost values Ei(0)=E0, Ei(1)=E1. See QPBO.h.
	void AddUnaryTerm(VID i, REAL E0, REAL E1) {optimizer.AddUnaryTerm(i, E0, E1);}

	// Adds pairwise term Eij(x_i, x_j) with cost values E00, E01, E10, E11. See QPBO.h.
	void AddPairwiseTerm(VID i, VID j, REAL E00, REAL E01, REAL E10, REAL E11)
	{
		optimizer.AddPairwiseTerm(i, j, E00, E01, E10, E11);
	}

	// Adds higher-order term Eij...k(x_i, x_j, ..., x_k) with cost values 
	// E00...00, E00...01, E00...10,..., E11...10, E11...11.
	// Note the order of bits very carefully. It is defined consistent with the order
	// used in the QPBO software. E00...01 is the energy when only the LAST variable v_k,
	// not the first, is 1.
	// vars is of size 'cliquesize', E is of size 2^cliquesize
	void AddHigherTerm(int cliquesize, VID vars[], REAL E[]) {AddTerms(*this, cliquesize, vars, E);}


	// Adds a monomial. Used in AddTerms.
	void add(REAL c, int degree, VID vars[]) {if (degree > 2) (*this)++; addTermsToQPBF(*this, degree, vars, c, *this);}
	void add1(REAL c, VID v) {optimizer.AddUnaryTerm(v, 0, c);}
	void add2(REAL c, VID v1, VID v2) {optimizer.AddPairwiseTerm(v1, v2, 0, 0, 0, c);}

// interfacees for use in addTermsToQPBF
	operator VID() const {return newvar;}
	void operator++(int) {newvar = optimizer.AddNode(1);}
private:
	OPTIMIZER& optimizer;
	VID newvar;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
// class PBF: Pseudo-Boolean Function
// Represents a pseud-Boolean function. Includes a reduction to quadratic pbf.
// REAL: type for coefficients
// MAXD: maximum degree

template <typename REAL, int MAXD>
class PBF
{
public:
	typedef int VID; // variable id
	static const int MaxDegree = MAXD;

	PBF() : constant(0) {}

	// clears to free memory
	void clear() {terms.clear();}

	// Adds unary term Ei(x_i) to the energy function with cost values Ei(0)=E0, Ei(1)=E1.
	// This adds the terms  E0(1 - x_i) + E1 x_i 
	void AddUnaryTerm(VID i, REAL E0, REAL E1) {constant += E0; add1(E1 - E0, i);}

	// Adds pairwise term Eij(x_i, x_j) with cost values E00, E01, E10, E11.
	// This adds the terms  E00(1-x_i)(1-x_j) + E01(1-x_i)x_j + E10 x_i(1-x_j) + E11 x_i x_j 
	void AddPairwiseTerm(VID i, VID j, REAL E00, REAL E01, REAL E10, REAL E11)
	{
		constant += E00;
		add1(E10 - E00, i);
		add1(E01 - E00, j);
		add2(E00 - E01 - E10 + E11, i, j);
	}

	// Adds higher-order term Eij...k(x_i, x_j, ..., x_k) with cost values 
	// E00...00, E00...01, E00...10,..., E11...10, E11...11.
	// Note the order of bits very carefully. It is defined consistent with the order
	// used in the QPBO software. E00...01 is the energy when only the LAST variable,
	// not the first, is 1.
	// vars is of size 'cliquesize'; E is of size 2^cliquesize
	void AddHigherTerm(int cliquesize, VID vars[], REAL E[]) {AddTerms(*this, cliquesize, vars, E);}

	// Adds a monomial. These just add it at the end of the container.
	// Use shrink() before using the PBF.
	void add(REAL c, int degree, VID vars[]) {if (degree == 0) constant += c; else terms.add(c, degree, vars);}
	void add1(REAL c, VID v) {terms.add1(c, v);}
	void add2(REAL c, VID v1, VID v2) {terms.add2(c, v1, v2);}

	// Monomial enumeration functions
	// Usage: startEnum(); while (get(degree, vars, coef)) { do something with it. }

	// Initializes an enumeration
	void startEnum() {terms.startEnum();}

	// Gets the data for one monomial: the degree, variable IDs, and the coefficient.
	// Returns false if the enumeration is at the end.
	bool get(int& degree, VID vars[], REAL& coef) 
	{
		if (!terms.next())
			return false;
		terms.get(degree, vars, coef);
		return true;
	}

	// Sort and merge the terms
	void shrink() {terms.shrink();}

	// Returns the total number of monomials, excluding the constant term.
	// Accurate only after shrink() and before add().
	int size() const {return terms.size();}

	// Returns the number of monomials of degree d. (d>=1)
	// Accurate only after shrink() and before add(). 
	int size(int d) const {return terms.size(d);}

	// Returns the maximum variable ID used.
	VID maxID() const {return terms.maxID();}

	// Reduces this PBF into a qpbf. Returns the variable ID to use for the next new variable.
	VID toQuadratic(PBF<REAL, 2>& qpbf) {return toQuadratic(qpbf, maxID() + 1);}
	VID toQuadratic(PBF<REAL, 2>& qpbf, int newvar)
	{
		int degree;
		VID vars[MAXD];
		REAL c;
		terms.startEnum();
		while (get(degree, vars, c))
			addTermsToQPBF(qpbf, degree, vars, c, newvar);
		qpbf.add(constant, 0, vars); // Thanks to Petter Strandmark
		return newvar;
	}

	// Returns the constant term.
	REAL cnst() const {return constant;}

private:
	REAL constant;
	TermsUpTo<REAL,VID,MAXD> terms;
};


////////////////////////////////////////////////////////////////////////////////////////////////////
// Converts a quadratic pseudo-Boolean function object (PBF<REAL,2>) to an
// optimization object with the same interface as the QPBO object by V. Kolmogorov
// REAL: can be int, float, double.
// OPTIMIZER: An optimizer class. The required interfaces are
// AddNode, AddUnaryTerm, and AddPairwiseTerm. See QPBO.h

template<typename REAL, typename OPTIMIZER>
void convert(OPTIMIZER& optimization, PBF<REAL,2>& qpbf)
{
	optimization.AddNode(qpbf.maxID() + 1);
	typename PBF<REAL,2>::VID vars[2];
	REAL c;
	int size;
	qpbf.startEnum();
	while (qpbf.get(size, vars, c))
	{
		if (size == 1)
			optimization.AddUnaryTerm(vars[0], 0, c);
		else
			optimization.AddPairwiseTerm(vars[0], vars[1], 0, 0, 0, c);
	}
	optimization.AddUnaryTerm(0, qpbf.cnst(), qpbf.cnst()); // Thanks to Petter Strandmark
}




#endif // __HOCR_H__

