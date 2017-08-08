/*//////////////////////////////////////////////////////////////////////////////////////////////////
///  HOCR0.h   Higher-Order Clique Reduction helper classes and functions
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

HOCR0.h
This file contains the internal classes and functions used in HOCR.h.
For the usage of the Higher-Order Clique Reduction software, see HOCR.h.

//////////////////////////////////////////////////////////////////////////////////////////////////*/


#ifndef __HOCR0_H__
#define __HOCR0_H__

#include<vector>
#include<algorithm>

// Monomial class of degree D. D is fixed at the compile time
template <typename VID, int D>
class Term
{
public:
	Term(const VID vs[]) {std::copy(vs, vs + D, vars); std::sort(vars, vars + D);}
	Term(const Term& t) {std::copy(t.vars, t.vars + D, vars);}
	bool operator<(const Term& t) const {return std::lexicographical_compare(vars, vars + D, t.vars, t.vars + D);}
	bool operator==(const Term& t) const {return std::equal(vars, vars + D, t.vars);}
	void get(VID vs[]) const {std::copy(vars, vars + D, vs);}
	VID maxID() const {return *std::max_element(vars, vars + D);}
private:
	VID vars[D];
};

// Specialization for degree 1
template <typename VID>
class Term<VID, 1>
{
public:
	Term(VID id) : var(id) {}
	Term(const Term& t) {var = t.var;}
	bool operator<(const Term& t) const {return var < t.var;}
	bool operator==(const Term& t) const {return var == t.var;}
	void get(VID vs[]) const {vs[0] = var;}
	VID maxID() const {return var;}
private:
	VID var;
};

// Specialization for degree 2
template <typename VID>
class Term<VID, 2>
{
public:
	Term(VID id1, VID id2) {if (id1 < id2) {vars[0] = id1; vars[1] = id2;} else {vars[0] = id2; vars[1] = id1;}}
	Term(VID vs[]) {if (vs[0] <  vs[1]) {vars[0] = vs[0]; vars[1] =  vs[1];} else {vars[0] =  vs[1]; vars[1] = vs[0];}}
	Term(const Term& t) {vars[0] = t.vars[0]; vars[1] = t.vars[1];}
	bool operator<(const Term& t) const {return vars[0] < t.vars[0] || (vars[0] == t.vars[0] && vars[1] < t.vars[1]);}
	bool operator==(const Term& t) const {return vars[0] == t.vars[0] && vars[1] == t.vars[1];}
	void get(VID vs[]) const {vs[0] = vars[0]; vs[1] = vars[1];}
	VID maxID() const {return std::max(vars[0], vars[1]);}
private:
	VID vars[2];
};


// Container for monomials of a fixed degree.
template <typename REAL, typename VID, int D>
class Terms
{
	typedef std::vector<std::pair<Term<VID,D>,REAL> > TVec;
public:
	Terms() : current(terms.end()) {}
	void clear() {terms.clear();}
	int size() const {return terms.size();}
	VID maxID() const 
	{
		VID m = 0;
		for (typename TVec::const_iterator i = terms.begin(); i != terms.end(); i++)
			m = std::max(m, (*i).first.maxID());
		return m;
	}
	void add(REAL c, const Term<VID,D>& t) 	{terms.push_back(std::make_pair(t, c));}
	void startEnum() {shrink(); current = terms.end();}
	void shrink() // Sorts the monomials and unifies the monomials with the same variables
	{
		std::sort(terms.begin(), terms.end());
		typename TVec::iterator i1 = terms.begin();
		typename TVec::iterator i2 = i1;
		while (i2 != terms.end())
		{
			if (i1 != i2)
				*i1 = *i2;
			i2++;
			while (i2 != terms.end() && (*i1).first == (*i2).first)
			{
				(*i1).second += (*i2).second;
				i2++;
			}
			if ((*i1).second != 0)
				i1++;
		}
		if (i1 != i2)
			terms.erase(i1, i2);
	}
	bool next()
	{
		if (current == terms.end())
			current = terms.begin();
		else
			current++;
		return current != terms.end();
	}
	void get(int& size, VID vars[], REAL& c) const
	{
		size = D;
		(*current).first.get(vars);
		c = (*current).second;
	}
private:
	TVec terms;
	typename TVec::iterator current;
};


template <typename REAL, typename VID, int MAXD> class TermsUpTo;


// Container of Terms objects of up to degree MAXD.
// MAXD is fixed at the compile time.
template <typename REAL, typename VID, int MAXD>
class TermsUpTo_
{
public:
	TermsUpTo_() : enumthis(true) {}
	void clear() {myterms.clear(); prev.clear();}
	void shrink() {myterms.shrink(); prev.shrink();}
	void startEnum() {myterms.startEnum(); prev.startEnum(); enumthis = true;}
	int size() const {return myterms.size() + prev.size();}
	int size(int cliquesize) const {return (MAXD == cliquesize) ? myterms.size() : prev.size(cliquesize);}
	VID maxID() const {return std::max(myterms.maxID(), prev.maxID());}
	void add(REAL c, int degree, VID vars[]) 
	{
		if (MAXD == degree) 
			myterms.add(c, Term<VID,MAXD>(vars));
		else 
			prev.add(c, degree, vars);
	}
	void add1(REAL c, VID v) {prev.add1(c, v);}
	void add2(REAL c, VID v1, VID v2) {prev.add2(c, v1, v2);}
	bool next()
	{
		if (enumthis)
		{
			if (myterms.next())
				return true;
			enumthis = false;
		}
		return prev.next();
	}
	void get(int& size, VID vars[], REAL& c)
	{
		if (enumthis)
			myterms.get(size, vars, c);
		else
			prev.get(size, vars, c);
	}
protected:
	bool enumthis; // For enumeration
	TermsUpTo<REAL,VID,MAXD-1> prev; // Recursively contains all Terms object down to D = 1
	Terms<REAL,VID,MAXD> myterms; // Contains one Terms object for D = MAXD
};


template <typename REAL, typename VID, int MAXD>
class TermsUpTo : public TermsUpTo_<REAL, VID, MAXD> {}; // Class for MAXD > 2


template <typename REAL, typename VID>
class TermsUpTo<REAL,VID,2> : public TermsUpTo_<REAL, VID, 2> // Specialization for MAXD == 2
{
public:
	void add2(REAL c, VID v1, VID v2) {TermsUpTo_<REAL, VID, 2>::myterms.add(c, Term<VID,2>(v1, v2));}
};


template <typename REAL, typename VID>
class TermsUpTo<REAL,VID,1> // Specialization for MAXD == 1
{
public:
	void clear() {myterms.clear();}
	void startEnum() {myterms.startEnum();}
	void shrink() {myterms.shrink();}
	int size() const {return myterms.size();}
	int size(int cliquesize) const {return (cliquesize == 1) ? myterms.size() : -1;}
	VID maxID() const {return myterms.maxID();}
	void add(REAL c, int degree, VID vars[]) {if (degree == 1) myterms.add(c, Term<VID,1>(vars[0]));}
	void add1(REAL c, VID v) {myterms.add(c, Term<VID,1>(v));}
	bool next() {return myterms.next();}
	void get(int& size, VID vars[], REAL& c) {myterms.get(size, vars, c);}
private:
	Terms<REAL,VID,1> myterms;
};

// Converts higher-order terms to quadratic, if necessary.
template<typename QPBF, typename VID, typename NVID, typename REAL>
void addTermsToQPBF(QPBF& qpbf, int degree, VID* vars, REAL c, NVID& newvar)
{
	if (degree == 0)
		return;
	else if (degree == 1)
		qpbf.add1(c, vars[0]);
	else if (degree == 2)
		qpbf.add2(c, vars[0], vars[1]);
	else if (c < 0)
	{
		for (int i = 0; i < degree; i++)
			qpbf.add2(c, vars[i], newvar);
		qpbf.add1((1 - degree) * c, newvar);
		newvar++;
	}
	else
	{
		int numNewVars = ((degree % 2) ? (degree - 1) : (degree - 2)) / 2;
		for (int i = 1; i <= numNewVars; i++)
		{
			bool b = (degree % 2) && i == numNewVars;
			REAL coef = c * (b ? -1 : -2);
			for (int j = 0; j < degree; j++) // S_1
				qpbf.add2(coef, vars[j], newvar);
			qpbf.add1(c * ((b ? 2 : 4) * i - 1), newvar);
			newvar++;
		}
		for (int i = 0; i < degree - 1; i++) // S_2
			for (int j = i + 1; j < degree; j++)
				qpbf.add2(c, vars[i], vars[j]);
	}
}

inline int matrixElement(int m, int n)
{
	if (~m & n)
		return 0;
	for (m ^= n, n = 0; m; m >>= 1)
		n ^= m & 1;
	return n ? -1 : 1;
}

// Gets the coefficient for the i'th monomial
// From an energy E that has n values
template<typename REAL>
REAL getCoef(int i, int n, REAL E[])
{
	REAL rv = 0;
	for (int j = 0; j < n; j++)
		rv += E[j] * matrixElement(i, j);
	return rv;
}

// Adds higher-order term Eij...k(x_i, x_j, ..., x_k) with cost values 
// E00...00, E00...01, E00...10,..., E11...10, E11...11.
// Note the order of bits very carefully. It is defined consistent with the order
// used in the QPBO software. E00...01 is the energy when only the LAST variable,
// not the first, is 1.
template<typename PBF, typename VID, typename REAL>
void AddTerms(PBF& pbf, int cliquesize, VID vars[], REAL E[])
{ // vars is of size 'cliquesize'; E is of size 2^cliquesize
	int tn = 1 << cliquesize;
	for (int ix = 0; ix < tn; ix++)
	{
		REAL c = getCoef(ix, tn, E);
		if (c == 0)
			continue;
		VID vs[PBF::MaxDegree];
		int j = 0, o = 0, b = 1;
		for (int i = 0; i < cliquesize; i++)
		{
			if (ix & b)
			{
				vs[j++] = vars[cliquesize - 1 - i];
				o++;
			}
			b <<= 1;
		}
		pbf.add(c, o, vs);
	}
}




#endif // __HOCR0_H__

